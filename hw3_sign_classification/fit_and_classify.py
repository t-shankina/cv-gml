from numpy import array, sqrt, arctan2, zeros, float32, asarray
from math import pi, floor, ceil
from scipy.ndimage import convolve, gaussian_filter
from skimage.transform import resize
from skimage.exposure import adjust_gamma
from skimage.color import rgb2gray
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

def extract_hog(img):

    pattern = img.astype(float32)
    pattern = rgb2gray(pattern)
    b_x = ceil(pattern.shape[0] * 0.17)
    b_y = ceil(pattern.shape[1] * 0.17)
    pattern = pattern[b_x:-b_x, b_y:-b_y]
    pattern = gaussian_filter(pattern, sigma=0.7)
    pattern = resize(pattern, (32,32))
    pattern = adjust_gamma(pattern, gamma=2)

    sobel_x = array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    ix = convolve(pattern, sobel_x)
    iy = convolve(pattern, sobel_y)
    abs_grad = sqrt(ix ** 2 + iy ** 2)
    angles = arctan2(ix, iy) # (-pi, pi]
    angles[angles < 0.0] = pi + angles[angles < 0.0] # [0, pi]
    angles[angles == pi] = angles[angles == pi] - pi # [0, pi) - unsigned gradient

    cellRows = 8
    cellCols = 8
    binCount = 9
    part = pi/ binCount
    h, w = angles.shape
    h1 = h // cellRows
    w1 = w // cellCols
    cells = zeros(shape=(h1, w1, binCount))
    for iCell in range(h1):
        for jCell in range(w1):
            for i in range(iCell * cellRows, (iCell+1) * cellRows):
                for j in range(jCell * cellCols, (jCell+1) * cellCols):
                    bin1 = int(floor(angles[i, j] / part))
                    if (angles[i, j] > (bin1 + 0.5) * part):
                        bin2 = (bin1 + 1) % binCount
                        cells[iCell, jCell, bin1] += (1.0 - (angles[i, j] - (bin1 + 0.5) * part) / part) * abs_grad[i, j]
                        cells[iCell, jCell, bin2] += (angles[i, j] - (bin1 + 0.5) * part) / part * abs_grad[i, j]
                    elif (angles[i, j] < (bin1 + 0.5) * part):
                        bin2 = (bin1 - 1) % binCount
                        cells[iCell, jCell, bin1] += (1.0 - ((bin1 + 0.5) * part - angles[i, j]) / part) * abs_grad[i, j]
                        cells[iCell, jCell, bin2] += ((bin1 + 0.5) * part - angles[i, j]) / part * abs_grad[i, j]
                    else:
                        cells[iCell, jCell, bin1] += abs_grad[i, j]

    blockRowCell = 2
    blockColCell = 2
    eps = 10 ** -9
    h2 = h1 - blockRowCell + 1
    w2 = w1 - blockColCell + 1
    vLen = binCount * blockRowCell * blockColCell
    blocks = zeros(shape=(h2, w2, vLen))
    for iBlock in range(h2):
        for jBlock in range(w2):
            v = list()
            for i in range(iBlock, iBlock+blockRowCell):
                for j in range(jBlock, jBlock+blockColCell):
                    v.extend(cells[i, j])
            v = asarray(v)
            blocks[iBlock, jBlock] = v / sqrt(sum(v ** 2) + eps)

    hog = blocks.reshape(vLen * h2 * w2)

    return hog

def fit_and_classify(train_features, train_labels, test_features):

    clf = SVC(kernel='linear')

    # skf = StratifiedKFold(n_splits=3, random_state=0)
    # scores = cross_val_score(clf, train_features, train_labels, cv=3)
    # print(scores)
    # pass

    clf.fit(train_features, train_labels)
    test_labels = clf.predict(test_features)

    return test_labels