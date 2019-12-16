from numpy import s_, roll, dstack, mean, sqrt, sum, float64, float32, finfo
from skimage.transform import resize

def mse_metrics(pred_img, gt_img):

    return mean((pred_img - gt_img) ** 2)

def normalized_cross_correlation(img1, img2):

    return sum(img1 * img2) / sqrt(sum(img1 ** 2) * sum(img2 ** 2))

def align(img, g_coord):

    h, w = img.shape
    h //= 3
    init_blue = img[:h, :]
    init_green = img[h:2*h, :]
    init_red = img[2*h:3*h, :]

    dy = int(h * 0.05)
    dx = int(w * 0.05)
    init_red = init_red[dy:-dy, dx:-dx]
    init_green = init_green[dy:-dy, dx:-dx]
    init_blue = init_blue[dy:-dy, dx:-dx]

    scale = 1
    h2, w2 = h1, w1 = init_green.shape
    while max(h1, w1) > 320 or scale < 8:
        h1 //= 2
        w1 //= 2
        scale *= 2

    shift = scale
    r_drow = r_dcol = b_drow = b_dcol = 0
    while scale > 0:
        red = resize(init_red, (h1, w1), anti_aliasing=False).astype(float32)
        green = resize(init_green, (h1, w1), anti_aliasing=False).astype(float32)
        blue = resize(init_blue, (h1, w1), anti_aliasing=False).astype(float32)

        r_i_indexes = range(2 * r_drow - shift, 2 * r_drow + shift + 1)
        r_j_indexes = range(2 * r_dcol - shift, 2 * r_dcol + shift + 1)
        b_i_indexes = range(2 * b_drow - shift, 2 * b_drow + shift + 1)
        b_j_indexes = range(2 * b_dcol - shift, 2 * b_dcol + shift + 1)

        r_mse = b_mse = finfo(float64).min

        for i in r_i_indexes:
            for j in r_j_indexes:
                if i >= 0 and j >= 0:
                    tmp = normalized_cross_correlation(red[i:, j:], green[:h1 - i, :w1 - j])
                elif i < 0 and j < 0:
                    tmp = normalized_cross_correlation(red[:h1 + i, :w1 + j], green[-i:, -j:])
                elif i >= 0 and j < 0:
                    tmp = normalized_cross_correlation(red[i:, :w1 + j], green[:h1 - i, -j:])
                else:
                    tmp = normalized_cross_correlation(red[:h1 + i, j:], green[-i:, :w1 - j])
                if tmp > r_mse:
                    r_mse, r_drow, r_dcol = tmp, i, j

        for i in b_i_indexes:
            for j in b_j_indexes:
                if i >= 0 and j >= 0:
                    tmp = normalized_cross_correlation(blue[i:, j:], green[:h1 - i, :w1 - j])
                elif i < 0 and j < 0:
                    tmp = normalized_cross_correlation(blue[:h1 + i, :w1 + j], green[-i:, -j:])
                elif i >= 0 and j < 0:
                    tmp = normalized_cross_correlation(blue[i:, :w1 + j], green[:h1 - i, -j:])
                else:
                    tmp = normalized_cross_correlation(blue[:h1 + i, j:], green[-i:, :w1 - j])
                if tmp > b_mse:
                    b_mse, b_drow, b_dcol = tmp, i, j

        scale //= 2
        shift = 1
        h1 *= 2
        w1 *= 2

    (r_row, r_col) = (g_coord[0] + r_drow + h, g_coord[1] + r_dcol)
    (b_row, b_col) = (g_coord[0] + b_drow - h, g_coord[1] + b_dcol)

    bgr_img = dstack((roll(roll(init_red, -r_dcol, axis=1), -r_drow, axis=0),
                     init_green,
                     roll(roll(init_blue, -b_dcol, axis=1), -b_drow, axis=0)))

    return bgr_img, (b_row, b_col), (r_row, r_col)