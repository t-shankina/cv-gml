from numpy import zeros, ndarray, transpose, argmin

def find_min_seam(img, mask):

    h, w = img.shape[:2]
    intensity = 0.299 * img[:,:, 0] + 0.587 * img[:,:, 1] + 0.114 * img[:,:, 2]

    energy = zeros(shape=(h,w))
    for y in range(h):
        for x in range(w):
            if x != 0 and x != w-1:
                dx = intensity[y, x+1] - intensity[y, x-1]
            elif x==0:
                dx = intensity[y,1] - intensity[y,0]
            else:
                dx = intensity[y, w-1] - intensity[y, w-2]

            if y != 0 and y != h-1:
                dy = intensity[y+1, x] - intensity[y-1, x]
            elif y==0:
                dy = intensity[1,x] - intensity[0,x]
            else:
                dy = intensity[h-1, x] - intensity[h-2, x]

            energy[y, x] = (dx ** 2 + dy ** 2) ** 0.5

    delta = w * h * 256
    energy += mask * delta

    for y in range(1,h):
        for x in range(w):
            if x != 0 and x != w-1:
                energy[y, x] += min(energy[y-1, x-1], energy[y-1, x], energy[y-1, x+1])
            elif x == 0:
                energy[y, x] += min(energy[y-1, 0], energy[y-1, 1])
            else:
                energy[y, x] += min(energy[y-1, w-2], energy[y-1, w-1])

    min_seam_mask = zeros(shape=(h,w))
    x = argmin(energy[-1, :])
    min_seam_mask[-1, x] = 1
    for y in range(h-2, -1, -1):
        tmp_min = x
        if x != 0 and energy[y, x-1] <= energy[y,x]:
            tmp_min = x-1
        if x != w-1 and energy[y, x+1] < energy[y, tmp_min]:
            tmp_min = x+1
        x = tmp_min
        min_seam_mask[y, x] = 1

    return min_seam_mask

def shrink(img, mask, min_seam_mask):

    h, w = img.shape[:2]
    resized_img = ndarray(shape=(h, w-1, 3))
    resized_mask = ndarray(shape=(h, w-1))

    for y in range(h):
        tmp_x = 0
        for x in range(w):
            if min_seam_mask[y, x] == 1:
                continue
            resized_img[y, tmp_x] = img[y, x]
            resized_mask[y, tmp_x] = mask[y, x]
            tmp_x += 1

    return resized_img, resized_mask

def expand(img, mask, min_seam_mask):
    h, w = img.shape[:2]
    resized_img = ndarray(shape=(h, w+1, 3))
    resized_mask = ndarray(shape=(h, w+1))
    for y in range(h):
        tmp_x = 0
        for x in range(w):
            resized_img[y, tmp_x] = img[y, x]
            resized_mask[y, tmp_x] = mask[y, x]
            tmp_x += 1
            if min_seam_mask[y, x] == 1:
                resized_mask[y, tmp_x] = 1
                if x != w-1:
                    resized_img[y, tmp_x] = (img[y, x] + img[y, x+1]) // 2
                else:
                    resized_img[y, tmp_x] = img[y, x]
                tmp_x += 1

    return resized_img, resized_mask

def seam_carve(img, mode, mask = None):

    mask_is_none = mask is None
    if mask_is_none:
        mask = zeros(shape=img.shape[0:2])

    if 'vertical' in mode:
        img = transpose(img, (1, 0, 2))
        mask = transpose(mask)

    min_seam_mask = find_min_seam(img, mask)

    if 'shrink' in mode:
        resized_img, resized_mask = shrink(img, mask, min_seam_mask)
    else:
        resized_img, resized_mask = expand(img, mask, min_seam_mask)

    if 'vertical' in mode:
        resized_img = resized_img.transpose(1, 0, 2)
        resized_mask = transpose(resized_mask)
        min_seam_mask = transpose(min_seam_mask)

    if mask_is_none:
        resized_mask = None

    return resized_img, resized_mask, min_seam_mask