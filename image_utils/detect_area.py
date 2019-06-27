import cv2
import numpy as np


def show_image(name, image, verbose_delay=0):
    if image is None:
        return
    cv2.imshow(name, image)
    cv2.waitKey(verbose_delay)


def crop(image, y, x, h, w):
    return image[y:y + h, x:x + w]


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def coordinates_to_yxhw(coordinates):
    max_x = 0
    max_y = 0
    min_x = 0
    min_y = 0
    for i in range(len(coordinates)):
        x = coordinates[i, 0]
        y = coordinates[i, 1]
        (min_x, min_y) = (x, y) if x + y < min_x + min_y else (min_x, min_y)
        (max_x, max_y) = (x, y) if x + y > max_x + max_y else (max_x, max_y)

    return min_y, min_x, max_y - min_y, max_x - min_x


def find_contour(img, area, threshold_type=1, verbose=0, mina=0, maxa=1000000, minp=0, maxp=4000):
    res = None
    (y, x, h, w) = area

    if h > 0:
        img_crop = crop(img, y, x, h, w)
    else:
        img_crop = img.copy()

    if verbose >= 2:
        show_image('crop', img_crop)

    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    if verbose >= 2:
        show_image('gray', gray)

    ret, thresh = cv2.threshold(gray, 127, 255, type=threshold_type)
    contours, h = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if verbose > 1:
        print('N of contours={}'.format(len(contours)))

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        # pprox = cv2.approxPolyDP(cnt, 0.1 * perimeter, True)
        # if minn <= len(approx) <= maxn
        carea = cv2.contourArea(cnt)
        if mina <= carea <= maxa and minp <= perimeter <= maxp:
            # res = True
            if verbose >= 2:
                cv2.drawContours(gray, [cnt], 0, (0, 255, 0), 2)
                show_image('gray', gray)

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            yxhw = coordinates_to_yxhw(np.array(box))
            res = yxhw[0] + area[0], yxhw[1] + area[1], yxhw[2], yxhw[3]
            if verbose > 1:
                print('yxhw={}, area={}, perimeter={}'.format(res, carea, perimeter))
    return res
