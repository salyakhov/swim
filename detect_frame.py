import cv2
import numpy as np
import argparse
import math
import datetime
import pytesseract


def show_image(name, image):
    # height, width, channels = img.shape
    if image is None:
        return
    cv2.imshow(name, image)
    cv2.waitKey(0)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-v", "--verbose", required=False, help="show debug info", action="count")
ap.add_argument("-minp", "--minperimeter", required=False, help="minimum figure perimeter", default=0)
ap.add_argument("-maxp", "--maxperimeter", required=False, help="maximum figure perimeter", default=100000)
ap.add_argument("-mina", "--minarea", required=False, help="minimum figure area")
ap.add_argument("-maxa", "--maxarea", required=False, help="maximum figure area")
ap.add_argument("-minn", "--minpolygonvertices", required=False, help="min number of vertices in polygon", default=4)
ap.add_argument("-maxn", "--maxpolygonvertices", required=False, help="max number of vertices in polygon", default=4)
ap.add_argument("-a", "--area", required=False, help="'y, x, h, w'")
ap.add_argument("-sf", "--saveframe", required=False, help="save good frame", default=False)
ap.add_argument("-l", "--lang", required=False, help="language: rus, eng", default='rus')
ap.add_argument("-d", "--dst", required=False, help="try 0 if 1 which is default doesn't work", default=1, type=int)
ap.add_argument("-b", "--blur", required=False, help="Blur image before text extraction", action='store_true')

args = vars(ap.parse_args())

minp = int(args["minperimeter"])
maxp = int(args["maxperimeter"])
mina = int(args["minarea"])
maxa = int(args["maxarea"])
minn = int(args["minpolygonvertices"])
maxn = int(args["maxpolygonvertices"])
is_save_frame = args["saveframe"]
lang = args["lang"]
dst = args["dst"]
do_blur = args["blur"]


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


px = 0
py = 0
ph = 0
pw = 0
if args["area"]:
    a = args["area"].split(",")
    py = int(a[0])
    px = int(a[1])
    ph = int(a[2])
    pw = int(a[3])

src = args["image"]
is_video = True if src.endswith('.mp4') else False

verbose = True if args["verbose"] and args["verbose"] > 0 else False
verbose2 = True if verbose and args["verbose"] > 1 else False


def extract_text(image, language):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_to_text = gray
    if do_blur:
        (thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        blurred = cv2.GaussianBlur(im_bw, (5, 5), 1)
        img_to_text = blurred
        if verbose2:
            show_image('txt', blurred)

    txt = pytesseract.image_to_string(img_to_text, lang=language)
    #txt = pytesseract.image_to_string(im_bw, lang=language)
    #txt = pytesseract.image_to_string(gray, lang=language)
    return txt


def sub_region(img, y, x, h, w):
    crop_img = img[y:y + h, x:x + w]
    if verbose2:
        show_image('crop', crop_img)
    return crop_img


def check_image(img, area):
    res = None
    (y, x, h, w) = area

    # height, width, channels = img.shape
    # print('h={}, w={}, ch={}, area={}'.format(height, width, channels, area))

    if h > 0:
        img_crop = sub_region(img, y, x, h, w)
    else:
        img_crop = img.copy()

    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    if verbose2:
        show_image('gray', gray)

    ret, thresh = cv2.threshold(gray, 127, 255, dst)
    contours, h = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if verbose:
        print('N of contours={}'.format(len(contours)))
    #if verbose2:
    #    cv2.drawContours(img, contours, 0, (0, 0, 255), 2)
    #    show_image('tmp', img)

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        # pprox = cv2.approxPolyDP(cnt, 0.1 * perimeter, True)
        # if minn <= len(approx) <= maxn
        carea = cv2.contourArea(cnt)
        if mina <= carea <= maxa and minp <= perimeter <= maxp:
            # res = True
            if verbose2:
                #print([cnt], cv2.isContourConvex(cnt))
                cv2.drawContours(gray, [cnt], 0, (0, 255, 0), 2)
                show_image('gray', gray)

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            yxhw = coordinates_to_yxhw(np.array(box))
            res = yxhw[0] + area[0], yxhw[1] + area[1], yxhw[2], yxhw[3]
            if verbose:
                print('yxhw={}, area={}, perimeter={}'.format(res, carea, perimeter))
            # if verbose2:
            #    cv2.drawContours(img_crop, [box], 0, (0, 0, 255), 2)
    return res


def check_video(src):
    print(src)
    cap = cv2.VideoCapture(src)
    frame_count = cap.get(7)
    print(frame_count)
    frame_rate = cap.get(5)  # frame rate

    duration = frame_count / frame_rate
    # minutes = int(duration/60)
    # seconds = duration%60
    t = str(datetime.timedelta(seconds=duration))
    # print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
    print(t)

    i = 0
    to_skip = 0
    # while(cap.isOpened() and i < 540):
    while cap.isOpened():
        frame_id = cap.get(1)  # current frame number
        if verbose and frame_id % 1000 == 0:
            t = str(datetime.timedelta(seconds=i))
            print('debug: frame_id={}, t={}'.format(frame_id, t))

        ret, frame = cap.read()
        if not ret:
            break

        if 0 == frame_id % math.floor(frame_rate):
            i = i + 1
            # print('frame_id={}'.format(frame_id))
            if to_skip <= 0:
                if verbose2:
                    show_image('frame', frame)
                box = check_image(frame.copy(), (py, px, ph, pw))
            else:
                to_skip -= 1
                continue

            to_skip = 3
            if box is not None:
                t = str(datetime.timedelta(seconds=i))
                if verbose2:
                    print('frame_id={}, goodFrame={}, i={}, t={}'.format(frame_id, box, i, t))

                (y, x, h, w) = box[0], box[1], box[2], box[3]
                crop_img = frame[y:y + h, x:x + w]

                if is_save_frame:
                    filename = is_save_frame + "/image_" + t.replace(':', '_') + ".jpg"
                    cv2.imwrite(filename, crop_img)

                txt = extract_text(crop_img, lang)
                if txt:
                    print('{} - {}'.format(t, txt))

                if verbose2:
                    cv2.drawContours(frame, np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], np.int32), 0,
                                     (0, 0, 255), 2)
                    show_image('frame', frame)


if is_video:
    check_video(src)
else:
    img = cv2.imread(src)
    if verbose2:
        show_image('tmp', img)
    print(check_image(img, (py, px, ph, pw)))
    if verbose2:
        show_image('tmp', img)
        # cv2.destroyAllWindows()
