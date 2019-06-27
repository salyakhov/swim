import cv2
import numpy as np
import argparse
import math
import datetime
import os
from image_utils.detect_area import show_image
from image_utils.detect_area import find_contour
from text_utils import text


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-v", "--verbose", required=False, help="show debug info", action="count", default=0)
ap.add_argument("-minp", "--minperimeter", required=False, help="minimum figure perimeter", default=0)
ap.add_argument("-maxp", "--maxperimeter", required=False, help="maximum figure perimeter", default=1000000)
ap.add_argument("-mina", "--minarea", required=False, help="minimum figure area")
ap.add_argument("-maxa", "--maxarea", required=False, help="maximum figure area")
ap.add_argument("-minn", "--minpolygonvertices", required=False, help="min number of vertices in polygon", default=4)
ap.add_argument("-maxn", "--maxpolygonvertices", required=False, help="max number of vertices in polygon", default=4)
ap.add_argument("-a", "--area", required=False, help="'y, x, h, w'")
ap.add_argument("-sf", "--saveframe", required=False, help="save good frame", default=False)
ap.add_argument("-l", "--lang", required=False, help="language: rus, eng", default='rus')
ap.add_argument("-t", "--threshold", required=False, help="try 0 if 1 which is default doesn't work", default=1, type=int)
ap.add_argument("-at", "--adaptive_threshold", required=False,
                help="Thresholding: 0 (default) - as is, 1 - Adaptive, 2 - Otsu's Binarization",
                default=0, type=int)

args = vars(ap.parse_args())

minp = int(args["minperimeter"])
maxp = int(args["maxperimeter"])
mina = int(args["minarea"])
maxa = int(args["maxarea"])
minn = int(args["minpolygonvertices"])
maxn = int(args["maxpolygonvertices"])
is_save_frame = args["saveframe"]
lang = args["lang"]
threshold = args["threshold"]
adaptive_threshold = args["adaptive_threshold"]

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

verbose = args["verbose"]

#verbose = True if args["verbose"] and args["verbose"] > 0 else False
#verbose2 = True if verbose and args["verbose"] > 1 else False


def check_video(src):
    print("Video:", src)
    cap = cv2.VideoCapture(src)
    frame_count = cap.get(7)
    print("Frames:", frame_count)
    frame_rate = cap.get(5)  # frame rate

    duration = frame_count / frame_rate
    # minutes = int(duration/60)
    # seconds = duration%60
    t = str(datetime.timedelta(seconds=duration))
    # print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
    print("Duration:", t)

    to_skip = 0
    frame_id = 0
    step = frame_rate
    while cap.isOpened():
        #frame_id = cap.get(1)  # current frame number
        frame_id += step + to_skip
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        sec = int(frame_id / frame_rate)
        t = str(datetime.timedelta(seconds=sec))
        if verbose >= 1 and frame_id % (frame_rate*60) == 0:
            print('debug: frame_id={}, sec={}, t={}'.format(frame_id, sec, t))

        ret, frame = cap.read()
        if not ret:
            break

        box = find_contour(frame, (py, px, ph, pw), threshold_type=threshold, mina=mina, maxa=maxa, verbose=verbose)
        if box is None:
            step = frame_rate
        else:
            step = 5  # move by specified number of frames
            if check_image(frame, box, t):
                to_skip = 60 * frame_rate
                t = str(datetime.timedelta(seconds=sec))
                if verbose >= 2:
                    print('Good frame! #{}, sec={}, t={}'.format(frame_id, sec, t))
            else:
                to_skip = 0


def check_image(image, box, time=None):
    res = False
    (y, x, h, w) = box[0], box[1], box[2], box[3]
    if verbose >= 2:
        cv2.drawContours(image, np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], np.int32), 0,
                         (0, 0, 255), 2)
        show_image('image', image)

    crop_img = image[y:y + h, x:x + w]

    if verbose >= 2:
        show_image('crop', crop_img)

    cnt = 0
    if is_save_frame:
        filename = os.join(is_save_frame, "image_" + (time.replace(':', '_') + cnt if time else str(cnt)) + ".jpg")
        cv2.imwrite(filename, crop_img)

    if adaptive_threshold == 1:
        txt = text.adaptive_tresholding(crop_img, lang, verbose=verbose)
    elif adaptive_threshold == 2:
        txt = text.otsus_binarization(crop_img, lang, verbose=verbose)
    else:
        txt = text.simple_extract(crop_img, lang)

    if txt:
        if len(txt.replace(' ', '').strip('\n')) > 16:
            res = True
            print('{} - {}'.format(time, txt))

    return res


if is_video:
    check_video(src)
else:
    img = cv2.imread(src)
    box = find_contour(img, (py, px, ph, pw), threshold_type=threshold, mina=mina, maxa=maxa)
    if box is not None:
        check_image(img, box)

cv2.destroyAllWindows()
