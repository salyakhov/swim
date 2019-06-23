import cv2
import datetime
import math
import types


def show_image(name, image, delay=0):
    # height, width, channels = img.shape
    if image is None:
        return
    cv2.imshow(name, image)
    cv2.waitKey(delay)


class VideoProcessor:
    def __init__(self, src):
        self.vs = cv2.VideoCapture(src)

    def process(self, frame_processor=None, verbose=True, verbose_delay=0, *args, **kwargs):
        """

        :param frame_processor: pass function to do something with each frame
        :param verbose: show video, debug info
        :param verbose_delay: value in ms, default 0 - infinite time
        :return:
        """
        frame_count = self.vs.get(7)
        if verbose:
            print(frame_count)
        frame_rate = self.vs.get(5)  # frame rate
        duration = frame_count / frame_rate
        t = str(datetime.timedelta(seconds=duration))
        if verbose:
            print('duration:', t)

        i = 0
        while self.vs.isOpened():
            frame_id = self.vs.get(1)  # current frame number
            if verbose and frame_id % 1000 == 0:
                t = str(datetime.timedelta(seconds=i))
                print('debug: frame_id={}, t={}'.format(frame_id, t))

            ret, frame = self.vs.read()
            if not ret:
                break

            if 0 == frame_id % math.floor(frame_rate):
                i += 1

                if verbose:
                    show_image('frame', frame, verbose_delay)

                if frame_processor is not None and isinstance(frame_processor, types.FunctionType):
                    processed = frame_processor(frame, *args, **kwargs)

                    if verbose:
                        show_image('processed', processed, verbose_delay)

