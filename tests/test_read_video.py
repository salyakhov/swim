from vutils import video_processor as vp
import os
import datetime

home = "../"


def show_video():
    path = os.path.join(home, "videos/day1_sample03.mp4")
    processor = vp.VideoProcessor(path)
    processor.process()


def crop_image(image, y, x, h, w):
    return image[y:y + h, x:x + w]


def video_processor():
    path = os.path.join(home, "videos/day1_sample03.mp4")
    processor = vp.VideoProcessor(path)

    processor.process(
        frame_processor=crop_image, y=0, x=0, h=100, w=100,
        verbose=True, verbose_delay=0
    )


if __name__ == "__main__":
    #show_video()
    video_processor()
