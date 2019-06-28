# Swim
swim-time-lapse

# Setup
* install OpenCV on Mac OS
https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/

# Utils
* get video info
youtube-dl -F https://www.youtube.com/watch?v=f6gRpmi5zCI

* download video
youtube-dl -f136 https://www.youtube.com/watch?v=f6gRpmi5zCI -o day1_720p.mp4

* create sample video using ffmpeg for testing reason
ffmpeg -ss 00:29:50 -i videos/day1_720p.mp4 -t 00:00:10 -acodec copy videos/day1_sample03.mp4

* extract frame using ffmpeg
ffmpeg -i videos/day1_sample03.mp4 -ss 00:00:05.00 -vframes 1 videos/day1_p03.jpg

# Tune Up
https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html


# Screencast
* Part 1 https://www.youtube.com/watch?v=6V1bCYrIJ7U
* Part 2 https://www.youtube.com/watch?v=PZNhzZMjwDc
