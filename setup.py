from setuptools import find_packages, setup, Command

NAME = 'detect_frame'
DESCRIPTION = "The idea is to extract titles from long-running videos using computer vision."
URL = 'https://github.com/salyakhov/swim-time-lapse'
EMAIL = 'ruslansa@gmail.com'
AUTHOR = 'Ruslan Salyakhov'
REQUIRES_PYTHON = '>=3.6.5'
VERSION = '0.0.0'

REQUIRED = [
    'cv2', 'pytesseract',
]

