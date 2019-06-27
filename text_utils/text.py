import cv2
import pytesseract
from image_utils.detect_area import show_image
from image_utils.detect_area import grayscale


def simple_extract(image, language):
    return pytesseract.image_to_string(grayscale(image), lang=language)


def blur_extract(image, language, verbose=0):
    """
    100% bullshit but it works in some cases
    """
    (thresh, im_bw) = cv2.threshold(grayscale(image), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    blurred = cv2.GaussianBlur(im_bw, (5, 5), 1)
    # (thresh, im_bw) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # blurred = cv2.medianBlur(im_bw, 3)
    if verbose >= 2:
        show_image('blur', blurred)
    return pytesseract.image_to_string(blurred, lang=language)


def adaptive_tresholding(image, language, verbose):
    """
    https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # should be gray scale
    """
    show_image('gray', gray)
    ret, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    print("th1", pytesseract.image_to_string(th1, lang=language))
    show_image('th1', th1)
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    print("th2", pytesseract.image_to_string(th2, lang=language))
    show_image('th2', th2)
    """
    th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if verbose >= 2:
        show_image('th3', th3)

    return pytesseract.image_to_string(th3, lang=language)


def otsus_binarization(image, language, verbose):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding
    """
    ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show_image('th2', th2)
    ret2_2, th2_2 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show_image('th2_2', th2_2)
    print("th2", pytesseract.image_to_string(th2, lang=language))
    print("th2_2", pytesseract.image_to_string(th2_2, lang=language))
    # Otsu's thresholding after Gaussian filtering
    blur1 = cv2.GaussianBlur(gray, (5, 5), 0)
    show_image('blur1', blur1)
    blur2 = cv2.GaussianBlur(gray, (5, 5), 1)
    show_image('blur2', blur2)
    ret3, th3 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("th3", pytesseract.image_to_string(th3, lang=language))
    show_image('th3', th3)
    """
    blur2 = cv2.GaussianBlur(gray, (5, 5), 1)
    ret4, th4 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if verbose >= 2:
        show_image('th4', th4)

    return pytesseract.image_to_string(th4, lang=language)
