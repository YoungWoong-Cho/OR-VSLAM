import cv2 as cv
import matplotlib.pyplot as plt
import os

def resize(img):
    img_resize = cv.resize(img, (1200, 1200))
    return img_resize

def SIFT(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des

def read_files(path):
    files_list =  sorted(os.listdir(path))
    return [os.path.join(path, fname) for fname in files_list]


def match(img1, img2, kp1, kp2, des1, des2):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img_match = cv.drawMatches(
        img1, kp1, img2, kp2, matches[:50], None)#, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_match), plt.show()

img_path = os.path.join(os.getcwd(), 'asset')
img_list = read_files(img_path)

img1 = cv.imread(img_list[0])
img2 = cv.imread(img_list[1])

img1 = resize(img1)
img2 = resize(img2)

kp1, des1 = SIFT(img1)
kp2, des2 = SIFT(img2)

match(img1, img2, kp1, kp2, des1, des2)
