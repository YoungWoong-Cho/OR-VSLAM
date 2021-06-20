import cv2
import matplotlib.pyplot as plt
import os
from pdb import set_trace as bp


def resize(img):
    img_resize = cv2.resize(img, (1200, 1200))
    return img_resize


def SIFT(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def read_img(path):
    files_list = sorted(os.listdir(path))
    return [os.path.join(path, fname) for fname in files_list]

def read_vid(path, num_frames):
    cap = cv2.VideoCapture(path)
    frame = 0
    img_list = []
    while True and frame < num_frames:
        ret, fram = cap.read()
        if ret:
            # gray = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('video', gray)
            # k == cv2.waitKey(1) & 0xFF
            # if k == 27:
            #     break
            resize(fram)
            img_list.append(fram)
            frame += 1
        else:
            print('error')
    return img_list

def match(img1, img2, kp1, kp2, des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img_match = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:100], None)  # , flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_match)
    if not os.path.exists('output'):
        os.mkdir('output')
    plt.savefig('output/out_sift.png')
    plt.show()


# img_path = os.path.join(os.getcwd(), 'asset')
# img_list = read_img(img_path)

img1, img2 = read_vid('asset/IMG_3927.MOV', num_frames=2)
# bp()

# img1 = cv2.imread(img_list[0])
# img2 = cv2.imread(img_list[1])

img1 = resize(img1)
img2 = resize(img2)

kp1, des1 = SIFT(img1)
kp2, des2 = SIFT(img2)

match(img1, img2, kp1, kp2, des1, des2)
