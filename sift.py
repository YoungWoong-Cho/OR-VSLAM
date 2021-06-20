import cv2
import matplotlib.pyplot as plt
import os
from pdb import set_trace as bp


class SIFT():
    def __init__(self, **kwargs):
        if 'img1' in kwargs.keys() and 'img2' in kwargs.keys():
            self.img1_path = kwargs['img1']
            self.img2_path = kwargs['img2']
            self.mode = 'img'
        elif 'vid' in kwargs.keys():
            self.vid_path = kwargs['vid']
            self.mode = 'vid'

    def read_img(self, path):
        files_list = sorted(os.listdir(path))
        return [os.path.join(path, fname) for fname in files_list]

    def resize(self, img):
        img_resize = cv2.resize(img, (1200, 1200))
        return img_resize

    def run_sift(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        return kp, des

    def read_vid(self, path, num_frames):
        cap = cv2.VideoCapture(path)
        frame = 0
        img_list = []
        while True and frame < num_frames:
            ret, fram = cap.read()
            if ret:
                self.resize(fram)
                img_list.append(fram)
                frame += 1
            else:
                print('[ERROR] frame not read')
        return img_list

    def match(self):
        if self.mode == 'img':
            self.img1 = cv2.imread(self.img1_path)
            self.img2 = cv2.imread(self.img2_path)
        elif self.mode == 'vid':
            self.img1, self.img2 = self.read_vid(self.vid_path, num_frames=2)

        self.kp1, self.des1 = self.run_sift(self.img1)
        self.kp2, self.des2 = self.run_sift(self.img2)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(self.des1, self.des2)
        self.matches = sorted(matches, key=lambda x: x.distance)

    def plot_fig(self):
        img_match = cv2.drawMatches(
            self.img1, self.kp1, self.img2, self.kp2, self.matches[:100], None)  # , flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_match)
        plt.show()

    def save_fit(self):
        if not os.path.exists('output'):
            os.mkdir('output')
        plt.savefig('output/out_sift.png')


if __name__ == '__main__':
    img1 = 'asset/img1.jpg'
    img2 = 'asset/img2.jpg'
    vid = 'asset/vid.MOV'
    # sift = SIFT(img1=img1, img2=img2)
    sift = SIFT(vid=vid)
    sift.match()
    sift.plot_fig()
