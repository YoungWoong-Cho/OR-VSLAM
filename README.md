# OR-VSLAM

## Dependencies
The code was tested under the following dependencies
- python 3.8.5
- opencv-python 4.5.1.48
- matplotlib 3.3.2

## Preparation
- If you want to test out the images, place two images in the `asset` folder, and update the lines from `sift.py` as follows:
```
if __name__ == '__main__':
    img1 = 'asset/{image_1_filename}'
    img2 = 'asset/{image_2_filename}'
    sift = SIFT(img1=img1, img2=img2)
    sift.match()
    sift.plot_fig()
```
- If you want to test out the video, place the video in the `asset` folder, and update the lines from `sift.py` as follows:
```
if __name__ == '__main__':
    vid = 'asset/vid.MOV'
    sift = SIFT(vid=vid)
    sift.match()
    sift.plot_fig()
```

## Execution
In order to execute, please type in the following command:
```
python3 sift.py
```
After the execution, the result will be saved under `output` folder.
If you place a video, the result will show the first two frames.

## Sample Output
![alt text](output/out_sift.png)
