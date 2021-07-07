import cv2
from utils import *
import pptk


if __name__ == '__main__':
    CCD_width = 5.6
    f_mm = 4.3
    w = 2160
    h = 2160
    f_px = f_mm * w / CCD_width
    K = build_calibration_matrix(CCD_width=CCD_width,
                                 f_mm=f_mm,
                                 w=w,
                                 h=h)
    P0 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]],
                  dtype=float)

    img1, img2 = load_images('asset/1.jpg', 'asset/2.jpg')
    img1_gray, img2_gray = gray_images(img1, img2)

    kp1, des1 = find_keypoints_descriptors(img1)
    kp2, des2 = find_keypoints_descriptors(img2)
    
    pts1, pts2, matches = match_keypoints(kp1, des1, kp2, des2)
    pts1 = pts1.reshape((-1, 2))
    pts2 = pts2.reshape((-1, 2))

    draw_keypoint_matches(img1, kp1, img2, kp2, matches)

    # Calculate E
    E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC) # p_C_px, p_I_px !!order!!
    E = -E # Should take negative!!

    # Find projection matrix from E
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    P1 = np.concatenate((R, t), axis=1)

    # Compute rays
    # Restored 3D are normalized by nature
    pt1_3D_normalized = np.vstack([to_3D(K, pt) for pt in pts1])
    pt2_3D_normalized = np.vstack([to_3D(K, pt) for pt in pts2])
    # Convert to line
    pt1_ray = np.vstack([line_plot(ray, P0, length=1) for ray in pt1_3D_normalized])
    pt2_ray = np.vstack([line_plot(ray, P1, length=1) for ray in pt2_3D_normalized])

    # Find 3d midpoints
    # First find corresponding 3d points
    # pt1_3D is in frame 1's language, whereas pt2_3D is in frame 2
    midpoint = []
    for pt1_3D, pt2_3D in zip(pt1_3D_normalized, pt2_3D_normalized):
        t1, u1 = to_line(pt1_3D, P0)
        t2, u2 = to_line(pt2_3D, P1)
        midpoint.append(midpoint_3D(t1, t2, u1, u2))
    midpoint = np.vstack(midpoint)

    frame_I = plot_frame(P0)
    frame_C = plot_frame(P1)
    v = pptk.viewer(np.vstack((frame_I, frame_C, pt1_ray, pt2_ray, midpoint)))
    v.set(point_size=0.005)
