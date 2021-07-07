import os
import sys
import cv2
import numpy as np
from PIL import Image, ExifTags
from pdb import set_trace as bp
import matplotlib.pyplot as plt
from numpy import linalg


def load_images(filename1, filename2, img_size=1200):
    '''Loads 2 images.'''
    # img1 and img2 are HxWx3 arrays (rows, columns, 3-colour channels)
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    # img1 = cv2.resize(img1, (img_size, img_size))
    # img2 = cv2.resize(img2, (img_size, img_size))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    return img1, img2

def load_from_np(filename1, filename2):
    '''Loads 2 images.'''
    img1 = filename1
    img2 = filename2
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    return img1, img2


def gray_images(img1, img2):
    '''Convert images to grayscale if the images are found.'''
    try:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    except:
        print("Image not found!")

    return img1_gray, img2_gray


def find_keypoints_descriptors(img):
    '''Detects keypoints and computes their descriptors.'''
    detector = cv2.SIFT_create()
    kp, des = detector.detectAndCompute(img, None)
    
    return kp, des


def match_keypoints(kp1, des1, kp2, des2):
    '''Matches the descriptors in one image with those in the second image'''
    MIN_MATCH_COUNT = 10

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # store all the good matches as per Lowe's ratio test
    ratio = 0.5
    good_matches = [m for m, n in matches if m.distance < n.distance * ratio]

    # src_pts and dst_pts are Nx1x2 arrays that contain the x and y pixel coordinates
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    else:
        raise Exception("Not enough matches were found - %d/%d" %
                        (len(good_matches), MIN_MATCH_COUNT))
    return src_pts, dst_pts, good_matches


def draw_keypoint_matches(img1, kp1, img2, kp2, good):
    img = None
    img = cv2.drawMatches(img1, kp1, img2, kp2, good,
                          img, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img)
    plt.show()


def normalize_points(K, src_pts, dst_pts):
    '''Normalize points by multiplying them with the inverse of the K matrix.
    By multiplying the inverse of K, the dimension changes from pixels to mm'''
    # convert to 3xN arrays by making the points homogeneous
    src_pts = np.vstack((np.array([pt[0] for pt in src_pts]).T, np.ones(src_pts.shape[0])))
    dst_pts = np.vstack((np.array([pt[0] for pt in dst_pts]).T, np.ones(dst_pts.shape[0])))

    # normalize with the calibration matrices
    # norm_pts1 and norm_pts2 are 3xN arrays
    #       K * [Xc, Yc, Zc].T ~ [x, y, 1]
    # -->   [Xc/Zc, Yc/Zc, 1].T = K^-1 * [x, y, 1].T
    norm_pts1 = np.dot(np.linalg.inv(K), src_pts)
    norm_pts2 = np.dot(np.linalg.inv(K), dst_pts)

    # convert back to Nx1x2 arrays
    # [Xc/Zc, Yc/Zc].T
    # In order to restore the "true" coordinate, have to find Zc and multiply
    norm_pts1 = np.array([[pt] for pt in norm_pts1[:2].T])
    norm_pts2 = np.array([[pt] for pt in norm_pts2[:2].T])

    return norm_pts1, norm_pts2


def find_essential_matrix(K, norm_pts1, norm_pts2):
    '''Estimate an essential matrix that satisfies the epipolar constraint for all the corresponding points.'''
    # K = K1, the calibration matrix of the first camera of the current image pair
    # convert to Nx2 arrays for findFundamentalMat
    norm_pts1 = np.array([ pt[0] for pt in norm_pts1 ])
    norm_pts2 = np.array([ pt[0] for pt in norm_pts2 ])
    # inliers (1 in mask) are features that satisfy the epipolar constraint
    F, mask = cv2.findFundamentalMat(norm_pts1, norm_pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=0.1)
    E = np.dot(K.T, np.dot(F, K))

    return E, mask

def find_projection_matrices(E, poses):
    '''Compute the second camera matrix (assuming the first camera matrix = [I 0]).
    Output is a list of 4 possible camera matrices for P2.'''
    # the first camera matrix is assumed to be the identity matrix for the first image,
    # or the pose of the camera for the second and subsequent images
    P1 = poses[-1]
        
    # make sure E is rank 2
    U, S, V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V
    E = np.dot(U, np.dot(np.diag([1,1,0]), V))

    # create matrices
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    # return all four solutions
    P2 = [np.vstack( (np.dot(U,np.dot(W,V)).T, U[:,2]) ).T, 
          np.vstack( (np.dot(U,np.dot(W,V)).T, -U[:,2]) ).T,
          np.vstack( (np.dot(U,np.dot(W.T,V)).T, U[:,2]) ).T, 
          np.vstack( (np.dot(U,np.dot(W.T,V)).T, -U[:,2]) ).T]

    return P1, P2


def filter_keypoints(mask, src_pts, dst_pts):
    '''Filter the keypoints using the mask of inliers generated by findFundamentalMat.'''
    # src_pts and dst_pts are Nx1x2 arrays that contain the x and y pixel coordinates
    src_pts = src_pts[mask.ravel() == 1]
    dst_pts = dst_pts[mask.ravel() == 1]
    return src_pts, dst_pts


def apply_mask(mask, norm_pts1, norm_pts2):
    '''Keep only those points that satisfy the epipolar constraint.'''
    norm_pts1 = norm_pts1[mask.ravel() == 1]
    norm_pts2 = norm_pts2[mask.ravel() == 1]
    return norm_pts1, norm_pts2


def refine_points(norm_pts1, norm_pts2, E):
    '''Refine the coordinates of the corresponding points using the Optimal Triangulation Method.'''
    # convert to 1xNx2 arrays for cv2.correctMatches
    refined_pts1 = np.array([ [pt[0] for pt in norm_pts1 ] ])
    refined_pts2 = np.array([ [pt[0] for pt in norm_pts2 ] ])
    refined_pts1, refined_pts2 = cv2.correctMatches(E, refined_pts1, refined_pts2)

    # refined_pts are 1xNx2 arrays
    return refined_pts1, refined_pts2


def triangulate_points(P1, P2, refined_pts1, refined_pts2):
    '''Reconstructs 3D points by triangulation using Direct Linear Transformation.'''
    # convert to 2xN arrays
    # refined_pts1 = refined_pts1[0].T
    # refined_pts2 = refined_pts2[0].T
    refined_pts1 = refined_pts1.T
    refined_pts2 = refined_pts2.T

    # pick the P2 matrix with the most scene points in front of the cameras after triangulation
    ind = 0
    maxres = 0

    # for i in range(4):
    #     # triangulate inliers and compute depth for each camera
    #     homog_3D = cv2.triangulatePoints(P1, P2[i], refined_pts1, refined_pts2)
    #     # the sign of the depth is the 3rd value of the image point after projecting back to the image
    #     d1 = np.dot(P1, homog_3D)[2]
    #     d2 = np.dot(P2[i], homog_3D)[2]
        
    #     if sum(d1 > 0) + sum(d2 < 0) > maxres:
    #         maxres = sum(d1 > 0) + sum(d2 < 0)
    #         ind = i
    #         infront = (d1 > 0) & (d2 < 0)

    # triangulate inliers and keep only points that are in front of both cameras
    # homog_3D is a 4xN array of reconstructed points in homogeneous coordinates, pts_3D is a Nx3 array
    homog_3D = cv2.triangulatePoints(P1, P2, refined_pts1, refined_pts2)
    # homog_3D = homog_3D[:, infront]
    homog_3D = homog_3D / homog_3D[3]
    pts_3D = np.array(homog_3D[:3]).T

    return homog_3D, pts_3D


def apply_infront_filter(infront, norm_pts1, norm_pts2):
    '''Keep only those points that are in front of the cameras.'''
    norm_pts1 = norm_pts1[infront.ravel() == 1]
    norm_pts2 = norm_pts2[infront.ravel() == 1]
    return norm_pts1, norm_pts2


def attach_indices(i, pts_3D, src_pts, dst_pts, pt_cloud_indexed=[]):
    '''Attach to each 3D point, indices into the original lists of keypoints and descriptors 
    of the 2D points that contributed to this 3D point in the cloud.'''

    def find_point(new_pt, pt_cloud_indexed):
        for old_pt in pt_cloud_indexed:
            try:
                if np.array_equal(new_pt.origin[i], old_pt.origin[i]):
                    return True, old_pt
            except KeyError:
                continue
        return False, None

    new_pts = [Point3D(pt, {i: src_pts[num], i+1: dst_pts[num]})
               for num, pt in enumerate(pts_3D)]

    if pt_cloud_indexed == []:
        pt_cloud_indexed = new_pts
    else:
        for num, new_pt in enumerate(new_pts):
            found, old_pt = find_point(new_pt, pt_cloud_indexed)
            if found:
                old_pt.origin[i+1] = dst_pts[num]
            else:
                pt_cloud_indexed.append(new_pt)

    return pt_cloud_indexed


class Point3D(object):
    def __init__(self, coords, origin):
        self.coords = coords
        self.origin = origin


def scan_cloud(i, prev_dst, src_pts, pt_cloud_indexed):
    '''Check for matches between the new frame and the current point cloud.'''
    # prev_dst contains the x & y coords of the keypoints from the second image in the last iteration
    # src_pts contains the x & y coords of the keypoints from the first image in the current iteration
    # the second image in the last iteration is the first image in the current iteration
    # therefore, check for matches by comparing the x & y coords
    matched_pts_2D = []
    matched_pts_3D = []
    indices = []

    for idx, new_pt in enumerate(src_pts):
        for old_pt in prev_dst:
            if np.array_equal(new_pt, old_pt):
                # found a match: a keypoint that contributed to both the last and current point clouds
                matched_pts_2D.append(new_pt)
                indices.append(idx)

    for pt_2D in matched_pts_2D:
        # pt_cloud_indexed is a list of 3D points from the previous cloud with their 2D pixel origins
        for pt in pt_cloud_indexed:
            try:
                if np.array_equal(pt.origin[i], pt_2D):
                    matched_pts_3D.append(pt.coords)
                    break
            except KeyError:
                continue
        continue

    matched_pts_2D = np.array(matched_pts_2D, dtype='float32')
    matched_pts_3D = np.array(matched_pts_3D, dtype='float32')
    return matched_pts_2D, matched_pts_3D, indices


def compute_cam_pose(K, matched_pts_2D, matched_pts_3D, poses):
    '''Compute the camera pose from a set of 3D and 2D correspondences.'''
    # if len(matched_pts_2D) == 0 or len(matched_pts_3D) == 0:
    #     return poses
    rvec, tvec = cv2.solvePnPRansac(matched_pts_3D, matched_pts_2D, K, None)[1:3]
    rmat = cv2.Rodrigues(rvec)[0]
    pose = np.hstack((rmat, tvec))
    poses.append(pose)
    return poses


def build_calibration_matrix(CCD_width, f_mm, w, h):
    '''Extract exif metadata from image files, and use them to build the 2 calibration matrices.'''
    # focal length in pixels = (focal length in mm) * (image width in pixels) / (CCD width in mm)
    f_p = (f_mm * w) / CCD_width
    K = np.array([[f_p, 0, w/2], [0, f_p, h/2], [0, 0, 1]])
    
    return K


# def Metadata(filename):
#     img = Image.open(filename)
#     exif = {
#         ExifTags.TAGS[k]: v
#         for k, v in img._getexif().items()
#         if k in ExifTags.TAGS
#     }
#     return exif

def draw_keypoints(img, kp):
    img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img)
    plt.show()

def gen_pt_cloud(i, K, image1, image2, poses):
    '''Generates a point cloud for a pair of images.'''
    print("    Loading images...")
    img1, img2 = load_from_np(image1, image2)
    img1_gray, img2_gray = gray_images(img1, img2)

    print("    Detecting keypoints...\n    Computing descriptors...")
    kp1, des1 = find_keypoints_descriptors(img1_gray)
    kp2, des2 = find_keypoints_descriptors(img2_gray)
    # draw_keypoints(img1, kp1)
    # draw_keypoints(img2, kp2)

    print("    Matching keypoints...")
    src_pts, dst_pts, good_matches = match_keypoints(kp1, des1, kp2, des2)
    # draw_keypoint_matches(img1, kp1, img2, kp2, good_matches)

    # print("    Normalizing keypoints...")
    norm_pts1, norm_pts2 = normalize_points(K, src_pts, dst_pts)

    print("    Finding the essential and projection matrices...")
    E, mask = find_essential_matrix(K, norm_pts1, norm_pts2)
    P1, P2 = find_projection_matrices(E, poses)
    src_pts, dst_pts = filter_keypoints(mask, src_pts, dst_pts)
    norm_pts1, norm_pts2 = apply_mask(mask, norm_pts1, norm_pts2)
    refined_pts1, refined_pts2 = refine_points(norm_pts1, norm_pts2, E)
    # E, mask = find_essential_matrix(K, src_pts, dst_pts)
    # # F = np.dot(np.linalg.inv(K.T), np.dot(E, np.linalg.inv(K)))
    # # P1, P2 = find_projection_matrices(E, poses)
    # retval, R, t, mask_ = cv2.recoverPose(E, src_pts, dst_pts, K)
    # P1 = poses[-1]
    # P2 = np.concatenate((R, t), axis=1)
    # src_pts, dst_pts = filter_keypoints(mask, src_pts, dst_pts)
    # # norm_pts1, norm_pts2 = apply_mask(mask, norm_pts1, norm_pts2)
    # refined_pts1, refined_pts2 = refine_points(src_pts, dst_pts, F)

    print("    Triangulating 3D points...")
    homog_3D, pts_3D, infront = triangulate_points(P1, P2, refined_pts1, refined_pts2)
    norm_pts1, norm_pts2 = apply_infront_filter(infront, norm_pts1, norm_pts2)

    print("    Initializing feature tracks...")
    pt_cloud_indexed = attach_indices(i, pts_3D, src_pts, dst_pts)

    return K, dst_pts, homog_3D, pts_3D, pt_cloud_indexed


def find_new_pts_feat(i, K, image1, image2, prev_dst, poses, pt_cloud_indexed, last):
    print("    Loading images...")
    img1, img2 = load_images(image1, image2)
    img1_gray, img2_gray = gray_images(img1, img2)

    print("    Detecting keypoints...\n    Computing descriptors...")
    prev_kp, prev_des = find_keypoints_descriptors(img1_gray)
    new_kp, new_des = find_keypoints_descriptors(img2_gray)
    print("    Matching keypoints...")
    src_pts, dst_pts = match_keypoints(prev_kp, prev_des, new_kp, new_des)
    
    print("    Normalizing keypoints...")
    norm_pts1, norm_pts2 = normalize_points(K, src_pts, dst_pts)

    print("    Finding the essential and projection matrices...")
    E, mask = find_essential_matrix(K, norm_pts1, norm_pts2)
    src_pts, dst_pts = filter_keypoints(mask, src_pts, dst_pts)
    norm_pts1, norm_pts2 = apply_mask(mask, norm_pts1, norm_pts2)
    refined_pts1, refined_pts2 = refine_points(norm_pts1, norm_pts2, E)

    print("    Scanning cloud...")
    matched_pts_2D, matched_pts_3D, indices = scan_cloud(i, prev_dst, src_pts, pt_cloud_indexed)
    print("    Computing camera pose...")
    poses = compute_cam_pose(K, matched_pts_2D, matched_pts_3D, poses)
    # P1, P2 = find_projection_matrices(E, poses)
    retval, R, t, mask_ = cv2.recoverPose(E, norm_pts1, norm_pts2, K)
    P1 = poses[-1]
    P2 = np.concatenate((R, t), axis=1)

    print("    Triangulating 3D points...")
    homog_3D, pts_3D = triangulate_points(P1, P2, refined_pts1, refined_pts2)
    # norm_pts1, norm_pts2 = apply_infront_filter(infront, norm_pts1, norm_pts2)

    print("    Assembling feature tracks...")
    pt_cloud_indexed = attach_indices(i, pts_3D, src_pts, dst_pts, pt_cloud_indexed)

    if last:
        # find the pose of the last camera
        matched_pts_2D = np.array([dst_pts[i] for i in indices])
        poses = compute_cam_pose(K, matched_pts_2D, matched_pts_3D, poses)

    return dst_pts, poses, homog_3D, pts_3D, pt_cloud_indexed

def triangulate_point_from_multiple_views_linear(proj_matricies, points):
    """Triangulates one point from multiple (N) views using direct linear transformation (DLT).
    For more information look at "Multiple view geometry in computer vision",
    Richard Hartley and Andrew Zisserman, 12.2 (p. 312).
    Args:
        proj_matricies numpy array of shape (N, 3, 4): sequence of projection matricies (3x4)
        points numpy array of shape (N, 2): sequence of points' coordinates
    Returns:
        point_3d numpy array of shape (3,): triangulated point
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)
    A = np.zeros((2 * n_views, 4))
    for j in range(len(proj_matricies)):
        A[j * 2 + 0] = points[j][0] * proj_matricies[j][2, :] - proj_matricies[j][0, :]
        A[j * 2 + 1] = points[j][1] * proj_matricies[j][2, :] - proj_matricies[j][1, :]

    u, s, vh =  np.linalg.svd(A, full_matrices=False)
    point_3d_homo = vh[3, :]

    point_3d = cv2.convertPointsFromHomogeneous(point_3d_homo)

    return point_3d

def plot_frame(P):
    R = P[:, :-1]
    t = P[:, -1]

    discrete_steps = np.arange(0, 1, 1e-2)
    i = R[:,0]
    j = R[:,1]
    k = R[:,2]
    x = np.array([i*step for step in discrete_steps] + t)
    y = np.array([j*step for step in discrete_steps] + t)
    z = np.array([k*step for step in discrete_steps] + t)
    return np.vstack((x, y, z))

def to_pixel(K, pt_3D):
    pt_px_homo = K@pt_3D
    pt_px = (pt_px_homo/pt_px_homo[-1])[:-1]
    return pt_px

def to_3D(K, pt_px):
    pt_px_homo = np.concatenate((pt_px, np.array([1])))
    Kinv = np.linalg.inv(K)
    pt_3D = Kinv@pt_px_homo
    return pt_3D


def line_plot(vec_3D, P, length=1):
    R = P[:, :-1]
    t = P[:, -1]
    vec_3D = R@vec_3D
    discrete_steps = np.arange(0, length, 1e-2)
    unit_vec = vec_3D/np.linalg.norm(vec_3D)
    line = [unit_vec * step for step in discrete_steps] + t
    return np.vstack(line)

def to_line(pt_3D_normalized, P):
    R = P[:, :-1]
    t = P[:, -1]
    vec_3D = R@pt_3D_normalized
    u = vec_3D/np.linalg.norm(vec_3D)
    # Returns origin and unit vector
    # ex) p = t + step*u
    return t, u

def midpoint_3D(t1, t2, u1, u2):
    n = np.cross(u1, u2)
    n1 = np.cross(u1, n)
    n2 = np.cross(u2, n)

    c1 = t1 + np.dot((t2 - t1), n2)/np.dot(u1, n2) * u1
    c2 = t2 + np.dot((t1 - t2), n1)/np.dot(u2, n1) * u2

    return (c1 + c2)/2