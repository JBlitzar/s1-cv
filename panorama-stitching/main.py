import time
import cv2
import numpy as np
from PIL import Image
import sys
import random
from tqdm import trange, tqdm


def match_next_image(kps1, kps2, matches):
    """
    Compute homography using custom implementaiton of RANSAC-like approach by sampling matches.
    Args:

        kps1: Keypoints from image 1.
        kps2: Keypoints from image 2.
        matches: List of matched keypoints between the two images. (in the form of cv2.DMatch objects)

    Returns:
        best_H: The best homography matrix found.
        best_score: The score of the best homography (lower is better).
    """

    # reordered so that the same indices correspond to matches.
    pts1_array = np.array(
        [kps1[m.queryIdx].pt for m in matches], dtype=np.float32
    )  # array of [[x,y],[x,y],[x,y],...]
    pts2_array = np.array(
        [kps2[m.trainIdx].pt for m in matches], dtype=np.float32
    )  # array of [[x,y],[x,y],[x,y],...]

    def attempt_get_homography():
        # get four random ones
        # compute homography
        # get score (transform points and compare to matches)
        sample_indices = random.sample(range(len(pts1_array)), 4)
        pts1_sample = pts1_array[sample_indices]
        pts2_sample = pts2_array[sample_indices]
        # print("Sampled pts1: ", pts1_sample)
        # print("Sampled pts2: ", pts2_sample)
        H, _ = cv2.findHomography(pts1_sample, pts2_sample)
        if H is None:
            print("DING DING DING H IS NONE")
            return np.eye(3), float("inf")

        # get whole score
        score = 0
        # transform all pts1
        # Claude sonnet 4 generated this snippet because I don't know linalg formally, haha
        ones = np.ones((pts1_array.shape[0], 1), dtype=np.float32)
        pts1_homogeneous = np.hstack((pts1_array, ones))
        transformed_pts1 = (H @ pts1_homogeneous.T).T
        transformed_pts1 /= transformed_pts1[:, 2][:, np.newaxis]
        transformed_pts1_xy = transformed_pts1[:, :2]
        # compute similarity

        for i in range(len(transformed_pts1_xy)):
            # get corresponding pts2 and compare distance with linalg.norm
            dist = np.linalg.norm(transformed_pts1_xy[i] - pts2_array[i])
            score += dist

        return H, score

    best_H = None
    best_score = float("inf")
    for _ in trange(1000):
        H, score = attempt_get_homography()
        if score < best_score:
            best_score = score
            best_H = H
    print("Best score: ", best_score)
    print("Best H: ", best_H)
    return best_H, best_score


def save(img, name="out.jpg"):
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(name)


def detect_and_compute(path):
    """Detect keypoints and compute descriptors for an image using cv2's builtin SIFT.
    Args:
        path: Path to the image file.
    Returns:
        img: The loaded image.
        kps: Detected keypoints.
        desc: Computed descriptors.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=150)
    kps, desc = sift.detectAndCompute(gray, None)
    return img, kps, desc


# ai-generated feature matching function (just uses builtin apis, no algorithmic complexity here)
# claude sonnet 4 via github copilot
def match_features(desc1, desc2, ratio=0.75):
    """Match features between two sets of descriptors using the ratio test and cv2's builtin BFMatcher and knnMatch.
    Args:

        desc1: Descriptors from image 1.
        desc2: Descriptors from image 2.
        ratio: Ratio threshold for the ratio test.
    Returns:
        good: List of good matches that passed the ratio test."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    return good


# ai-generated homography computation function (just uses builtin apis, no algorithmic complexity here)
# claude sonnet 4 via github copilot
def compute_homography(kps1, kps2, matches, ransac_thresh=5.0):
    """Compute homography between two sets of keypoints using matched features and cv2's builtin findHomography with RANSAC.
    Args:
        kps1: Keypoints from image 1.
        kps2: Keypoints from image 2.
        matches: List of matched keypoints between the two images. (in the form of cv2.DMatch objects)
        ransac_thresh: RANSAC reprojection threshold.
    Returns:
        H: The computed homography matrix.
        mask: Mask of inliers used in the homography computation. (unused)"""
    if len(matches) < 4:
        return None, None
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransac_thresh)
    return H, mask


def stitch(img1, img2, H):  # ai-generated image manipulation code.
    """
    Stitch two images together using the provided homography matrix.
    Args:
        img1: The first image (to be the base).
        img2: The second image (to be warped and stitched onto the first).
        H: The homography matrix mapping img2 to img1.
    Returns:
        result: The stitched panorama image in the form of a numpy array.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H)
    all_corners = np.concatenate(
        (
            np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2),
            warped_corners,
        ),
        axis=0,
    )
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translate = [-x_min, -y_min]
    H_trans = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]])
    result = cv2.warpPerspective(img2, H_trans.dot(H), (x_max - x_min, y_max - y_min))
    result[translate[1] : translate[1] + h1, translate[0] : translate[0] + w1] = img1
    return result


if __name__ == "__main__":
    img1_path = "images/1.jpg"
    img2_path = "images/2.jpg"

    img1, kps1, desc1 = detect_and_compute(img1_path)
    img2, kps2, desc2 = detect_and_compute(img2_path)

    matches = match_features(desc1, desc2)
    print(f"Matches found: {len(matches)}")
    print(
        "matches[0].queryidx: ",
        matches[0].queryIdx,
        ", matches[0].trainidx:",
        matches[0].trainIdx,
    )  # indices
    print("Corresponding coordinates:")
    print("kps1[queryIdx]: ", kps1[matches[0].queryIdx].pt)
    print("kps2[trainIdx]: ", kps2[matches[0].trainIdx].pt)

    H, _ = compute_homography(kps1, kps2, matches)
    print("Computed homography (builtin opencv):")
    print(H)

    stitched = stitch(img1, img2, H)
    save(stitched, "stitched.jpg")
    print("Saved stitched.jpg")

    img1, kps1, desc1 = detect_and_compute(img1_path)
    img2, kps2, desc2 = detect_and_compute(img2_path)
    matches = match_features(desc1, desc2)

    H, _ = match_next_image(kps1, kps2, matches)
    print("Calculated homography:")
    print(H)
    print(type(H))
    print("Shape of H: ", H.shape)
    print("H.dtype: ", H.dtype)
    stitched_custom = stitch(img2, img1, H)
    save(stitched_custom, "stitched_custom.jpg")
    print("Saved stitched_custom.jpg")
