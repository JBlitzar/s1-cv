import cv2
import numpy as np
from PIL import Image
import sys
import random
from tqdm import trange, tqdm


class FeatureDescriptor:
    def __init__(self, x, y, descriptor):
        self.x = x
        self.y = y
        self.descriptor = descriptor

    def compute_similarity(self, other):
        return np.linalg.norm(self.descriptor - other.descriptor)

    def get_closest_match(self, others):
        best_match, best_distance = None, float("inf")
        for other in others:
            dist = self.compute_similarity(other)
            if dist < best_distance:
                best_distance, best_match = dist, other
        return best_match

    def get_closest_xy_match(self, others):
        best_match, best_distance = None, float("inf")
        for other in others:
            dist = np.hypot(self.x - other.x, self.y - other.y)
            if dist < best_distance:
                best_distance, best_match = dist, other
        return best_match


def match_next_image(descriptors1, descriptors2):
    def attempt_get_homography():
        pts1 = random.choices(descriptors1, k=4)
        pts2 = [p.get_closest_match(descriptors2) for p in pts1]
        H, _ = cv2.findHomography(
            np.array([[p.x, p.y] for p in pts1]), np.array([[p.x, p.y] for p in pts2])
        )
        if H is None:
            return np.eye(3), float("inf")
        score = 0.0

        # I had ai (claude sonnet 4) optimize this (which was previously O(n^2) WITH object instantiation each time) because I don't have the bandwidth to do it myself atm
        transform_point = np.array([0.0, 0.0, 1.0])
        for p in tqdm(descriptors1, leave=False):
            transform_point[0] = p.x
            transform_point[1] = p.y
            transformed = H @ transform_point
            transformed /= transformed[2]

            best_match, best_distance = None, float("inf")
            for other in descriptors2:
                dist = np.hypot(transformed[0] - other.x, transformed[1] - other.y)
                if dist < best_distance:
                    best_distance, best_match = dist, other

            score += np.linalg.norm(p.descriptor - best_match.descriptor)
        return H, score

    best_H, best_score = None, float("inf")
    for _ in trange(1000, desc="RANSAC iterations"):
        H, score = attempt_get_homography()
        if score < best_score:
            best_H, best_score = H, score
    print("Best score:", best_score)
    return best_H


def save(img, name="out.jpg"):
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(name)


def detect_and_compute(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(gray, None)
    return img, kps, desc


def to_feature_descriptors(kps, desc):
    if desc is None or len(kps) == 0:
        return []
    return [
        FeatureDescriptor(kp.pt[0], kp.pt[1], d.astype(np.float32))
        for kp, d in zip(kps, desc)
    ]


def match_features(desc1, desc2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    return good


def compute_homography(kps1, kps2, matches, ransac_thresh=5.0):
    if len(matches) < 4:
        return None, None
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransac_thresh)
    return H, mask


def stitch(img1, img2, H):  # ai-generated
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

    H, mask = compute_homography(kps1, kps2, matches)
    if H is None:
        print("Not enough matches to compute homography.")
        sys.exit(1)

    stitched = stitch(img1, img2, H)
    save(stitched, "stitched.jpg")
    print("Saved stitched.jpg")

    # Incremental test: reuse SIFT keypoints/descriptors but estimate H via custom match_next_image
    fd1 = to_feature_descriptors(kps1, desc1)
    fd2 = to_feature_descriptors(kps2, desc2)
    if len(fd1) >= 4 and len(fd2) >= 4:
        H12 = match_next_image(fd1, fd2)  # maps img1 -> img2
        try:
            H21 = np.linalg.inv(H12)
            stitched_custom = stitch(img1, img2, H21)
            save(stitched_custom, "stitched_custom.jpg")
            print("Saved stitched_custom.jpg")
        except np.linalg.LinAlgError:
            print("Custom H was singular; skipping stitched_custom.jpg")
    else:
        print("Not enough features for custom matcher.")
