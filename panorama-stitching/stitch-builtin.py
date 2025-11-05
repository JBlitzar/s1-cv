import cv2
import numpy as np
from PIL import Image
import sys

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

def match_features(desc1, desc2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = [m for m,n in knn if m.distance < ratio * n.distance]
    return good

def compute_homography(kps1, kps2, matches, ransac_thresh=5.0):
    if len(matches) < 4:
        return None, None
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransac_thresh)
    return H, mask

def stitch(img1, img2, H): # ai-generated
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners_img2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H)
    all_corners = np.concatenate((np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2), warped_corners), axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translate = [-x_min, -y_min]
    H_trans = np.array([[1,0,translate[0]],[0,1,translate[1]],[0,0,1]])
    result = cv2.warpPerspective(img2, H_trans.dot(H), (x_max - x_min, y_max - y_min))
    result[translate[1]:translate[1]+h1, translate[0]:translate[0]+w1] = img1
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