from stitch_two_images import stitch, detect_and_compute, match_features, save, match_next_image, compute_homography
import cv2
import numpy as np
import glob
from tqdm import trange



if __name__ == "__main__":
    imgs = sorted(glob.glob("images/*.jpg"))

    # get H mapping 2 onto 1, then 3 onto 2, then 4 onto 3, etc
    # then stitch all together
    homographies = []
    for i in trange(len(imgs) - 1):
        img1, kps1, desc1 = detect_and_compute(imgs[i])
        img2, kps2, desc2 = detect_and_compute(imgs[i + 1])
        matches = match_features(desc1, desc2)
        H, _ = match_next_image(kps1, kps2, matches)
        homographies.append(H)

    cumulative_homographies = []
    cumulative_H = np.eye(3)
    for H in homographies:
        cumulative_H = cumulative_H.dot(H)
        cumulative_homographies.append(np.linalg.inv(cumulative_H.copy()))

    for i in range(len(cumulative_homographies)):
        print(f"Cumulative homography {i}:\n{cumulative_homographies[i]}\n")
    stitched = cv2.imread(imgs[-1])
    for i in trange(len(imgs) - 2, -1, -1):
        img_next = cv2.imread(imgs[i])
        H = cumulative_homographies[i]
        stitched = stitch(stitched, img_next, H)
    
    save(stitched, "stitched_full.jpg")

    

    # save out whole stitched panorama with all images