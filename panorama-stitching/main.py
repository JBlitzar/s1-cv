from stitch_two_images import stitch, detect_and_compute, match_features, save, match_next_image, compute_homography
import cv2
import numpy as np
import glob



if __name__ == "__main__":
    imgs = sorted(glob.glob("images/*.jpg"))

    # stitch 1 onto 2. Use that new image to stitch 3, etc.
    i1 = imgs[0]
    i2 = imgs[1]

    for i in range(2, len(imgs)+1):
        print(f"Stitching {i1} and {i2}...")
        img1, kps1, desc1 = detect_and_compute(i1)
        img2, kps2, desc2 = detect_and_compute(i2)

        matches = match_features(desc1, desc2)
        print(f"Matches found: {len(matches)}")

        H, _ = compute_homography(kps1, kps2, matches)
        
        if H is None:
            print("Not enough matches to compute homography.")
            break

        stitched = stitch(img1, img2, H)
        save(stitched, "stitched_temp.jpg")
        print("Saved stitched_temp.jpg")

        i1 = "stitched_temp.jpg"
        if i < len(imgs):
            i2 = imgs[i]