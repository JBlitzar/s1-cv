from stitch_two_images import stitch, detect_and_compute, match_features, save, match_next_image, compute_homography
import cv2
import numpy as np
import glob
from tqdm import trange


def stitch_multi(imgs, homographies):
    """
    Stitch multiple images together using the provided homographies (each maps img_{i+1} -> img_i).
    """
    # Compute cumulative homographies mapping each image to the first image
    cumulative_H = [np.eye(3)]
    for H in homographies:
        cumulative_H.append(cumulative_H[-1] @ H)

    # Load all images
    images = [cv2.imread(p) for p in imgs]
    if any(im is None for im in images):
        raise ValueError("One or more images could not be loaded.")

    # Determine panorama bounding box
    all_corners = []
    for img, H in zip(images, cumulative_H):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped_corners)

    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translate = [-x_min, -y_min]
    H_translate = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]])

    pano_w = x_max - x_min
    pano_h = y_max - y_min
    result = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
    mask_accum = np.zeros((pano_h, pano_w), dtype=np.float32)

    # Warp and blend all images
    for img, H in zip(images, cumulative_H):
        H_final = H_translate @ H
        warped = cv2.warpPerspective(img, H_final, (pano_w, pano_h))
        mask = (warped > 0).astype(np.float32)[..., 0]  # 1 where non-black
        mask_accum += mask
        result += warped.astype(np.float32)

    mask_accum[mask_accum == 0] = 1
    result /= mask_accum[..., None]
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


if __name__ == "__main__":
    imgs = sorted(glob.glob("images/*.jpg"))
    homographies = []

    # Compute pairwise homographies (img_{i+1} -> img_i)
    for i in trange(len(imgs) - 1):
        img1, kps1, desc1 = detect_and_compute(imgs[i])
        img2, kps2, desc2 = detect_and_compute(imgs[i + 1])
        matches = match_features(desc1, desc2)
        H, _ = match_next_image(kps1, kps2, matches)
        H = np.linalg.inv(H)
        homographies.append(H)

    stitched = stitch_multi(imgs, homographies)
    save(stitched, "stitched_full.jpg")
