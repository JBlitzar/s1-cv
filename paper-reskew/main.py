# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "opencv-python",
#     "pillow",
# ]
# ///


import cv2
import numpy as np
from PIL import Image


def save(img, name="out.png"):
    im = Image.fromarray(img)  # (img * 255).astype(np.uint8)
    im.save(name)


img = np.array(Image.open("2.jpg"))
print(img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

gray = gray / np.max(gray) * 255


gray = (gray > 160) * 255

gray = gray.astype(np.uint8)

kernel = np.ones((5, 5), np.uint8)
gray = cv2.dilate(gray, kernel, iterations=5)
gray = cv2.erode(gray, kernel, iterations=5)


contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    cnt = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    corners = approx.reshape(-1, 2)



    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    for c in corners:
        cv2.circle(mask, tuple(c), 50, 120, -1)

    gray = mask
    if len(corners) == 4:

        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect

        rect = order_points(corners)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        unskewed = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        save(unskewed, "unskewed.png")

save(gray, "gray.png")


