import cv2
import numpy as np
from PIL import Image

def save(img, name="out.jpg"):
    im = Image.fromarray(img)  # (img * 255).astype(np.uint8)
    im.save(name)

img_orig = np.array(Image.open("images/1.jpg"), dtype=np.uint8)

print(img_orig.shape)
img = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
img = (img / np.max(img) * 255).astype(np.uint8)
img = cv2.equalizeHist(img) # attempt to make features invariant to lighting changes
features = cv2.goodFeaturesToTrack(img,100,0.01,10)
features = np.int32(features)

drawing_img = img_orig.copy()
for f in features:
    x, y = f.ravel()
    cv2.circle(drawing_img, (x, y), 10, (0, 255, 0), -1)
save(drawing_img, "features.png")


def descriptor(img, x, y, size=16):
    patch = img[y - size//2:y + size//2, x - size//2:x + size//2]
    patch /= np.linalg.norm(patch) # brightness check!
    #??? do something so it's invariant to rotation
    # rotate it based on something. Gradient?

    gy, gx = np.gradient(patch)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    # pretty sure SIFT does something similar to this
    hist, bin_edges = np.histogram(angle, bins=36, range=(-np.pi, np.pi), weights=magnitude)
    dominant_angle = bin_edges[np.argmax(hist)]

    M = cv2.getRotationMatrix2D((size//2, size//2), dominant_angle * 180/np.pi, 1)
    patch = cv2.warpAffine(patch, M, (size, size))

    # NOT scale-invariant but hopefully doesn't really matter on a small enough patch size. Fine tune size to get enough detail without noticing scale changes

    desc = patch.flatten()
    return desc
