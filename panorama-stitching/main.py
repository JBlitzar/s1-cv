import cv2
import numpy as np
from PIL import Image

def save(img, name="out.jpg"):
    im = Image.fromarray(img)  # (img * 255).astype(np.uint8)
    im.save(name)

img = np.array(Image.open("images/1.jpg"))

print(img.shape)
features = cv2.goodFeaturesToTrack(img, maxCorners=100, qualityLevel=0.01, minDistance=10)
features = np.int0(features)
for f in features:
    x, y = f.ravel()
    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
save(img, "features.png")