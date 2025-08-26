import numpy as np
from PIL import Image
import os
# read image
img = np.array(Image.open("image.png"))
# get gradient magnitude for saliency
print(img.shape)
gradient_horiz = np.zeros_like(img)[:,:,0]
for row in range(len(img)):
    above_row = img[row - 1] if row > 0 else np.zeros_like(row)
    below_row = img[row + 1] if row < len(img) - 1 else np.zeros_like(row)
    gradient_horiz[row] = np.sum(np.abs(above_row - below_row), axis=1)

gradient_vert = np.zeros_like(img)[:,:,0]
for col in range(len(img[0])):
    left_col = img[:, col - 1] if col > 0 else np.zeros_like(img[:, col])
    right_col = img[:, col + 1] if col < len(img[0]) - 1 else np.zeros_like(img[:, col])
    gradient_vert[:, col] = np.sum(np.abs(left_col - right_col), axis=1)

gradient = np.sqrt(gradient_horiz**2 + gradient_vert**2)

# save as image
Image.fromarray((gradient * 255).astype(np.uint8)).save("gradient.png")

# get list of seams (insane dynamic programming algorithm?)
# https://en.wikipedia.org/wiki/Seam_carving#Dynamic_programming
# remove low-energy seams
# repeat