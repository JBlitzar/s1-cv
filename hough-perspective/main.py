import numpy as np
from PIL import Image
import os
import cv2
from copy import deepcopy
from tqdm import trange
from sklearn.cluster import KMeans


def save(img, file="out.png"):
    im = Image.fromarray(img)  # (img * 255).astype(np.uint8)
    im.save(file)


# read image
img = np.array(Image.open("road.png"))


if img.shape[2] == 4:
    img = img[:, :, :3]

image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(image, 200, 350, apertureSize=3)
def convolution2d(image, kernel, bias=0):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel) + bias
    return new_image

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])
image = np.array(Image.open("road.png").convert("L"))
grad_x = convolution2d(image, sobel_x)
grad_y = convolution2d(image, sobel_y)
sobel = np.sqrt(grad_x**2 + grad_y**2)
save(edges,"edges.png")
sobel = (sobel / sobel.max() * 255)

save(sobel.astype(np.uint8),"sobel.png")

lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)

if lines is not None:
    img_lines = img.copy()
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    save(img_lines, "lines.png")
