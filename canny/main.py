# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pillow",
# ]
# ///

from PIL import Image
import numpy as np

def save(img):
    im = Image.fromarray(img.astype(np.uint8))  # (img * 255).astype(np.uint8)
    im.save("out.png")

image = np.array(Image.open("camera.png").convert("L"))


# 1. Apply gaussian filter to smooth the image

# https://stackoverflow.com/questions/2448015/2d-convolution-using-python-and-numpy
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

gaussian_filter = np.array([[2, 4, 5, 4, 2],[4, 9, 12, 9, 4], [5, 12, 15, 12, 5],[4, 9, 12, 9, 4],[2, 4, 5, 4, 2]])

image = (1/159) * convolution2d(image, gaussian_filter)


sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

grad_x = convolution2d(image, sobel_x)
grad_y = convolution2d(image, sobel_y)
image = np.sqrt(grad_x**2 + grad_y**2)

theta = np.atan2(grad_y, grad_x)


for y, row in enumerate(image):
    for x, pixel in enumerate(row):
        edge_strength = pixel
        angle = (theta[y][x] * 180.0 / np.pi +  + 180) % 180 

        dy = 0
        dx = 0

        if angle < 22.5 or angle > 157.5:
            dy = 0
            dx = 1
        elif 22.5 < angle < 67.5:
            dy = -1
            dx = 1

        


        if (0 <= angle < 22.5) or (157.5 <= angle < 180):
            before = image[y, x-1] if x-1 >= 0 else 0
            after = image[y, x+1] if x+1 < image.shape[1] else 0
        elif (22.5 <= angle < 67.5):
            before = image[y-1, x+1] if y-1 >= 0 and x+1 < image.shape[1] else 0
            after = image[y+1, x-1] if y+1 < image.shape[0] and x-1 >= 0 else 0
        elif (67.5 <= angle < 112.5):
            before = image[y-1, x] if y-1 >= 0 else 0
            after = image[y+1, x] if y+1 < image.shape[0] else 0
        else:
            before = image[y-1, x-1] if y-1 >= 0 and x-1 >= 0 else 0
            after = image[y+1, x+1] if y+1 < image.shape[0] and x+1 < image.shape[1] else 0

        if edge_strength < before or edge_strength < after:
            image[y, x] = 0



save(image)
