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
        elif 67.5 < angle < 112.5:
            dy = -1
            dx = 0
        else:
            dy = -1
            dx = -1
        
        before = 0
        try:
            before = image[y + dy][x + dx]
        except IndexError:
            pass

        after = 0
        try:
            after = image[y - dy][x - dx]
        except IndexError:
            pass

        if edge_strength < before or edge_strength < after:
            image[y, x] = 0

HIGH_THRESH = 200
LOW_THRESH = 20
for y, row in enumerate(image):
    for x, pixel in enumerate(row):
        if pixel > HIGH_THRESH:
            image[y][x] = 1
        elif pixel > LOW_THRESH:
            image[y][x] = 0.5
        else:
            image[y][x] = 0

flag = True
while flag: 
    flag = False
    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            if pixel == 0.5:
                def has_neighborhood():
                    dys = [-1,0,1]
                    dxs = [-1,0,1]
                    for dy in dys:
                        for dx in dxs:
                            try:
                                if image[y+dy][x+dx] == 1:
                                    return True
                            except IndexError:
                                pass
                    return False
                if has_neighborhood():
                    image[y][x] = 1
                    flag = True


for y, row in enumerate(image):
    for x, pixel in enumerate(row):
        if pixel == 0.5:
            image[y][x] = 0

save(image * 255)
