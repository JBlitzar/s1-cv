import numpy as np
from PIL import Image
import os
import cv2
from copy import deepcopy
from tqdm import trange
from sklearn.cluster import KMeans
from tqdm import tqdm

def save(img, file="out.png"):
    im = Image.fromarray(img)  # (img * 255).astype(np.uint8)
    im.save(file)


# read image
img = np.array(Image.open("cube.png"))


if img.shape[2] == 4:
    img = img[:, :, :3]

image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# apply Gaussian blur to reduce noise before edge detection
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

lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

hough_vis = deepcopy(img)
# For each line.
# We want lines that are long. How do we determine that?
# For each line, count how many edge pixels are like within three pixels of the line.
# Establish end points, then if the line is long enough, draw it.
_linelist = []
if lines is not None:
    for l in lines:
        rho, theta = l[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        _linelist.append(((x1, y1), (x2, y2)))
lines = _linelist
save(hough_vis, "hough_lines_all.png")



# https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

votes = []
for line1 in lines:
    for line2 in lines:
        if line1 != line2:
            try:
                x, y = line_intersection(line1, line2)
                votes.append((x, y))
            except Exception as e:
                pass

votes = np.array(votes)
print(votes)
vote_votes_es = []
for x, y in votes:
    vote_votes = 0
    for vx,vy in votes:
        if (vx-x)**2 + (vy-y)**2 < 5**2:
            vote_votes += 1
    vote_votes_es.append(vote_votes)

vote_votes_es = np.array(vote_votes_es)
new_votes = []
new_vote_vote_counts = []
for idx, vote in enumerate(votes):
    flag = False
    for new_vote in new_votes:
        if (new_vote[0]-vote[0])**2 + (new_vote[1]-vote[1])**2 < 5**2:
            flag = True
    if not flag:
        new_votes.append(vote)
        new_vote_vote_counts.append(vote_votes_es[idx])
print(new_votes, new_vote_vote_counts)
order = np.argsort(-1 * new_vote_vote_counts)
order = order.tolist()
votes_sorted = new_votes[order]
vote_counts_sorted = new_vote_vote_counts[order]

print(votes_sorted[:3])
