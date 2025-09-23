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

min_x = np.min(votes[:,0])
max_x = np.max(votes[:,0])
min_y = np.min(votes[:,1])
max_y = np.max(votes[:,1])
print(min_x, max_x, min_y, max_y)
step_size = 10
x_bins = np.arange(min_x, max_x, step_size)
y_bins = np.arange(min_y, max_y, step_size)

new_votes = []
new_vote_vote_counts = []
for x in x_bins:
    for y in y_bins:
        new_vote_vote_counts.append(0)
        new_votes.append((x,y))
        for vote in votes:
            if (vote[0]-x)**2 + (vote[1]-y)**2 < (step_size/2)**2:
                new_vote_vote_counts[-1] += 1

newer_votes = [] 
for vote in new_votes:
    real_votes_this_vote = []
    for real_vote in votes:
        if (vote[0] - real_vote[0])**2 + (vote[1] - real_vote[1])**2 < (step_size/2)**2:
            real_votes_this_vote.append(real_vote)

    if len(real_votes_this_vote) > 0:
        newer_votes.append(np.mean(real_votes_this_vote, axis=0))
    else:
        newer_votes.append(vote)

new_votes = newer_votes

print("got new votes")
#print(new_votes, new_vote_vote_counts)
new_votes = np.array(new_votes)
print(new_vote_vote_counts)
order = np.argsort(new_vote_vote_counts)
print(order)
order = order.tolist()[::-1][:30]
print("asdf")
print(order)
for o in order:
    print(new_votes[o], new_vote_vote_counts[o])


# Begin ai-generated display code
# Expand canvas to fit vanishing points
vanishing_points = [new_votes[o] for o in order]
vanishing_points = np.array(vanishing_points)

# Find min/max coordinates to determine needed canvas size
all_points = np.vstack([[[0, 0], [hough_vis.shape[1], hough_vis.shape[0]]], vanishing_points])
min_x = int(np.floor(np.min(all_points[:, 0])))
max_x = int(np.ceil(np.max(all_points[:, 0])))
min_y = int(np.floor(np.min(all_points[:, 1])))
max_y = int(np.ceil(np.max(all_points[:, 1])))

# Compute offsets if vanishing points are outside the image
offset_x = -min(0, min_x)
offset_y = -min(0, min_y)
new_w = max(max_x + offset_x, hough_vis.shape[1] + offset_x)
new_h = max(max_y + offset_y, hough_vis.shape[0] + offset_y)

# Ensure new_h and new_w are at least as large as the original image plus offsets
canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
canvas[offset_y:offset_y + hough_vis.shape[0], offset_x:offset_x + hough_vis.shape[1]] = hough_vis

# Draw vanishing points
for pt in vanishing_points:
    x, y = int(pt[0] + offset_x), int(pt[1] + offset_y)
    cv2.circle(canvas, (x, y), 10, (0, 0, 255), -1)
    cv2.putText(canvas, f"({int(pt[0])}, {int(pt[1])})", (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

save(canvas, "vanishing_points.png")

# end ai-generated display code


image_center = np.array([hough_vis.shape[1]/2, hough_vis.shape[0]/2])
def get_angle(point):
    direction = point - image_center
    return np.atan2(direction[1], direction[0])

top_point = vanishing_points[0]
direction = get_angle(top_point)
print("direction", direction / np.pi * 180)

for vp in vanishing_points:
    angle = get_angle(vp)
    if abs(angle - direction) > np.pi / 2:
        vp2 = vp
        break
print("vp2 angle", get_angle(vp2) / np.pi * 180)

