# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "opencv-python",
#     "pillow",
#     "tqdm",
# ]
# ///
import numpy as np
from PIL import Image
import os
import cv2
from copy import deepcopy
from tqdm import trange


def save(img):
    im = Image.fromarray(img)  # (img * 255).astype(np.uint8)
    im.save("out.png")


# read image
img = np.array(Image.open("mercator.png"))

if img.shape[2] == 4:
    img = img[:, :, :3]


# Claude Sonnet 4 via github copilot used to refactor `oop.py` into the file seen here (faster I guess, less oop.)
# prompt used: eliminate the pixel class and instead operate on rows of pixels at a time. So store energies in a numpy array, cumulative energies in a numpy array, and parents in a numpy array
class PixelGrid:
    def __init__(self, img, energies=None):
        if energies is None:
            energies = cv2.Canny(img, 100, 200)

        self.energies = energies.astype(np.float64)
        self.img = img
        self.height, self.width = energies.shape

        # Initialize cumulative energy array and parent tracking
        self.cumulative = np.full((self.height, self.width), np.inf)
        self.parents = np.full((self.height, self.width), -1, dtype=np.int32)
        self.seam = None

    def populate(self):
        # Initialize first row
        self.cumulative[0, :] = self.energies[0, :]

        # Process each row
        for y in range(1, self.height):
            for x in range(self.width):
                # Find minimum cumulative energy from possible parents
                min_energy = np.inf
                best_parent = -1

                # Check three possible parent positions
                for dx in [-1, 0, 1]:
                    parent_x = x + dx
                    if 0 <= parent_x < self.width:
                        parent_energy = self.cumulative[y - 1, parent_x]
                        if parent_energy < min_energy:
                            min_energy = parent_energy
                            best_parent = parent_x

                self.cumulative[y, x] = min_energy + self.energies[y, x]
                self.parents[y, x] = best_parent

        # Find the seam by backtracking from minimum in last row
        last_row = self.cumulative[-1, :]
        min_x = np.argmin(last_row)

        # Backtrack to find the seam
        seam = []
        current_x = min_x
        for y in range(self.height - 1, -1, -1):
            seam.append((current_x, y))
            if y > 0:
                current_x = self.parents[y, current_x]

        self.seam = seam[::-1]  # Reverse to get top-to-bottom order

    def visualize(self):
        img2 = self.img.copy()
        for x, y in self.seam:
            img2[y, x] = [255, 0, 255]
        # save(img2)

    def remove_seam(self):
        # Create new image without the seam
        new_img = np.zeros(
            (self.height, self.width - 1, self.img.shape[2]), dtype=self.img.dtype
        )

        for y in range(self.height):
            seam_x = self.seam[y][0]
            # Copy pixels before seam
            new_img[y, :seam_x] = self.img[y, :seam_x]
            # Copy pixels after seam
            new_img[y, seam_x:] = self.img[y, seam_x + 1 :]

        # save(new_img)
        return new_img


grid = PixelGrid(img)
grid.populate()
grid.visualize()
new = grid.remove_seam()

desired_width = img.shape[1] // 2

cur = new
for i in trange(img.shape[1] - desired_width):
    grid = PixelGrid(cur)
    grid.populate()
    cur = grid.remove_seam()

# rotate 90
cur = np.rot90(cur, 1, (0, 1))
for i in trange(cur.shape[1] - cur.shape[1] // 2):
    grid = PixelGrid(cur)
    grid.populate()
    cur = grid.remove_seam()

cur = np.rot90(cur, 3, (0, 1))

save(cur)
