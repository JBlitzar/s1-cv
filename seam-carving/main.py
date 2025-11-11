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
# After looking at the implementation, I re-coded populate() myself
# I subsequently modified the code a lot to make it actually work, haha
class PixelGrid:
    def __init__(self, img, energies=None):
        if energies is None:
            energies = cv2.Canny(np.array(img, dtype="uint8"), 100, 200)

        self.energies = energies.astype(np.float64)
        self.img = np.array(img, dtype="uint8")
        self.height, self.width = energies.shape

        self.cumulative = np.full(
            (self.height, self.width), np.inf
        )  # cumulative energies
        self.parents = np.full((self.height, self.width), -1, dtype=np.int32)
        self.seam = None

    def populate(self):
        self.cumulative[0, :] = self.energies[0, :]

        for row in range(1, self.height):
            for x in range(self.width):
                min_energy = np.inf
                best_parent = 0

                for offset in [-1, 0, 1]:
                    potential_y = row - 1
                    potential_x = x + offset

                    if 0 <= potential_x < self.width:
                        energy = self.cumulative[potential_y][potential_x]
                        if energy < min_energy:
                            best_parent = offset
                            min_energy = energy

                self.cumulative[row][x] = min_energy + self.energies[row][x]
                self.parents[row][x] = best_parent

        last_row = self.cumulative[-1]
        min_x = np.argmin(last_row)

        seam = []
        current_x = min_x
        for y in range(self.height - 1, -1, -1):
            seam.append((current_x, y))
            if y > 0:
                current_x += self.parents[y, current_x]

        self.seam = seam[::-1]

    def visualize(self):
        img2 = self.img.copy()
        for pixel in self.seam:
            img2[pixel[1], pixel[0]] = [255, 0, 255]

        save(img2)

    def remove_seam(self):
        img2 = self.img.copy().tolist()
        seam_rev = self.seam[::-1]

        new_rows = []

        for row_idx in range(len(img2)):
            row = deepcopy(img2[row_idx])
            p = seam_rev[row_idx]
            del row[p[0]]

            new_rows.append(row)
        save(np.array(new_rows, dtype="uint8"))
        return new_rows


grid = PixelGrid(img)
grid.populate()
grid.visualize()
new = grid.remove_seam()

cur = new
for i in trange(200):
    grid = PixelGrid(cur)
    grid.populate()
    grid.visualize()
    cur = grid.remove_seam()
    cur = np.array(cur, dtype="uint8")
