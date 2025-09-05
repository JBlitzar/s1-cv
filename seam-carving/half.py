# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "imageio",
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
import imageio


def save(img, name="out.png"):
    im = Image.fromarray(img)  # (img * 255).astype(np.uint8)
    im.save(name)


# read image
img = np.array(Image.open("minion.png"))

if img.shape[2] == 4:
    img = img[:, :, :3]


# Claude Sonnet 4 via github copilot used to refactor `oop.py` into the file seen here (faster I guess, less oop.)
# prompt used: eliminate the pixel class and instead operate on rows of pixels at a time. So store energies in a numpy array, cumulative energies in a numpy array, and parents in a numpy array
# After looking at the implementation, I re-coded populate() myself
class PixelGrid:
    def __init__(self, img, energies=None):
        if energies is None:
            energies = cv2.Canny(img, 100, 200)

        self.energies = energies.astype(np.float64)
        self.img = img
        self.height, self.width = energies.shape

        self.cumulative = np.full((self.height, self.width), np.inf) # cumulative energies
        self.parents = np.full((self.height, self.width), -1, dtype=np.int32)
        self.seam = None

    

    def populate(self):
        self.cumulative[0, :] = self.energies[0, :]


        for row in range(self.height):
            #print(row)
            # if row != 0:
            #     for pixel in self.pixels[row]:
            #         pixel.post_reception()
            # for pixel in self.pixels[row]:
            #     pixel.advertise_around()
            for x in range(self.width):
                min_energy = np.inf
                best_parent = -1

                for offset in [-1,0,1]:
                    # scan parent positions
                    potential_y = row - 1
                    potential_x = x + offset
                    try:
                        energy = self.cumulative[potential_y][potential_x]
                        if energy < min_energy:
                            best_parent = offset
                            min_energy = energy
                    except IndexError:
                        pass
                
                self.cumulative[row][x] = min_energy + self.energies[row][x]
                self.parents[row][x] = best_parent

        last_row = self.cumulative[-1,:]
        min_x = np.argmin(last_row)

        seam = []
        current_x = min_x
        for y in range(self.height - 1, -1, -1):
            seam.append((current_x, y))
            if y > 0:
                current_x += self.parents[y, current_x]

        self.seam = seam[::-1]

    # def visualize(self):
    #     img2 = self.img.copy()
    #     for pixel in self.seam:
    #         img2[pixel.y, pixel.x] = [255,0,255]
        
    #     save(img2)

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



def remove_seam_horizontal(image):
    rotated = np.rot90(image, 1, (0, 1))
    grid = PixelGrid(rotated)
    grid.populate()

    new_img = grid.remove_seam()
    result = np.rot90(np.array(new_img), 3, (0, 1))
    return result
def remove_seam_vertical(image):
    grid = PixelGrid(image)
    grid.populate()

    new_img = grid.remove_seam()
    return np.array(new_img)

N=100


for i in trange(1, N):

    img = remove_seam_horizontal(img).astype("uint8")

    img = remove_seam_vertical(img).astype("uint8")

    save(img, f"frame_{i:03d}.png")

# Gather all generated frames
frames = []
original_img = np.array(Image.open("minion.png"))
orig_h, orig_w = original_img.shape[:2]

for i in range(1, N):
    frame = Image.open(f"frame_{i:03d}.png")
    frame_w, frame_h = frame.size

    # Create new white background
    new_frame = Image.new("RGB", (orig_w, orig_h), (255, 255, 255))
    # Center the frame
    x_offset = (orig_w - frame_w) // 2
    y_offset = (orig_h - frame_h) // 2
    new_frame.paste(frame, (x_offset, y_offset))
    frames.append(new_frame)

# Save as GIF
frames[0].save(
    "out.gif",
    save_all=True,
    append_images=frames[1:],
    duration=50,
    loop=0,
)

# Remove individual frame images
for i in range(1, 99):
    os.remove(f"frame_{i:03d}.png")


