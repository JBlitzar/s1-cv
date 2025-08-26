# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "opencv-python",
#     "pillow",
# ]
# ///
import numpy as np
from PIL import Image
import os
import cv2

def save(img):
    im = Image.fromarray(img) # (img * 255).astype(np.uint8)
    im.save("out.png")


# read image
img = np.array(Image.open("image.png"))
# get energies
energies = cv2.Canny(img, 100, 200)
print(energies.shape) # 651, 960


# get list of seams (insane dynamic programming algorithm?)
# https://en.wikipedia.org/wiki/Seam_carving#Dynamic_programming

class PixelGrid:
    def __init__(self,img):

        self.pixels = np.array([[Pixel(x,y,self) for x in range(len(img[y]))] for y in range(len(img))])
        self.img = img
        self.seam = None

    def get_pixel(self, x, y):
        # unsafe fn
        return self.pixels[y][x]
    
    def populate(self):
        for row in range(len(energies)):
            #print(row)
            if row != 0:
                for pixel in self.pixels[row]:
                    pixel.post_reception()
            for pixel in self.pixels[row]:
                pixel.advertise_around()

        last_row = self.pixels[-1]
        lowest = None
        lowest_value = float("inf")
        for pixel in last_row:
            if pixel.cumulative < lowest_value:
                lowest = pixel
                lowest_value = pixel.cumulative
        
        #print(lowest_value)
        cur = lowest
        seam = []
        for _ in range(len(energies)):
            seam.append(cur)
            # print(cur)
            # print(cur.cumulative)
            # print(cur.energy)
            cur = cur.parent
        #print(seam)
        self.seam = seam
    

    def visualize(self):
        img2 = self.img.copy()
        for pixel in self.seam:
            img2[pixel.y, pixel.x] = 120
        
        save(img2)


class Pixel: 
    def __init__(self, x, y,grid):
        self.x = x
        self.y = y
        self.energy = energies[y][x]
        self.grid = grid

        self.cumulative = self.energy
        self.records = []
        self.parent_records = []
        self.parent = None

    def advertise_around(self):
        self.advertise_to_pixel( self.x, self.y + 1)
        self.advertise_to_pixel( self.x - 1, self.y + 1)
        self.advertise_to_pixel( self.x + 1, self.y + 1)


    def post_reception(self):
        self.cumulative = min(self.records)
        self.parent = self.parent_records[self.records.index(self.cumulative)]
        del self.records, self.parent_records

    def advertise_to_pixel(self, px, py):

        #_dummy = energies[px][py] # to get indexerror, I guess?
        #print(px,py,self.x, self.y)
        if px < 0 or py < 0 or py >= len(energies) or px >= len(energies[0]):
            return
        if py >= len(energies) or px >= len(energies[0]):
            return
        other = self.grid.get_pixel(px, py)
        other.receive(self)


    def receive(self, other):
        #print("receive", self.x, self.y, other.x, other.y)
        self.records.append(other.cumulative + self.energy) 
        self.parent_records.append(other)

    def __repr__(self):
        return f"Pixel({self.x},{self.y})"

grid = PixelGrid(energies)
grid.populate()
grid.visualize()
# remove low-energy seams
# repeat
