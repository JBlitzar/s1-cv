# Panorama Stitching

Welcome to the subfolder! You'll find development notes in NOTES.md and code in main.py. `uv run main.py` to run. I'll hopefully update this with more explanations on the code and the images that it produces, but I'll try to write in NOTES.md as I go so you'll be able to see that.

Main code is in `main.py`

Algorithm works as such:

- Identity keypoints in each image
- Obtain feature descriptor for each keypoint
- Use RANSAC algorithm to correlate points with similar feature descriptors and obtain projection matrix H
- Apply H to stitch the two images together.

Keypoints should be outputted at `keypoints.png` when you run `main.py`

Source images are located in `images/`. `sample_imgs/` is for feature descriptor testing purposes.
