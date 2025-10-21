# Panorama stitching notes

Did some research. Here's the general outline of how this should work.

 - Get keypoints. Areas with interesting "meaningful information"
 - get descriptors for those keypoints. Descriptors should somehow uniquely describe the keypoint and still be resistant to rotation, brightness changes, scaling, and affine transformations.
 - Funny RANSAC algorithm: choose four random points, get homography, compute reprojection error (keypoints match keypoints with similar descriptors), repeat N times and use best result. 