# Panorama stitching notes

Did some research. Here's the general outline of how this should work.

 - Get keypoints. Areas with interesting "meaningful information"
 - get descriptors for those keypoints. Descriptors should somehow uniquely describe the keypoint and still be resistant to rotation, brightness changes, scaling, and affine transformations.
 - Funny RANSAC algorithm: choose four random points, get homography, compute reprojection error (keypoints match keypoints with similar descriptors), repeat N times and use best result. 


Got feature detection working with `cv2.goodFeaturesToTrack`. I'll try to implement the descriptor algorithm by myself though because that's the heart of this. For the same reason I didn't recode the convolution operation when doing canny edge detector but I still didn't just use cv2.cannyEdgeDetector. But I did use cv2.cannyedgedetector for seam carving because that's just the starting step. 