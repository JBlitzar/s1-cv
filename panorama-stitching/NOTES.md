# Panorama stitching notes

Did some research. Here's the general outline of how this should work.

- Get keypoints. Areas with interesting "meaningful information"
- get descriptors for those keypoints. Descriptors should somehow uniquely describe the keypoint and still be resistant to rotation, brightness changes, scaling, and affine transformations.
- Funny RANSAC algorithm: choose four random points, get homography, compute reprojection error (keypoints match keypoints with similar descriptors), repeat N times and use best result.

Got feature detection working with `cv2.goodFeaturesToTrack`. I'll try to implement the descriptor algorithm by myself though because that's the heart of this. For the same reason I didn't recode the convolution operation when doing canny edge detector but I still didn't just use cv2.cannyEdgeDetector. But I did use cv2.cannyedgedetector for seam carving because that's just the starting step.

Implemented it fully, but seems like descriptor computation is a bit lacking currently

- Descriptor distance test for same image: 2466.896025372776
- Descriptor distance test for different image: 2192.1872638987757

Used AI to try the opencv builtin SIFT and it worked so now I have to debug my algorithm.

Working on incrementally adding back stuff from main.py into stitch-builtin until stuff breaks.

My builtin RANSAC doesn't work and I don't know why. Maybe it's just not enough iterations? I timed stuff to identify bottlenecks.

Soo I just vectorized with numpy and increased the number of keypoints. Except now all of the points are collapsing??

I've now spent an hour on this and still don't know why it's not working. Yes, I tried asking AI.

---

I've gotten the homemade RANSAC implementation to work! I've migrated everything to main.py and deleted stitch-builtin.

Using cv2 builtin to find descriptors and correlate the keypoints, but I implemented RANSAC + all the glue code myself, because I feel like those are the most important things to implement to understand what's going on. And the results look pretty great! see `stitched_custom.jpg`. 

Works pretty well. I rewrote the README a bit but will need to rewrite it again because I changed the API / entry points / added more files.

TBD because atm stitching all the images together produces a... less than desireable result. Integrated error + overwriting as opposed to blending, I guess. I'll investigate further. 

Iff *everything* works and I still have time, I'll add back my scuffed feature descriptor implementation. 