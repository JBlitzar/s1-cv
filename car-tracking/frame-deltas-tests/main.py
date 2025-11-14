import cv2
import numpy as np

video_path = "../test1.mp4"

cap = cv2.VideoCapture(video_path)
ret, prev_frame = cap.read()

if not ret:
    print("Error reading video")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_sobel = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 1, ksize=3)
prev_deltas = []  # store the last two deltas (so we can check across 3 frames)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=3)
    
    delta = cv2.absdiff(sobel, prev_sobel)

    # normalize and convert to uint8 for masking/display
    delta /= np.max(delta) + 1e-10  # avoid division by zero
    delta *= 255.0
    delta = np.uint8(delta)

    # persistent motion across 3 frames: current delta AND previous two deltas
    persistent_motion_3 = np.zeros_like(delta)
    if len(prev_deltas) >= 2:
        current_mask = delta > 0
        prev_mask_1 = prev_deltas[-1] > 0
        prev_mask_2 = prev_deltas[-2] > 0
        persistent_mask_3 = current_mask & prev_mask_1 & prev_mask_2
        persistent_motion_3[persistent_mask_3] = 255

    # update delta buffer (keep last two)
    prev_deltas.append(delta.copy())
    if len(prev_deltas) > 2:
        prev_deltas.pop(0)

    cv2.imshow('Frame Delta', delta)
    cv2.imshow('Persistent Motion (3-frame)', persistent_motion_3)
    cv2.imshow("Sobel", cv2.convertScaleAbs(sobel))
    cv2.imshow("Original Frame", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    prev_sobel = sobel

cap.release()
cv2.destroyAllWindows()