import cv2
import numpy as np

video_path = "test1.mp4"

cap = cv2.VideoCapture(video_path)
ret, prev_frame = cap.read()

if not ret:
    print("Error reading video")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_sobel = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 1, ksize=3)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=3)
    
    delta = cv2.absdiff(sobel, prev_sobel)

    delta /= np.max(delta) + 1e-10  # avoid division by zero
    delta *= 255.0
    delta = np.uint8(delta)


    
    cv2.imshow('Frame Delta', delta)

    cv2.imshow("Sobel", cv2.convertScaleAbs(sobel))

    cv2.imshow("Original Frame", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    prev_sobel = sobel

cap.release()
cv2.destroyAllWindows()