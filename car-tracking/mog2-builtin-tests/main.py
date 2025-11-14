import cv2

cap = cv2.VideoCapture("../test1.mp4")

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500,     
    varThreshold=10,
    detectShadows=False
)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = gray
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    mask = fgbg.apply(img)

    # # clean up noise, isolate coherent blobs
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    

    cv2.imshow("Raw Frame", frame)
    cv2.imshow("Processed Frame", img)
    cv2.imshow("Motion Mask", mask)

    superimposed = img.copy()
    superimposed = cv2.cvtColor(superimposed, cv2.COLOR_GRAY2BGR)
    superimposed[mask == 255] = [0, 0, 255]  # Red tint for motion pixels
    cv2.imshow("Superimposed", superimposed)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
