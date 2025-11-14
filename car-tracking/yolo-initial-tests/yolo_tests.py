import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolo11n.pt')  # or yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Open video capture (0 for webcam, or path to video file)
cap = cv2.VideoCapture("test.png")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO inference
    results = model(frame)
    
    # Process results
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                # Get class ID and check if it's a car (class 2 in COCO dataset)
                class_id = int(box.cls[0])
                if class_id == 2:  # car class
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Print coordinates
                    print(f"Car detected: x1={x1}, y1={y1}, x2={x2}, y2={y2}, confidence={confidence:.2f}")
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Car {confidence:.2f}', (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Car Detection', frame)

    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

i = input()
cap.release()
cv2.destroyAllWindows()