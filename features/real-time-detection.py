from ultralytics import YOLO
import cv2

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Open video
cap = cv2.VideoCapture("4K Road traffic video for object detection and tracking - free download now.mp4")

# Classes we want to detect
vehicle_classes = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO
    results = model(frame, verbose=False)

    for result in results:

        boxes = result.boxes

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Only vehicles
            if class_id in vehicle_classes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = vehicle_classes[class_id]

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw label
                text = f"{label} {confidence:.2f}"

                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # Show result
    cv2.imshow("Traffic Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()