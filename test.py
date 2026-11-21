from ultralytics import YOLO
import cv2

# =========================
# Load Models
# =========================

traffic_model = YOLO("yolo11s.pt")
emergency_model = YOLO("Emergency_Detection.pt")


# =========================
# Video
# =========================

video_path = "test (2).mp4"

cap = cv2.VideoCapture(video_path)


# =========================
# Classes - Traffic Model
# =========================

traffic_classes = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


# =========================
# Emergency Classes
# =========================

emergency_classes = {
    0: "Ambulance",
    1: "Fire truck",
    2: "Police car"
}


# =========================
# Video Loop
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # =====================================
    # 1. Normal Traffic Tracking
    # =====================================

    traffic_results = traffic_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        iou=0.5,
        classes=list(traffic_classes.keys()),
        verbose=False
    )


    # =====================================
    # Draw Traffic Objects
    # =====================================

    if traffic_results[0].boxes.id is not None:

        boxes = traffic_results[0].boxes

        for box, track_id, cls, conf in zip(
            boxes.xyxy,
            boxes.id,
            boxes.cls,
            boxes.conf
        ):

            x1, y1, x2, y2 = map(int, box)

            track_id = int(track_id)
            cls = int(cls)

            name = traffic_classes.get(cls, "Unknown")

            # Center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{name} ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                4,
                (0, 0, 255),
                -1
            )


    # =====================================
    # 2. Emergency Vehicle Tracking
    # =====================================

    emergency_results = emergency_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        iou=0.5,
        verbose=False
    )


    # =====================================
    # Draw Emergency Vehicles
    # =====================================

    if emergency_results[0].boxes.id is not None:

        boxes = emergency_results[0].boxes

        for box, track_id, cls, conf in zip(
            boxes.xyxy,
            boxes.id,
            boxes.cls,
            boxes.conf
        ):

            x1, y1, x2, y2 = map(int, box)

            track_id = int(track_id)
            cls = int(cls)

            name = emergency_classes.get(cls, "Emergency")

            # Center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                f"{name} ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                5,
                (255, 0, 0),
                -1
            )


    # =====================================
    # Display
    # =====================================

    cv2.imshow("Smart Traffic Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()