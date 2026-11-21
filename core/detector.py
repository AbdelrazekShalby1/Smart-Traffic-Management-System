from ultralytics import YOLO


class TrafficDetector:
    """
    Detection module for the Smart Traffic Management System.

    Responsibilities:
    - Load traffic YOLO model
    - Load emergency vehicle YOLO model
    - Perform detection only
    - Return standardized detection results

    Tracking is NOT handled here.
    """

    # ==========================================
    # Traffic Classes
    # ==========================================

    TRAFFIC_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    # ==========================================
    # Emergency Classes
    # ==========================================

    EMERGENCY_CLASSES = {
        0: "Ambulance",
        1: "Fire truck",
        2: "Police car"
    }

    def __init__(
        self,
        traffic_model_path="yolo11s.pt",
        emergency_model_path="Emergency_Detection.pt",
        traffic_conf=0.4,
        emergency_conf=0.4,
        iou=0.5
    ):

        # ======================================
        # Settings
        # ======================================

        self.traffic_conf = traffic_conf
        self.emergency_conf = emergency_conf
        self.iou = iou

        # ======================================
        # Load Models
        # ======================================

        print("Loading traffic detection model...")

        self.traffic_model = YOLO(
            traffic_model_path
        )

        print("Loading emergency detection model...")

        self.emergency_model = YOLO(
            emergency_model_path
        )

        print("Detection models loaded successfully.")

    # ==========================================
    # Calculate Center
    # ==========================================

    @staticmethod
    def get_center(bbox):
        """
        Calculate the center point of a bounding box.

        bbox:
        [x1, y1, x2, y2]
        """

        x1, y1, x2, y2 = bbox

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        return cx, cy

    # ==========================================
    # Traffic Detection
    # ==========================================

    def detect_traffic(self, frame):
        """
        Detect normal traffic objects.

        Returns:
            List of detections.
        """

        results = self.traffic_model.predict(
            frame,
            conf=self.traffic_conf,
            iou=self.iou,
            classes=list(self.TRAFFIC_CLASSES.keys()),
            verbose=False
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        boxes = result.boxes

        for box, cls, conf in zip(
            boxes.xyxy,
            boxes.cls,
            boxes.conf
        ):

            x1, y1, x2, y2 = map(
                int,
                box.tolist()
            )

            cls = int(cls)
            confidence = float(conf)

            name = self.TRAFFIC_CLASSES.get(
                cls,
                "Unknown"
            )

            bbox = [
                x1,
                y1,
                x2,
                y2
            ]

            cx, cy = self.get_center(
                bbox
            )

            detections.append({

                "type": name,

                "class_id": cls,

                "bbox": bbox,

                "center": (cx, cy),

                "confidence": confidence,

                "is_emergency": False,

                "emergency_type": None

            })

        return detections

    # ==========================================
    # Emergency Detection
    # ==========================================

    def detect_emergency(self, frame):
        """
        Detect emergency vehicles.

        Returns:
            List of emergency detections.
        """

        results = self.emergency_model.predict(
            frame,
            conf=self.emergency_conf,
            iou=self.iou,
            verbose=False
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        boxes = result.boxes

        for box, cls, conf in zip(
            boxes.xyxy,
            boxes.cls,
            boxes.conf
        ):

            x1, y1, x2, y2 = map(
                int,
                box.tolist()
            )

            cls = int(cls)
            confidence = float(conf)

            name = self.EMERGENCY_CLASSES.get(
                cls,
                "Emergency"
            )

            bbox = [
                x1,
                y1,
                x2,
                y2
            ]

            cx, cy = self.get_center(
                bbox
            )

            detections.append({

                "type": name,

                "class_id": cls,

                "bbox": bbox,

                "center": (cx, cy),

                "confidence": confidence,

                "is_emergency": True,

                "emergency_type": name

            })

        return detections

    # ==========================================
    # Detect Everything
    # ==========================================

    def detect(self, frame):
        """
        Run both traffic and emergency detection.

        Returns:

            {
                "traffic": [...],
                "emergency": [...]
            }
        """

        traffic_detections = self.detect_traffic(
            frame
        )

        emergency_detections = self.detect_emergency(
            frame
        )

        return {
            "traffic": traffic_detections,
            "emergency": emergency_detections
        }