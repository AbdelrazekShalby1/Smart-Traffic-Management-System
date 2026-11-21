import torch     # هي مكتبة للتعامل مع الـ Tensors والعمليات الحسابية الخاصة بالـ Deep Learning.

from types import SimpleNamespace
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


class TrafficTracker:

    def __init__(          # ByteTracker Parameters
        self,
        track_high_thresh=0.5,         # هل الـ Detection قوية كفاية للمرحلة الأولى؟
        track_low_thresh=0.1,          # هل أقدر أستخدم Detection ضعيفة لاستعادة Track؟
        new_track_thresh=0.6,          # هل أدي العربية دي ID جديدة؟
        track_buffer=30,               # العربية اختفت، أستنى كام Frame قبل حذفها؟
        match_thresh=0.8,              # هل العربية اللي ظهرت هي نفس العربية القديمة؟
        frame_rate=30                  # الفيديو فيه كام Frame في الثانية؟
    ):

        self.frame_rate = frame_rate

        # ----------------------------------------------------
        # Tracker configuration
        # ----------------------------------------------------

        self.args = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=True                     # يعني: "وأنا بربط العربية الحالية بالعربية السابقة، خُد ثقة YOLO في الاعتبار مع مدى تطابق الـ Bounding Boxes."
        )

        # ----------------------------------------------------
        # Separate trackers
        # ----------------------------------------------------

        self.traffic_tracker = self._create_tracker()

        self.emergency_tracker = self._create_tracker()

        # ----------------------------------------------------
        # Frame counter
        # ----------------------------------------------------

        self.frame_number = 0

    # ========================================================
    # CREATE TRACKER
    # ========================================================

    def _create_tracker(self):

        return BYTETracker(
            args=self.args
        )

    # ========================================================
    # CONVERT DETECTIONS
    # ========================================================

    def _create_results(self, detections, frame):
        """
        Convert our detector detections into an Ultralytics
        Boxes object compatible with the installed BYTETracker.
        """

        height, width = frame.shape[:2]

        # --------------------------------------------------------
        # No detections
        # --------------------------------------------------------

        if not detections:

            data = torch.empty(
                (0, 6),
                dtype=torch.float32
            )

            return Boxes(
                data,
                orig_shape=(height, width)
            )

        # --------------------------------------------------------
        # Build:
        # [x1, y1, x2, y2, confidence, class_id]
        # --------------------------------------------------------

        data = []

        for detection in detections:

            x1, y1, x2, y2 = detection["bbox"]

            confidence = detection["confidence"]

            class_id = detection["class_id"]

            data.append([
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(confidence),
                float(class_id)
            ])

        data = torch.tensor(
            data,
            dtype=torch.float32
        )

        # --------------------------------------------------------
        # Return Boxes directly
        # --------------------------------------------------------

        return Boxes(
            data,
            orig_shape=(height, width)
        )
    # ========================================================
    # UPDATE TRACKER
    # ========================================================

    def _update_tracker(
        self,
        tracker,
        detections,
        frame
    ):

        dets = self._create_results(
            detections,
            frame
        )

        tracks = tracker.update(
            dets,
            frame
        )

        return tracks

    # ========================================================
    # CONVERT TRACKS
    # ========================================================

    def _convert_tracks(self, tracks, detections, is_emergency=False):
        converted_tracks = []

        if tracks is None or len(tracks) == 0:
            return converted_tracks

        for row in tracks:

            # BYTETracker output:
            # [x1, y1, x2, y2, track_id, score, class_id, detection_index]

            x1 = float(row[0])
            y1 = float(row[1])
            x2 = float(row[2])
            y2 = float(row[3])

            track_id = int(row[4])
            confidence = float(row[5])
            class_id = int(row[6])

            # Index of original detection
            detection_index = int(row[7]) if len(row) > 7 else -1

            # Get information from original detection
            if 0 <= detection_index < len(detections):

                original_detection = detections[detection_index]

                object_type = original_detection["type"]

                # Keep original confidence if available
                confidence = original_detection.get(
                    "confidence",
                    confidence
                )

                is_emergency_object = original_detection.get(
                    "is_emergency",
                    is_emergency
                )

                emergency_type = original_detection.get(
                    "emergency_type",
                    None
                )

            else:

                # Fallback
                if is_emergency:
                    object_type = "emergency"
                else:
                    object_type = "unknown"

                is_emergency_object = is_emergency
                emergency_type = None

            # Center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            track = {
                "id": track_id,

                "type": object_type,

                "class_id": class_id,

                "bbox": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ],

                "center": (cx, cy),

                "confidence": confidence,

                "is_emergency": is_emergency_object,

                "emergency_type": emergency_type
            }

            converted_tracks.append(track)

        return converted_tracks

    # ========================================================
    # MAIN UPDATE
    # ========================================================

    def update(
        self,
        detection_results,
        frame
    ):

        self.frame_number += 1

        traffic_detections = detection_results.get(
            "traffic",
            []
        )

        emergency_detections = detection_results.get(
            "emergency",
            []
        )

        # ----------------------------------------------------
        # Traffic tracking
        # ----------------------------------------------------

        traffic_tracks_raw = self._update_tracker(
            self.traffic_tracker,
            traffic_detections,
            frame
        )

        traffic_tracks = self._convert_tracks(
            traffic_tracks_raw,
            traffic_detections,
            is_emergency=False
        )

        # ----------------------------------------------------
        # Emergency tracking
        # ----------------------------------------------------

        emergency_tracks_raw = self._update_tracker(
            self.emergency_tracker,
            emergency_detections,
            frame
        )

        emergency_tracks = self._convert_tracks(
            emergency_tracks_raw,
            emergency_detections,
            is_emergency=True
        )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {

            "traffic": traffic_tracks,

            "emergency": emergency_tracks
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.traffic_tracker = self._create_tracker()

        self.emergency_tracker = self._create_tracker()

        self.frame_number = 0