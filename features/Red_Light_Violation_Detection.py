import cv2
import numpy as np
from collections import deque


class RedLightViolationDetector:
    """
    Red Light Violation Detection Module.

    Responsibilities:
    - Detect traffic light state from a predefined ROI
    - Apply optional temporal smoothing
    - Detect vehicle crossing the stop line
    - Detect red-light violations
    - Update vehicle_state.py

    Detection and Tracking are NOT handled here.
    """

    def __init__(
        self,
        light_roi,
        stop_line_y=720,
        use_stable_light=True,
        history_size=7,
        min_red_frames=4,
        min_brightness=50,
        min_score_difference=5
    ):

        # ==========================================
        # Traffic Light ROI
        # ==========================================

        self.light_x1 = light_roi[0]
        self.light_y1 = light_roi[1]
        self.light_x2 = light_roi[2]
        self.light_y2 = light_roi[3]

        # ==========================================
        # Stop Line
        # ==========================================

        self.stop_line_y = stop_line_y

        # ==========================================
        # Traffic Light Smoothing
        # ==========================================

        self.use_stable_light = use_stable_light

        self.light_history = deque(
            maxlen=history_size
        )

        self.min_red_frames = min_red_frames

        # ==========================================
        # Detection Thresholds
        # ==========================================

        self.min_brightness = min_brightness

        self.min_score_difference = (
            min_score_difference
        )

        # ==========================================
        # Vehicle Position History
        #
        # {
        #     vehicle_id: previous_bottom_y
        # }
        # ==========================================

        self.previous_positions = {}

        # ==========================================
        # Vehicles Already Marked
        # ==========================================

        self.violators = set()

    # ============================================================
    # Detect Traffic Light
    # ============================================================

    def detect_traffic_light(self, frame):
        """
        Detect RED / YELLOW / GREEN / UNKNOWN
        from the configured traffic-light ROI.

        Returns:
            {
                "state": "RED",
                "red_score": ...,
                "yellow_score": ...,
                "green_score": ...
            }
        """

        # ==========================================
        # Crop ROI
        # ==========================================

        roi = frame[
            self.light_y1:self.light_y2,
            self.light_x1:self.light_x2
        ]

        if roi is None or roi.size == 0:

            return {
                "state": "UNKNOWN",
                "red_score": 0.0,
                "yellow_score": 0.0,
                "green_score": 0.0
            }

        # ==========================================
        # Resize
        # ==========================================

        roi = cv2.resize(
            roi,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        # ==========================================
        # Remove Background
        # ==========================================

        h, w = roi.shape[:2]

        x1 = int(w * 0.20)
        x2 = int(w * 0.80)

        center = roi[:, x1:x2]

        # ==========================================
        # Split into 3 Sections
        # ==========================================

        h = center.shape[0]

        third = h // 3

        red_region = center[
            0:third,
            :
        ]

        yellow_region = center[
            third:2 * third,
            :
        ]

        green_region = center[
            2 * third:h,
            :
        ]

        # ==========================================
        # Brightness Score
        # ==========================================

        def get_score(region):

            if region.size == 0:
                return 0.0

            gray = cv2.cvtColor(
                region,
                cv2.COLOR_BGR2GRAY
            )

            # Reduce noise
            gray = cv2.GaussianBlur(
                gray,
                (5, 5),
                0
            )

            pixels = gray.reshape(-1)

            # Sort pixels
            pixels = np.sort(pixels)

            # Brightest 5%
            n = max(
                1,
                int(len(pixels) * 0.05)
            )

            score = np.mean(
                pixels[-n:]
            )

            return float(score)

        # ==========================================
        # Calculate Scores
        # ==========================================

        red_score = get_score(
            red_region
        )

        yellow_score = get_score(
            yellow_region
        )

        green_score = get_score(
            green_region
        )

        scores = {
            "RED": red_score,
            "YELLOW": yellow_score,
            "GREEN": green_score
        }

        # ==========================================
        # Strongest State
        # ==========================================

        state = max(
            scores,
            key=scores.get
        )

        strongest_score = scores[state]

        # ==========================================
        # Minimum Brightness
        # ==========================================

        if strongest_score < self.min_brightness:

            return {
                "state": "UNKNOWN",
                "red_score": red_score,
                "yellow_score": yellow_score,
                "green_score": green_score
            }

        # ==========================================
        # Difference from Second Strongest
        # ==========================================

        sorted_scores = sorted(
            scores.values(),
            reverse=True
        )

        highest = sorted_scores[0]

        second_highest = sorted_scores[1]

        difference = (
            highest -
            second_highest
        )

        # ==========================================
        # Ambiguous Detection
        # ==========================================

        if difference < self.min_score_difference:

            return {
                "state": "UNKNOWN",
                "red_score": red_score,
                "yellow_score": yellow_score,
                "green_score": green_score
            }

        # ==========================================
        # Return
        # ==========================================

        return {
            "state": state,
            "red_score": red_score,
            "yellow_score": yellow_score,
            "green_score": green_score
        }

    # ============================================================
    # Stable Traffic Light State
    # ============================================================

    def _get_stable_light_state(
        self,
        current_state
    ):
        """
        Apply temporal smoothing to traffic light state.
        """

        self.light_history.append(
            current_state
        )

        # ==========================================
        # Not enough history
        # ==========================================

        if len(self.light_history) < 4:

            return "UNKNOWN"

        # ==========================================
        # Count states
        # ==========================================

        red_count = self.light_history.count(
            "RED"
        )

        yellow_count = self.light_history.count(
            "YELLOW"
        )

        green_count = self.light_history.count(
            "GREEN"
        )

        # ==========================================
        # Stable State
        # ==========================================

        if red_count >= self.min_red_frames:

            return "RED"

        elif green_count >= self.min_red_frames:

            return "GREEN"

        elif yellow_count >= self.min_red_frames:

            return "YELLOW"

        return "UNKNOWN"

    # ============================================================
    # Update
    # ============================================================

    def update(
        self,
        frame,
        tracks,
        state_manager=None
    ):
        """
        Process current frame.

        Parameters:
            frame:
                Current video frame.

            tracks:
                Output from tracker.update().

            state_manager:
                VehicleStateManager instance.

        Returns:
            {
                "traffic_light": ...,
                "red_score": ...,
                "yellow_score": ...,
                "green_score": ...,
                "violations": [...]
            }
        """

        # ==========================================
        # 1. Detect Traffic Light
        # ==========================================

        light_result = self.detect_traffic_light(
            frame
        )

        current_light = light_result[
            "state"
        ]

        # ==========================================
        # 2. Stable State
        # ==========================================

        if self.use_stable_light:

            traffic_light = (
                self._get_stable_light_state(
                    current_light
                )
            )

        else:

            traffic_light = current_light

        # ==========================================
        # 3. Get Tracks
        # ==========================================

        traffic_tracks = tracks

        # ==========================================
        # 4. We don't need emergency vehicles
        # for red-light detection
        # ==========================================

        current_violations = []

        # ==========================================
        # 5. Process Vehicles
        # ==========================================

        for track in traffic_tracks:

            vehicle_id = track["id"]

            # --------------------------------------
            # Bounding Box
            # --------------------------------------

            x1, y1, x2, y2 = track["bbox"]

            # --------------------------------------
            # Bottom center
            # --------------------------------------

            bottom_y = int(y2)

            # --------------------------------------
            # Previous position
            # --------------------------------------

            previous_y = self.previous_positions.get(
                vehicle_id,
                bottom_y
            )

            # ======================================
            # Stop Line Crossing
            # ======================================

            crossed_line = (
                previous_y > self.stop_line_y
                and
                bottom_y <= self.stop_line_y
            )

            # ======================================
            # Red Light Violation
            # ======================================

            if (
                traffic_light == "RED"
                and
                crossed_line
                and
                vehicle_id not in self.violators
            ):

                # ----------------------------------
                # Add violation
                # ----------------------------------

                self.violators.add(
                    vehicle_id
                )

                violation = {
                    "vehicle_id": vehicle_id,
                    "vehicle_type": track["type"],
                    "traffic_light": traffic_light,
                    "violation_type": "RED_LIGHT"
                }

                current_violations.append(
                    violation
                )

                # ----------------------------------
                # Update Vehicle State
                # ----------------------------------

                if state_manager is not None:

                    state_manager.set_red_light_violation(
                        vehicle_id,
                        True
                    )

                    state_manager.add_violation(
                        vehicle_id,
                        "RED_LIGHT"
                    )

            # ======================================
            # Update Previous Position
            # ======================================

            self.previous_positions[
                vehicle_id
            ] = bottom_y

        # ==========================================
        # Return Results
        # ==========================================

        return {
            "traffic_light": traffic_light,

            "raw_light_state": current_light,

            "red_score": light_result[
                "red_score"
            ],

            "yellow_score": light_result[
                "yellow_score"
            ],

            "green_score": light_result[
                "green_score"
            ],

            "violations": current_violations,

            "total_violations": len(
                self.violators
            )
        }

    # ============================================================
    # Get Violation Status
    # ============================================================

    def is_violator(
        self,
        vehicle_id
    ):
        """
        Check whether a vehicle has
        already committed a red-light violation.
        """

        return vehicle_id in self.violators

    # ============================================================
    # Remove Vehicle
    # ============================================================

    def remove_vehicle(
        self,
        vehicle_id
    ):
        """
        Remove tracking history for a vehicle.

        The violation itself remains stored.
        """

        self.previous_positions.pop(
            vehicle_id,
            None
        )

    # ============================================================
    # Clear
    # ============================================================

    def clear(self):
        """
        Clear runtime state.
        """

        self.previous_positions.clear()

        self.violators.clear()

        self.light_history.clear()