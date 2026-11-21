import cv2
import numpy as np
from collections import deque


class TrafficDensity:

    def __init__(
        self,
        roi_points,
        low_threshold=8,
        medium_threshold=18,
        high_threshold=30,
        smoothing_window=15
    ):
        """
        Traffic Density Calculator

        Parameters:
            roi_points:
                Polygon points defining the road area.

            low_threshold:
                Maximum occupancy percentage for LOW density.

            medium_threshold:
                Maximum occupancy percentage for MEDIUM density.

            high_threshold:
                Maximum occupancy percentage for HIGH density.

            smoothing_window:
                Number of previous frames used for smoothing.
        """

        self.roi_points = np.array(
            roi_points,
            dtype=np.int32
        ).reshape((-1, 1, 2))

        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold

        self.density_history = deque(
            maxlen=smoothing_window
        )

    # =========================================================
    # Create ROI Mask
    # =========================================================

    def _create_roi_mask(self, frame):

        roi_mask = np.zeros(
            frame.shape[:2],
            dtype=np.uint8
        )

        cv2.fillPoly(
            roi_mask,
            [self.roi_points],
            255
        )

        return roi_mask

    # =========================================================
    # Check Vehicle Inside ROI
    # =========================================================

    def _is_inside_roi(self, center):

        center_x, center_y = center

        return cv2.pointPolygonTest(
            self.roi_points,
            (center_x, center_y),
            False
        ) >= 0

    # =========================================================
    # Calculate Density
    # =========================================================

    def update(self, frame, tracks):

        """
        Parameters:
            frame:
                Current video frame.

            tracks:
                Tracks returned from tracker.py

        Returns:
            {
                "vehicle_count": ...,
                "occupancy": ...,
                "density": ...,
                "vehicles_inside": [...]
            }
        """

        # -----------------------------------------------------
        # ROI Mask
        # -----------------------------------------------------

        roi_mask = self._create_roi_mask(frame)

        # -----------------------------------------------------
        # Occupancy Mask
        # -----------------------------------------------------

        occupancy_mask = np.zeros(
            frame.shape[:2],
            dtype=np.uint8
        )

        vehicles_inside = []

        # -----------------------------------------------------
        # Process Tracks
        # -----------------------------------------------------

        for track in tracks:

            vehicle_id = track["id"]

            x1, y1, x2, y2 = map(
                int,
                track["bbox"]
            )

            center = track["center"]

            # Check if vehicle center is inside ROI
            if not self._is_inside_roi(center):
                continue

            vehicles_inside.append(vehicle_id)

            # -------------------------------------------------
            # Vehicle Mask
            # -------------------------------------------------

            vehicle_mask = np.zeros(
                frame.shape[:2],
                dtype=np.uint8
            )

            cv2.rectangle(
                vehicle_mask,
                (x1, y1),
                (x2, y2),
                255,
                -1
            )

            # Keep only vehicle area inside ROI
            vehicle_mask = cv2.bitwise_and(
                vehicle_mask,
                roi_mask
            )

            # Add vehicle to occupancy mask
            occupancy_mask = cv2.bitwise_or(
                occupancy_mask,
                vehicle_mask
            )

        # -----------------------------------------------------
        # Vehicle Count
        # -----------------------------------------------------

        vehicle_count = len(vehicles_inside)

        # -----------------------------------------------------
        # ROI Area
        # -----------------------------------------------------

        roi_area = cv2.countNonZero(
            roi_mask
        )

        # -----------------------------------------------------
        # Occupied Area
        # -----------------------------------------------------

        occupied_area = cv2.countNonZero(
            occupancy_mask
        )

        # -----------------------------------------------------
        # Occupancy Percentage
        # -----------------------------------------------------

        if roi_area > 0:

            occupancy_percentage = (
                occupied_area / roi_area
            ) * 100

        else:

            occupancy_percentage = 0.0

        # -----------------------------------------------------
        # Temporal Smoothing
        # -----------------------------------------------------

        self.density_history.append(
            occupancy_percentage
        )

        smoothed_occupancy = np.mean(
            self.density_history
        )

        # -----------------------------------------------------
        # Density Classification
        # -----------------------------------------------------

        if smoothed_occupancy < self.low_threshold:

            density = "LOW"

        elif smoothed_occupancy < self.medium_threshold:

            density = "MEDIUM"

        elif smoothed_occupancy < self.high_threshold:

            density = "HIGH"

        else:

            density = "SEVERE"

        # -----------------------------------------------------
        # Return Results
        # -----------------------------------------------------

        return {
            "vehicle_count": vehicle_count,
            "occupancy": float(smoothed_occupancy),
            "density": density,
            "vehicles_inside": vehicles_inside
        }

    # =========================================================
    # Reset
    # =========================================================

    def reset(self):

        self.density_history.clear()