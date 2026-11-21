from collections import deque


class SpeedEstimator:
    """
    Speed estimation module.

    Responsibilities:
    - Detect when a vehicle crosses Speed Line A
    - Detect when the same vehicle crosses Speed Line B
    - Calculate vehicle speed
    - Smooth speed using median filtering
    - Detect speeding
    - Update vehicle_state.py

    Detection and Tracking are NOT handled here.
    """

    def __init__(
        self,
        line_a=200,
        line_b=350,
        distance_meters=15.0,
        speed_limit=60.0,
        min_time_seconds=0.15,
        max_reasonable_speed=180.0,
        smoothing_window=5
    ):

        # ==========================================
        # Speed Measurement Settings
        # ==========================================

        self.line_a = line_a
        self.line_b = line_b

        # IMPORTANT:
        # This must be the REAL physical distance
        # between Line A and Line B.
        self.distance_meters = distance_meters

        # Speed limit
        self.speed_limit = speed_limit

        # Reject unrealistically small times
        self.min_time_seconds = min_time_seconds

        # Reject unrealistic speed values
        self.max_reasonable_speed = max_reasonable_speed

        # Number of previous speed values
        # used for median smoothing.
        self.smoothing_window = smoothing_window

        # ==========================================
        # Vehicle State
        # ==========================================

        # Previous Y position for every vehicle
        #
        # {
        #     vehicle_id: previous_y
        # }
        self.previous_y = {}

        # Frame in which vehicle crossed Line A
        #
        # {
        #     vehicle_id: frame_number
        # }
        self.line_a_frames = {}

        # Speed history for smoothing
        #
        # {
        #     vehicle_id: deque(...)
        # }
        self.speed_history = {}

        # Final calculated speeds
        #
        # {
        #     vehicle_id: speed
        # }
        self.vehicle_speeds = {}

        # Vehicles that exceeded speed limit
        self.speeding_vehicles = set()

    # ============================================================
    # Calculate Raw Speed
    # ============================================================

    def _calculate_speed(
        self,
        frame_a,
        frame_b,
        fps
    ):
        """
        Calculate speed between Line A and Line B.
        """

        # ------------------------------------------
        # Validate FPS
        # ------------------------------------------

        if fps is None or fps <= 0:
            return None

        # ------------------------------------------
        # Number of frames
        # ------------------------------------------

        elapsed_frames = frame_b - frame_a

        if elapsed_frames <= 0:
            return None

        # ------------------------------------------
        # Convert frames to seconds
        # ------------------------------------------

        elapsed_time = (
            elapsed_frames / fps
        )

        # ------------------------------------------
        # Reject very small time
        # ------------------------------------------

        if elapsed_time < self.min_time_seconds:
            return None

        # ------------------------------------------
        # Speed m/s
        # ------------------------------------------

        speed_mps = (
            self.distance_meters /
            elapsed_time
        )

        # ------------------------------------------
        # Convert to km/h
        # ------------------------------------------

        speed_kmh = speed_mps * 3.6

        # ------------------------------------------
        # Validate
        # ------------------------------------------

        if speed_kmh <= 0:
            return None

        if speed_kmh > self.max_reasonable_speed:
            return None

        return speed_kmh

    # ============================================================
    # Smooth Speed
    # ============================================================

    def _smooth_speed(
        self,
        vehicle_id,
        speed
    ):
        """
        Smooth speed using median filtering.
        """

        if speed is None:
            return None

        # Create history for new vehicle
        if vehicle_id not in self.speed_history:

            self.speed_history[vehicle_id] = deque(
                maxlen=self.smoothing_window
            )

        # Add new speed
        self.speed_history[
            vehicle_id
        ].append(speed)

        # Get sorted values
        values = sorted(
            self.speed_history[
                vehicle_id
            ]
        )

        # ------------------------------------------
        # Median
        # ------------------------------------------

        n = len(values)
        middle = n // 2

        if n % 2 == 1:

            median_speed = values[middle]

        else:

            median_speed = (
                values[middle - 1] +
                values[middle]
            ) / 2

        return median_speed

    # ============================================================
    # Update
    # ============================================================

    def update(
        self,
        tracks,
        frame_number,
        fps,
        state_manager=None
    ):
        """
        Process current frame tracks.

        Parameters
        ----------
        tracks:
            Output from tracker.update()

        frame_number:
            Current video frame number

        fps:
            Video FPS

        state_manager:
            VehicleStateManager instance

        Returns
        -------
        dict:
            {
                vehicle_id: {
                    "speed": ...,
                    "speeding": ...
                }
            }
        """

        results = {}

        # ========================================================
        # Traffic tracks
        # ========================================================

        traffic_tracks = tracks
        # ========================================================
        # Emergency tracks
        # ========================================================

        emergency_tracks = []
        # ========================================================
        # Combine tracks
        # ========================================================

        all_tracks = (
            traffic_tracks +
            emergency_tracks
        )

        # ========================================================
        # Process every vehicle
        # ========================================================

        for track in all_tracks:

            vehicle_id = track["id"]

            # --------------------------------------
            # Bounding box
            # --------------------------------------

            x1, y1, x2, y2 = track["bbox"]

            # Use bottom-center of vehicle
            # instead of bounding box center.
            #
            # This is generally more suitable
            # for road-line crossing.
            current_y = int(y2)

            # --------------------------------------
            # Previous position
            # --------------------------------------

            previous_y = self.previous_y.get(
                vehicle_id
            )

            # First time seeing vehicle
            if previous_y is None:

                self.previous_y[
                    vehicle_id
                ] = current_y

                continue

            # ====================================================
            # LINE A
            # ====================================================

            crossed_line_a = (
                previous_y < self.line_a
                and
                current_y >= self.line_a
            )

            if crossed_line_a:

                # Only save the first crossing
                if vehicle_id not in self.line_a_frames:

                    self.line_a_frames[
                        vehicle_id
                    ] = frame_number

            # ====================================================
            # LINE B
            # ====================================================

            crossed_line_b = (
                previous_y < self.line_b
                and
                current_y >= self.line_b
                and
                vehicle_id in self.line_a_frames
                and
                vehicle_id not in self.vehicle_speeds
            )

            if crossed_line_b:

                # ----------------------------------------------
                # Frame A
                # ----------------------------------------------

                frame_a = self.line_a_frames[
                    vehicle_id
                ]

                # ----------------------------------------------
                # Frame B
                # ----------------------------------------------

                frame_b = frame_number

                # ----------------------------------------------
                # Calculate raw speed
                # ----------------------------------------------

                raw_speed = self._calculate_speed(
                    frame_a=frame_a,
                    frame_b=frame_b,
                    fps=fps
                )

                if raw_speed is not None:

                    # ------------------------------------------
                    # Smooth speed
                    # ------------------------------------------

                    speed = self._smooth_speed(
                        vehicle_id,
                        raw_speed
                    )

                    if speed is not None:

                        # Save speed
                        self.vehicle_speeds[
                            vehicle_id
                        ] = speed

                        # --------------------------------------
                        # Speeding
                        # --------------------------------------

                        is_speeding = (
                            speed > self.speed_limit
                        )

                        if is_speeding:

                            self.speeding_vehicles.add(
                                vehicle_id
                            )

                        # --------------------------------------
                        # Result
                        # --------------------------------------

                        results[vehicle_id] = {
                            "speed": speed,
                            "raw_speed": raw_speed,
                            "speeding": is_speeding,
                            "speed_limit": self.speed_limit
                        }

                        # --------------------------------------
                        # Update Vehicle State
                        # --------------------------------------

                        if state_manager is not None:

                            state_manager.set_speed(
                                vehicle_id,
                                speed
                            )

                            state_manager.set_speeding(
                                vehicle_id,
                                is_speeding
                            )

                            if is_speeding:

                                state_manager.add_violation(
                                    vehicle_id,
                                    "SPEEDING"
                                )

            # ====================================================
            # Update Previous Position
            # ====================================================

            self.previous_y[
                vehicle_id
            ] = current_y

        return results

    # ============================================================
    # Get Vehicle Speed
    # ============================================================

    def get_speed(
        self,
        vehicle_id
    ):
        """
        Get calculated speed for a vehicle.
        """

        return self.vehicle_speeds.get(
            vehicle_id
        )

    # ============================================================
    # Is Speeding
    # ============================================================

    def is_speeding(
        self,
        vehicle_id
    ):
        """
        Check whether a vehicle is speeding.
        """

        return vehicle_id in (
            self.speeding_vehicles
        )

    # ============================================================
    # Remove Vehicle
    # ============================================================

    def remove_vehicle(
        self,
        vehicle_id
    ):
        """
        Remove vehicle-related speed data.
        """

        self.previous_y.pop(
            vehicle_id,
            None
        )

        self.line_a_frames.pop(
            vehicle_id,
            None
        )

        self.speed_history.pop(
            vehicle_id,
            None
        )

        self.vehicle_speeds.pop(
            vehicle_id,
            None
        )

        self.speeding_vehicles.discard(
            vehicle_id
        )

    # ============================================================
    # Clear
    # ============================================================

    def clear(self):
        """
        Clear all speed data.
        """

        self.previous_y.clear()

        self.line_a_frames.clear()

        self.speed_history.clear()

        self.vehicle_speeds.clear()

        self.speeding_vehicles.clear()