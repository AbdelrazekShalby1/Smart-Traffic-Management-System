class VehicleCounter:
    """
    Vehicle Counting Module.

    Responsibilities:
    - Receive tracked vehicles
    - Detect crossing of a virtual counting line
    - Count vehicles by type
    - Prevent counting the same vehicle twice

    Detection and Tracking are NOT handled here.
    """

    def __init__(
        self,
        line_y=800
    ):

        # ==========================================
        # Counting Line
        # ==========================================

        self.line_y = line_y

        # ==========================================
        # Previous Y position of each vehicle
        #
        # {
        #     vehicle_id: previous_y
        # }
        # ==========================================

        self.previous_positions = {}

        # ==========================================
        # Vehicles already counted
        # ==========================================

        self.counted_ids = set()

        # ==========================================
        # Counters
        # ==========================================

        self.vehicle_count = {
            "bicycle": 0,
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }

        # ==========================================
        # Total count
        # ==========================================

        self.total_count = 0

    # ============================================================
    # Check Line Crossing
    # ============================================================

    def _crossed_line(
        self,
        previous_y,
        current_y
    ):
        """
        Check whether the vehicle crossed
        the counting line.

        The vehicle must move from above
        the line to below it.
        """

        if previous_y is None:
            return False

        return (
            previous_y < self.line_y
            and
            current_y >= self.line_y
        )

    # ============================================================
    # Update
    # ============================================================

    def update(
        self,
        tracks,
        state_manager=None
    ):
        """
        Process current tracked vehicles.

        Parameters
        ----------
        tracks:
            Output from tracker.update()

        state_manager:
            VehicleStateManager instance

        Returns
        -------
        dict:
            Current counting results.
        """

        # ==========================================
        # Get normal traffic tracks
        # ==========================================

        traffic_tracks = tracks
        
        # ==========================================
        # Process each vehicle
        # ==========================================

        new_counts = []

        for track in traffic_tracks:

            vehicle_id = track["id"]

            vehicle_type = track["type"]

            # --------------------------------------
            # Bounding box
            # --------------------------------------

            x1, y1, x2, y2 = track["bbox"]

            # --------------------------------------
            # Center Y
            # --------------------------------------

            center_y = int(
                (y1 + y2) / 2
            )

            # --------------------------------------
            # Previous position
            # --------------------------------------

            previous_y = (
                self.previous_positions.get(
                    vehicle_id
                )
            )

            # ======================================
            # Check crossing
            # ======================================

            crossed = self._crossed_line(
                previous_y,
                center_y
            )

            # ======================================
            # Count vehicle
            # ======================================

            if (
                crossed
                and
                vehicle_id not in self.counted_ids
            ):

                self.counted_ids.add(
                    vehicle_id
                )

                # ----------------------------------
                # Make sure type exists
                # ----------------------------------

                if vehicle_type not in self.vehicle_count:

                    self.vehicle_count[
                        vehicle_type
                    ] = 0

                # ----------------------------------
                # Increment count
                # ----------------------------------

                self.vehicle_count[
                    vehicle_type
                ] += 1

                self.total_count += 1

                # ----------------------------------
                # Store event
                # ----------------------------------

                count_event = {
                    "vehicle_id": vehicle_id,
                    "vehicle_type": vehicle_type
                }

                new_counts.append(
                    count_event
                )

                # ----------------------------------
                # Update Vehicle State
                # ----------------------------------

                if state_manager is not None:

                    state_manager.vehicles[
                        vehicle_id
                    ]["counted"] = True

            # ======================================
            # Update previous position
            # ======================================

            self.previous_positions[
                vehicle_id
            ] = center_y

        # ==========================================
        # Return Results
        # ==========================================

        return {
            "counts": self.vehicle_count.copy(),

            "total": self.total_count,

            "new_counts": new_counts
        }

    # ============================================================
    # Get Count
    # ============================================================

    def get_count(
        self,
        vehicle_type=None
    ):
        """
        Get count for a specific vehicle type
        or total count.
        """

        if vehicle_type is None:
            return self.total_count

        return self.vehicle_count.get(
            vehicle_type,
            0
        )

    # ============================================================
    # Get All Counts
    # ============================================================

    def get_all_counts(self):
        """
        Return all vehicle counts.
        """

        return self.vehicle_count.copy()

    # ============================================================
    # Remove Vehicle
    # ============================================================

    def remove_vehicle(
        self,
        vehicle_id
    ):
        """
        Remove position history.

        The vehicle remains counted if it
        already crossed the line.
        """

        self.previous_positions.pop(
            vehicle_id,
            None
        )

    # ============================================================
    # Reset
    # ============================================================

    def reset(self):
        """
        Reset all counting data.
        """

        self.previous_positions.clear()

        self.counted_ids.clear()

        for vehicle_type in self.vehicle_count:

            self.vehicle_count[
                vehicle_type
            ] = 0

        self.total_count = 0