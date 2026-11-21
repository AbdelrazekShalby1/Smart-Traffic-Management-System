class VehicleStateManager:
    """
    Manage the state of every tracked vehicle.

    Responsibilities:
    - Create a state for each new vehicle
    - Update vehicle position and information
    - Keep track history
    - Store speed information
    - Store violation states
    - Store license plate information
    - Store emergency information
    - Remove vehicles that disappeared for too long

    Detection and tracking are NOT handled here.
    """

    def __init__(self, max_missing_frames=30):

        # ==========================================
        # All active vehicles
        #
        # {
        #     vehicle_id: {
        #         ...
        #     }
        # }
        # ==========================================

        self.vehicles = {}

        # Maximum number of frames a vehicle can
        # disappear before being removed.
        self.max_missing_frames = max_missing_frames

    # ============================================================
    # Create New Vehicle
    # ============================================================

    def _create_vehicle(self, track):

        vehicle_id = track["id"]

        vehicle = {

            # --------------------------------------
            # Identity
            # --------------------------------------

            "id": vehicle_id,

            "type": track["type"],

            "class_id": track["class_id"],

            # --------------------------------------
            # Tracking
            # --------------------------------------

            "bbox": track["bbox"],

            "center": track["center"],

            "confidence": track["confidence"],

            "frame_number": track.get(
                "frame_number",
                0
            ),

            # --------------------------------------
            # Position history
            # --------------------------------------

            "positions": [
                track["center"]
            ],

            # --------------------------------------
            # Speed
            # --------------------------------------

            "speed": None,

            "speeding": False,

            # --------------------------------------
            # Red Light
            # --------------------------------------

            "red_light_violation": False,

            # --------------------------------------
            # License Plate
            # --------------------------------------

            "license_plate": None,

            "plate_confidence": None,

            # --------------------------------------
            # Emergency
            # --------------------------------------

            "is_emergency": track.get(
                "is_emergency",
                False
            ),

            "emergency_type": track.get(
                "emergency_type",
                None
            ),

            # --------------------------------------
            # Violations
            # --------------------------------------

            "violations": [],

            # --------------------------------------
            # Counting
            # --------------------------------------

            "counted": False,

            # --------------------------------------
            # Missing frames
            # --------------------------------------

            "missing_frames": 0,

            # --------------------------------------
            # Active state
            # --------------------------------------

            "active": True
        }

        return vehicle

    # ============================================================
    # Update Existing Vehicle
    # ============================================================

    def _update_vehicle(
        self,
        vehicle,
        track
    ):

        # ==========================================
        # Basic tracking information
        # ==========================================

        vehicle["bbox"] = track["bbox"]

        vehicle["center"] = track["center"]

        vehicle["confidence"] = track["confidence"]

        vehicle["frame_number"] = track.get(
            "frame_number",
            vehicle["frame_number"]
        )

        # ==========================================
        # Position history
        # ==========================================

        vehicle["positions"].append(
            track["center"]
        )

        # Keep history from growing forever.
        # 100 positions are enough for the modules
        # such as speed estimation.
        if len(vehicle["positions"]) > 100:

            vehicle["positions"] = (
                vehicle["positions"][-100:]
            )

        # ==========================================
        # Emergency information
        # ==========================================

        if track.get("is_emergency", False):

            vehicle["is_emergency"] = True

            vehicle["emergency_type"] = (
                track.get("emergency_type")
            )

        # ==========================================
        # Vehicle is visible again
        # ==========================================

        vehicle["missing_frames"] = 0

        vehicle["active"] = True

    # ============================================================
    # Update All Vehicles
    # ============================================================

    def update(
        self,
        tracks
    ):
        """
        Update vehicle states using tracker output.

        Parameters:
            tracks:
                Output from tracker.update()

        Returns:
            Dictionary containing all active vehicles.
        """

        # ========================================================
        # Collect all tracks
        # ========================================================

        traffic_tracks = tracks.get(
            "traffic",
            []
        )

        emergency_tracks = tracks.get(
            "emergency",
            []
        )

        all_tracks = (
            traffic_tracks +
            emergency_tracks
        )

        # IDs visible in the current frame
        current_ids = set()

        # ========================================================
        # Process current tracks
        # ========================================================

        for track in all_tracks:

            vehicle_id = track["id"]

            current_ids.add(vehicle_id)

            # --------------------------------------
            # New vehicle
            # --------------------------------------

            if vehicle_id not in self.vehicles:

                self.vehicles[vehicle_id] = (
                    self._create_vehicle(track)
                )

            # --------------------------------------
            # Existing vehicle
            # --------------------------------------

            else:

                self._update_vehicle(
                    self.vehicles[vehicle_id],
                    track
                )

        # ========================================================
        # Update missing vehicles
        # ========================================================

        for vehicle_id, vehicle in list(
            self.vehicles.items()
        ):

            if vehicle_id not in current_ids:

                vehicle["missing_frames"] += 1

                # ----------------------------------
                # Vehicle disappeared
                # ----------------------------------

                if (
                    vehicle["missing_frames"]
                    > self.max_missing_frames
                ):

                    vehicle["active"] = False

                    del self.vehicles[
                        vehicle_id
                    ]

        # ========================================================
        # Return active vehicles
        # ========================================================

        return self.vehicles

    # ============================================================
    # Set Speed
    # ============================================================

    def set_speed(
        self,
        vehicle_id,
        speed
    ):
        """
        Update vehicle speed.
        """

        if vehicle_id not in self.vehicles:
            return

        self.vehicles[
            vehicle_id
        ]["speed"] = speed

    # ============================================================
    # Set Speeding State
    # ============================================================

    def set_speeding(
        self,
        vehicle_id,
        speeding
    ):
        """
        Update speeding status.
        """

        if vehicle_id not in self.vehicles:
            return

        self.vehicles[
            vehicle_id
        ]["speeding"] = bool(speeding)

    # ============================================================
    # Set Red Light Violation
    # ============================================================

    def set_red_light_violation(
        self,
        vehicle_id,
        violated=True
    ):
        """
        Update red-light violation status.
        """

        if vehicle_id not in self.vehicles:
            return

        self.vehicles[
            vehicle_id
        ]["red_light_violation"] = bool(
            violated
        )

    # ============================================================
    # Set License Plate
    # ============================================================

    def set_license_plate(
        self,
        vehicle_id,
        plate,
        confidence=None
    ):
        """
        Update license plate information.
        """

        if vehicle_id not in self.vehicles:
            return

        self.vehicles[
            vehicle_id
        ]["license_plate"] = plate

        self.vehicles[
            vehicle_id
        ]["plate_confidence"] = confidence

    # ============================================================
    # Add Violation
    # ============================================================

    def add_violation(
        self,
        vehicle_id,
        violation_type
    ):
        """
        Add a violation to a vehicle.

        Duplicate violations are prevented.
        """

        if vehicle_id not in self.vehicles:
            return

        violations = self.vehicles[
            vehicle_id
        ]["violations"]

        if violation_type not in violations:

            violations.append(
                violation_type
            )

    # ============================================================
    # Get Vehicle
    # ============================================================

    def get_vehicle(
        self,
        vehicle_id
    ):
        """
        Get a single vehicle state.
        """

        return self.vehicles.get(
            vehicle_id
        )

    # ============================================================
    # Get All Active Vehicles
    # ============================================================

    def get_active_vehicles(self):
        """
        Return all active vehicles.
        """

        return {
            vehicle_id: vehicle
            for vehicle_id, vehicle
            in self.vehicles.items()
            if vehicle["active"]
        }

    # ============================================================
    # Get Active Vehicle Count
    # ============================================================

    def get_active_count(self):
        """
        Return number of active vehicles.
        """

        return len(
            self.get_active_vehicles()
        )

    # ============================================================
    # Clear
    # ============================================================

    def clear(self):
        """
        Clear all vehicle states.
        """

        self.vehicles.clear()