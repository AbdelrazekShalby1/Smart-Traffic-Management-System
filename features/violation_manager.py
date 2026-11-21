import os
import sqlite3
import cv2
from datetime import datetime


class ViolationManager:

    def __init__(
        self,
        db_path="DataBase/traffic_system.db",
        evidence_folder="violations"
    ):

        self.db_path = db_path
        self.evidence_folder = evidence_folder

        # Prevent duplicate violations during current run
        self.saved_violations = set()

        # Store evidence information
        self.violation_files = {}

        # Create evidence folder
        os.makedirs(
            self.evidence_folder,
            exist_ok=True
        )

        self._create_table()

        # Load already saved violations
        self._load_existing_violations()

    # =========================================================
    # DATABASE TABLE
    # =========================================================

    def _create_table(self):

        conn = sqlite3.connect(
            self.db_path
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS violations (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                vehicle_id INTEGER,

                vehicle_type TEXT,

                violation_type TEXT,

                speed REAL,

                license_plate TEXT,

                timestamp TEXT,

                evidence_path TEXT

            )
            """
        )

        conn.commit()
        conn.close()

    # =========================================================
    # LOAD EXISTING VIOLATIONS
    # =========================================================

    def _load_existing_violations(self):

        if not os.path.exists(self.db_path):
            return

        conn = sqlite3.connect(
            self.db_path
        )

        cursor = conn.cursor()

        try:

            cursor.execute(
                """
                SELECT
                    vehicle_id,
                    violation_type,
                    evidence_path
                FROM violations
                """
            )

            rows = cursor.fetchall()

            for row in rows:

                vehicle_id = row[0]
                violation_type = row[1]
                evidence_path = row[2]

                key = (
                    vehicle_id,
                    violation_type
                )

                self.saved_violations.add(
                    key
                )

                self.violation_files[
                    key
                ] = evidence_path

        except sqlite3.Error as e:

            print(
                f"[DB WARNING] {e}"
            )

        conn.close()

    # =========================================================
    # DRAW EVIDENCE
    # =========================================================

    def _create_evidence(
        self,
        frame,
        vehicle,
        violation_type
    ):

        evidence = frame.copy()

        vehicle_id = vehicle.get(
            "id"
        )

        vehicle_type = vehicle.get(
            "type",
            "Unknown"
        )

        speed = vehicle.get(
            "speed"
        )

        license_plate = vehicle.get(
            "license_plate"
        )

        bbox = vehicle.get(
            "bbox"
        )

        # =====================================================
        # VEHICLE BOX
        # =====================================================

        if bbox:

            x1, y1, x2, y2 = map(
                int,
                bbox
            )

            # Keep bbox inside frame

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                evidence.shape[1] - 1,
                x2
            )

            y2 = min(
                evidence.shape[0] - 1,
                y2
            )

            # -------------------------------------------------
            # Red bounding box
            # -------------------------------------------------

            cv2.rectangle(
                evidence,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                4
            )

            # =================================================
            # INFORMATION PANEL
            # =================================================

            panel_width = 400
            panel_height = 120

            panel_x = x1

            panel_y = max(
                10,
                y1 - panel_height - 10
            )

            # Keep panel inside image

            panel_x = min(
                panel_x,
                evidence.shape[1] - panel_width - 10
            )

            panel_x = max(
                10,
                panel_x
            )

            # -------------------------------------------------
            # Panel background
            # -------------------------------------------------

            cv2.rectangle(
                evidence,
                (
                    panel_x,
                    panel_y
                ),
                (
                    panel_x + panel_width,
                    panel_y + panel_height
                ),
                (0, 0, 0),
                -1
            )

            # -------------------------------------------------
            # Violation
            # -------------------------------------------------

            cv2.putText(
                evidence,
                f"VIOLATION: {violation_type}",
                (
                    panel_x + 10,
                    panel_y + 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

            # -------------------------------------------------
            # Vehicle ID
            # -------------------------------------------------

            cv2.putText(
                evidence,
                f"Vehicle ID: {vehicle_id}",
                (
                    panel_x + 10,
                    panel_y + 50
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            # -------------------------------------------------
            # Vehicle Type
            # -------------------------------------------------

            cv2.putText(
                evidence,
                f"Type: {vehicle_type}",
                (
                    panel_x + 10,
                    panel_y + 75
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            # -------------------------------------------------
            # Speed
            # -------------------------------------------------

            if speed is not None:

                speed_text = (
                    f"Speed: {speed:.1f} km/h"
                )

            else:

                speed_text = (
                    "Speed: N/A"
                )

            cv2.putText(
                evidence,
                speed_text,
                (
                    panel_x + 10,
                    panel_y + 100
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            # =================================================
            # LICENSE PLATE
            # =================================================

            if license_plate:

                plate_text = (
                    f"Plate: {license_plate}"
                )

            else:

                plate_text = (
                    "Plate: Not Detected"
                )

            # -------------------------------------------------
            # Plate label background
            # -------------------------------------------------

            plate_box_y1 = min(
                evidence.shape[0] - 55,
                y2 + 10
            )

            plate_box_y2 = min(
                evidence.shape[0] - 5,
                y2 + 50
            )

            plate_box_x1 = x1

            plate_box_x2 = min(
                evidence.shape[1] - 5,
                x1 + 320
            )

            cv2.rectangle(
                evidence,
                (
                    plate_box_x1,
                    plate_box_y1
                ),
                (
                    plate_box_x2,
                    plate_box_y2
                ),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                evidence,
                plate_text,
                (
                    plate_box_x1 + 10,
                    plate_box_y1 + 27
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        return evidence

    # =========================================================
    # SAVE VIOLATION
    # =========================================================
    def save_violation(self, vehicle, violation_type, frame):

        vehicle_id = vehicle.get("id")

        # منع تكرار نفس المخالفة لنفس العربية
        key = (vehicle_id, violation_type)

        if key in self.saved_violations:
            return None

        # -------------------------------------------------
        # Vehicle information
        # -------------------------------------------------

        vehicle_type = vehicle.get(
            "type",
            "Unknown"
        )

        speed = vehicle.get(
            "speed",
            None
        )

        license_plate = vehicle.get(
            "license_plate",
            None
        )

        # لو مفيش plate نخزن None عادي
        if license_plate == "":
            license_plate = None

        # -------------------------------------------------
        # Timestamp
        # -------------------------------------------------

        timestamp = datetime.now()

        timestamp_db = timestamp.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        timestamp_file = timestamp.strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        # -------------------------------------------------
        # Violation name
        # -------------------------------------------------

        safe_violation_type = (
            violation_type
            .replace(" ", "_")
            .replace("/", "_")
        )

        # -------------------------------------------------
        # Evidence path
        # -------------------------------------------------

        filename = (
            f"{vehicle_id}_"
            f"{safe_violation_type}_"
            f"{timestamp_file}.jpg"
        )

        evidence_path = os.path.join(
            self.evidence_folder,
            filename
        )

        # -------------------------------------------------
        # Save image
        # -------------------------------------------------

        success = cv2.imwrite(
            evidence_path,
            frame
        )

        if not success:

            print(
                f"[ERROR] Failed to save evidence: "
                f"{evidence_path}"
            )

            return None

        # -------------------------------------------------
        # Save to database
        # -------------------------------------------------

        conn = sqlite3.connect(
            self.db_path
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO violations
            (
                vehicle_id,
                vehicle_type,
                violation_type,
                speed,
                license_plate,
                timestamp,
                evidence_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                vehicle_id,
                vehicle_type,
                violation_type,
                speed,
                license_plate,
                timestamp_db,
                evidence_path
            )
        )

        conn.commit()
        conn.close()

        # -------------------------------------------------
        # Mark as saved
        # -------------------------------------------------

        self.saved_violations.add(
            key
        )

        # -------------------------------------------------
        # Debug
        # -------------------------------------------------

        print(
            f"[VIOLATION SAVED] "
            f"Vehicle={vehicle_id} | "
            f"Type={violation_type} | "
            f"Plate={license_plate}"
        )

        return {
            "vehicle_id": vehicle_id,
            "vehicle_type": vehicle_type,
            "violation_type": violation_type,
            "speed": speed,
            "license_plate": license_plate,
            "timestamp": timestamp_db,
            "evidence_path": evidence_path
        }

    # =========================================================
    # UPDATE PLATE AFTER OCR FINDS IT
    # =========================================================

    def _update_violation_plate(
        self,
        vehicle_id,
        violation_type,
        license_plate,
        vehicle,
        frame
    ):

        key = (
            vehicle_id,
            violation_type
        )

        evidence_path = self.violation_files.get(
            key
        )

        if not evidence_path:

            return

        # =====================================================
        # UPDATE DATABASE
        # =====================================================

        conn = sqlite3.connect(
            self.db_path
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE violations
            SET license_plate = ?
            WHERE vehicle_id = ?
            AND violation_type = ?
            AND (
                license_plate IS NULL
                OR license_plate = ''
            )
            """,
            (
                license_plate,
                vehicle_id,
                violation_type
            )
        )

        updated_rows = cursor.rowcount

        conn.commit()
        conn.close()

        # =====================================================
        # UPDATE EVIDENCE IMAGE
        # =====================================================

        if os.path.exists(
            evidence_path
        ):

            # -------------------------------------------------
            # Read original evidence
            # -------------------------------------------------

            evidence = cv2.imread(
                evidence_path
            )

            if evidence is not None:

                # ---------------------------------------------
                # IMPORTANT:
                # Use current vehicle information
                # ---------------------------------------------

                updated_vehicle = vehicle.copy()

                updated_vehicle[
                    "license_plate"
                ] = license_plate

                # ---------------------------------------------
                # Rebuild evidence
                # ---------------------------------------------

                evidence = self._create_evidence(
                    evidence,
                    updated_vehicle,
                    violation_type
                )

                cv2.imwrite(
                    evidence_path,
                    evidence
                )

        # =====================================================
        # LOG
        # =====================================================

        if updated_rows > 0:

            print(
                f"[DB UPDATE] "
                f"Vehicle {vehicle_id} | "
                f"Plate = {license_plate}"
            )

    # =========================================================
    # GET SAVED COUNT
    # =========================================================

    def get_saved_count(self):

        return len(
            self.saved_violations
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.saved_violations.clear()

        self.violation_files.clear()

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        pass