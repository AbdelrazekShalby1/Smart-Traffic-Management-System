import cv2
import re
from collections import Counter

from ultralytics import YOLO
import easyocr


class LicensePlateRecognizer:

    def __init__(
        self,
        plate_model_path="Models/license_plate_detection.pt",
        confidence=0.25,
        image_size=640,
        max_readings=5,
        max_attempts=12,
        process_interval=15,
        ocr_confidence=0.20,
        gpu=False,
        debug=False
    ):
        """
        License Plate Detection + OCR

        The module receives already tracked vehicles.

        Performance strategy:
        - Plate detection is NOT performed every frame.
        - OCR is NOT performed every frame.
        - Processing happens every `process_interval` calls.
        - Multiple OCR readings are combined using voting.
        """

        # =====================================================
        # Settings
        # =====================================================

        self.confidence = confidence
        self.image_size = image_size

        self.max_readings = max_readings
        self.max_attempts = max_attempts

        # Process one vehicle every N calls
        self.process_interval = process_interval

        self.ocr_confidence = ocr_confidence

        self.debug = debug

        # =====================================================
        # Plate Detection Model
        # =====================================================

        print(
            "[INFO] Loading license plate detection model..."
        )

        self.plate_model = YOLO(
            plate_model_path
        )

        print(
            "[INFO] License plate model loaded."
        )

        # =====================================================
        # EasyOCR
        # =====================================================

        print(
            "[INFO] Loading EasyOCR..."
        )

        self.reader = easyocr.Reader(
            ["en", "ar"],
            gpu=gpu
        )

        print(
            "[INFO] EasyOCR loaded."
        )

        # =====================================================
        # Storage
        # =====================================================

        # Valid OCR readings
        #
        # {
        #     vehicle_id: [
        #         "ABC123",
        #         "ABC123",
        #         ...
        #     ]
        # }
        self.plate_readings = {}

        # Final plate result
        #
        # {
        #     vehicle_id: "ABC123"
        # }
        self.plate_results = {}

        # Number of times we attempted processing
        self.attempt_counts = {}

        # Number of calls received for each vehicle
        self.call_counts = {}

        # Last confidence obtained
        self.best_confidence = {}

        # Vehicles that already have a reliable result
        self.finished_vehicles = set()

    # =========================================================
    # Clean OCR Text
    # =========================================================

    def _clean_plate_text(self, text):

        if not text:
            return ""

        text = text.upper()

        # Remove spaces and symbols.
        # Keep:
        # A-Z
        # Arabic characters
        # English digits
        # Arabic digits
        text = re.sub(
            r"[^A-Z0-9\u0600-\u06FF\u0660-\u0669]",
            "",
            text
        )

        return text

    # =========================================================
    # Normalize Arabic Digits
    # =========================================================

    def _normalize_digits(self, text):

        if not text:
            return ""

        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        english_digits = "0123456789"

        translation_table = str.maketrans(
            arabic_digits,
            english_digits
        )

        return text.translate(
            translation_table
        )

    # =========================================================
    # Validate Plate
    # =========================================================

    def _is_valid_plate(self, text):

        if not text:
            return False

        # Very short OCR results are usually noise.
        #
        # Examples:
        # "7"   -> reject
        # "0"   -> reject
        # "٢"   -> reject
        if len(text) < 4:
            return False

        # Must contain at least one digit
        has_digit = any(
            char.isdigit()
            or "\u0660" <= char <= "\u0669"
            for char in text
        )

        if not has_digit:
            return False

        return True

    # =========================================================
    # Detect Plate
    # =========================================================

    def _detect_plate(self, vehicle_crop):

        results = self.plate_model(
            vehicle_crop,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False
        )

        plates = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                px1, py1, px2, py2 = map(
                    int,
                    box.xyxy[0]
                )

                # -------------------------------------------------
                # Keep bbox inside vehicle crop
                # -------------------------------------------------

                px1 = max(
                    0,
                    px1
                )

                py1 = max(
                    0,
                    py1
                )

                px2 = min(
                    vehicle_crop.shape[1],
                    px2
                )

                py2 = min(
                    vehicle_crop.shape[0],
                    py2
                )

                if px2 <= px1 or py2 <= py1:
                    continue

                # -------------------------------------------------
                # Add small padding
                # -------------------------------------------------

                width = px2 - px1
                height = py2 - py1

                pad_x = int(
                    width * 0.12
                )

                pad_y = int(
                    height * 0.18
                )

                px1 = max(
                    0,
                    px1 - pad_x
                )

                py1 = max(
                    0,
                    py1 - pad_y
                )

                px2 = min(
                    vehicle_crop.shape[1],
                    px2 + pad_x
                )

                py2 = min(
                    vehicle_crop.shape[0],
                    py2 + pad_y
                )

                # -------------------------------------------------
                # Crop
                # -------------------------------------------------

                plate_crop = vehicle_crop[
                    py1:py2,
                    px1:px2
                ]

                if plate_crop.size == 0:
                    continue

                # -------------------------------------------------
                # Plate area
                # -------------------------------------------------

                area = (
                    px2 - px1
                ) * (
                    py2 - py1
                )

                plates.append(
                    {
                        "bbox": [
                            px1,
                            py1,
                            px2,
                            py2
                        ],
                        "crop": plate_crop,
                        "area": area
                    }
                )

        if self.debug:

            print(
                f"[PLATE DEBUG] "
                f"Detected: {len(plates)}"
            )

        return plates

    # =========================================================
    # Prepare OCR Images
    # =========================================================

    def _prepare_ocr_images(self, plate_crop):

        # -----------------------------------------------------
        # Resize
        # -----------------------------------------------------

        plate = cv2.resize(
            plate_crop,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        # -----------------------------------------------------
        # Grayscale
        # -----------------------------------------------------

        gray = cv2.cvtColor(
            plate,
            cv2.COLOR_BGR2GRAY
        )

        # -----------------------------------------------------
        # CLAHE
        # -----------------------------------------------------

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(
            gray
        )

        # -----------------------------------------------------
        # OTSU
        # -----------------------------------------------------

        binary = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        )[1]

        return [
            plate,
            enhanced,
            binary
        ]

    # =========================================================
    # OCR
    # =========================================================

    def _read_plate(self, plate_crop):

        if (
            plate_crop is None
            or plate_crop.size == 0
        ):
            return "", 0.0

        images = self._prepare_ocr_images(
            plate_crop
        )

        candidates = []

        # =====================================================
        # OCR each preprocessing version
        # =====================================================

        for img in images:

            try:

                results = self.reader.readtext(
                    img,
                    detail=1,
                    paragraph=False,
                    text_threshold=0.25,
                    low_text=0.15,
                    link_threshold=0.25
                )

            except Exception as e:

                print(
                    f"[OCR ERROR] {e}"
                )

                continue

            for detection in results:

                if len(detection) < 3:
                    continue

                text = detection[1]

                confidence = float(
                    detection[2]
                )

                if self.debug:

                    print(
                        f"[OCR RAW] "
                        f"Text='{text}' "
                        f"Confidence={confidence:.3f}"
                    )

                # -------------------------------------------------
                # Confidence filter
                # -------------------------------------------------

                if confidence < self.ocr_confidence:
                    continue

                # -------------------------------------------------
                # Clean
                # -------------------------------------------------

                cleaned = self._clean_plate_text(
                    text
                )

                # -------------------------------------------------
                # Normalize Arabic digits
                # -------------------------------------------------

                cleaned = self._normalize_digits(
                    cleaned
                )

                # -------------------------------------------------
                # Validate
                # -------------------------------------------------

                if not self._is_valid_plate(
                    cleaned
                ):
                    continue

                candidates.append(
                    (
                        cleaned,
                        confidence
                    )
                )

        # =====================================================
        # No valid candidate
        # =====================================================

        if not candidates:

            if self.debug:

                print(
                    "[OCR DEBUG] "
                    "No valid candidate"
                )

            return "", 0.0

        # =====================================================
        # Voting
        # =====================================================

        counter = Counter(
            text
            for text, confidence in candidates
        )

        # Most common result
        best_text, votes = counter.most_common(
            1
        )[0]

        # Get highest confidence for this text
        best_confidence = max(
            confidence
            for text, confidence in candidates
            if text == best_text
        )

        if self.debug:

            print(
                f"[OCR DEBUG] "
                f"Best='{best_text}' "
                f"Confidence={best_confidence:.3f} "
                f"Votes={votes}"
            )

        return (
            best_text,
            best_confidence
        )

    # =========================================================
    # Process Vehicle
    # =========================================================

    def process_vehicle(
        self,
        frame,
        track
    ):

        vehicle_id = track["id"]

        # =====================================================
        # If we already have a reliable plate
        # =====================================================

        if vehicle_id in self.finished_vehicles:

            return self.plate_results.get(
                vehicle_id
            )

        # =====================================================
        # Count calls
        #
        # main.py calls this function every frame.
        #
        # We don't want to run YOLO + OCR every frame.
        # =====================================================

        self.call_counts[
            vehicle_id
        ] = self.call_counts.get(
            vehicle_id,
            0
        ) + 1

        call_number = self.call_counts[
            vehicle_id
        ]

        # -----------------------------------------------------
        # Process only every N calls
        # -----------------------------------------------------

        if (
            call_number % self.process_interval
            != 0
        ):

            return self.plate_results.get(
                vehicle_id
            )

        # =====================================================
        # Check maximum attempts
        # =====================================================

        attempts = self.attempt_counts.get(
            vehicle_id,
            0
        )

        if attempts >= self.max_attempts:

            return self.plate_results.get(
                vehicle_id
            )

        self.attempt_counts[
            vehicle_id
        ] = attempts + 1

        # =====================================================
        # Vehicle Bounding Box
        # =====================================================

        x1, y1, x2, y2 = map(
            int,
            track["bbox"]
        )

        # -----------------------------------------------------
        # Keep inside frame
        # -----------------------------------------------------

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            frame.shape[1],
            x2
        )

        y2 = min(
            frame.shape[0],
            y2
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):

            return self.plate_results.get(
                vehicle_id
            )

        # =====================================================
        # Vehicle Crop
        # =====================================================

        vehicle_crop = frame[
            y1:y2,
            x1:x2
        ]

        if vehicle_crop.size == 0:

            return self.plate_results.get(
                vehicle_id
            )

        # =====================================================
        # Detect License Plate
        # =====================================================

        plates = self._detect_plate(
            vehicle_crop
        )

        if not plates:

            return self.plate_results.get(
                vehicle_id
            )

        # =====================================================
        # Select largest plate
        # =====================================================

        plates.sort(
            key=lambda p: p["area"],
            reverse=True
        )

        plate_crop = plates[0]["crop"]

        # =====================================================
        # OCR
        # =====================================================

        text, confidence = self._read_plate(
            plate_crop
        )

        if not text:

            return self.plate_results.get(
                vehicle_id
            )

        # =====================================================
        # Initialize readings
        # =====================================================

        if vehicle_id not in self.plate_readings:

            self.plate_readings[
                vehicle_id
            ] = []

        # =====================================================
        # Store reading
        # =====================================================

        self.plate_readings[
            vehicle_id
        ].append(
            text
        )

        # Keep only last N readings
        self.plate_readings[
            vehicle_id
        ] = self.plate_readings[
            vehicle_id
        ][-self.max_readings:]

        # =====================================================
        # Update best confidence
        # =====================================================

        previous_confidence = self.best_confidence.get(
            vehicle_id,
            0.0
        )

        if confidence > previous_confidence:

            self.best_confidence[
                vehicle_id
            ] = confidence

        # =====================================================
        # Voting
        # =====================================================

        readings = self.plate_readings[
            vehicle_id
        ]

        counter = Counter(
            readings
        )

        best_plate, votes = counter.most_common(
            1
        )[0]

        self.plate_results[
            vehicle_id
        ] = best_plate

        # =====================================================
        # Debug
        # =====================================================

        if self.debug:

            print(
                f"[PLATE] "
                f"Vehicle={vehicle_id} "
                f"Result={best_plate} "
                f"Votes={votes} "
                f"Confidence={confidence:.3f}"
            )

        # =====================================================
        # Finish if result is reliable
        #
        # Example:
        #
        # ABC123
        # ABC123
        # ABC123
        #
        # 3 votes -> reliable
        # =====================================================

        if (
            votes >= 3
            and confidence >= 0.30
        ):

            self.finished_vehicles.add(
                vehicle_id
            )

            if self.debug:

                print(
                    f"[PLATE FINAL] "
                    f"Vehicle {vehicle_id}: "
                    f"{best_plate}"
                )

        return best_plate

    # =========================================================
    # Get Plate
    # =========================================================

    def get_plate(
        self,
        vehicle_id
    ):

        return self.plate_results.get(
            vehicle_id
        )

    # =========================================================
    # Get All Plates
    # =========================================================

    def get_all_plates(self):

        return self.plate_results.copy()

    # =========================================================
    # Remove Vehicle
    # =========================================================

    def remove_vehicle(
        self,
        vehicle_id
    ):

        self.plate_readings.pop(
            vehicle_id,
            None
        )

        self.plate_results.pop(
            vehicle_id,
            None
        )

        self.attempt_counts.pop(
            vehicle_id,
            None
        )

        self.call_counts.pop(
            vehicle_id,
            None
        )

        self.best_confidence.pop(
            vehicle_id,
            None
        )

        self.finished_vehicles.discard(
            vehicle_id
        )

    # =========================================================
    # Reset
    # =========================================================

    def reset(self):

        self.plate_readings.clear()

        self.plate_results.clear()

        self.attempt_counts.clear()

        self.call_counts.clear()

        self.best_confidence.clear()

        self.finished_vehicles.clear()