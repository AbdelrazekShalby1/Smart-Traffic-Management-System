import cv2
import sqlite3
import os
import sys
import threading
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent / "smart_traffic_ai_agent"

if str(AGENT_ROOT) not in sys.path:

    sys.path.insert(0, str(AGENT_ROOT))

from core.detector import TrafficDetector
from core.tracker import TrafficTracker
from core.vehicle_state import VehicleStateManager
from features.Speed_Estimation import SpeedEstimator
from features.Red_Light_Violation_Detection import RedLightViolationDetector
from features.Vehicle_Counting import VehicleCounter
from features.Traffic_Density import TrafficDensity
from features.License_plate_detection import LicensePlateRecognizer
from features.violation_manager import ViolationManager
from agent.tools.traffic_tools import (
    set_traffic_context,
    update_live_results,
)
from config.settings import DB_PATH
from agent_app import run_agent

# ============================================================
# AI AGENT CHAT THREAD
# ============================================================

def agent_chat_loop():
    """Run the AI Agent chat while the CV pipeline continues."""

    print("\n" + "=" * 60)
    print("AI TRAFFIC AGENT READY")
    print("Ask questions about the live traffic system.")
    print("Type 'exit' to stop the AI chat.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("[INFO] AI Agent chat stopped.")
            break

        try:
            response = run_agent(user_input)
            print("\nAgent:")
            print(response)
        except Exception as e:
            print(f"\n[AGENT ERROR] {e}")

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = (
    "Utils/"
    "4K Road traffic video for object detection and tracking - free download now.mp4"
)

# ============================================================
# SPEED CONFIGURATION
# ============================================================

SPEED_LINE_A = 200
SPEED_LINE_B = 350
DISTANCE_METERS = 15.0
SPEED_LIMIT = 60.0

# ============================================================
# COUNTING CONFIGURATION
# ============================================================

COUNTING_LINE_Y = 800

# ============================================================
# DENSITY ROI
# ============================================================

ROI_POINTS = [
    [250, 180],
    [1000, 180],
    [1100, 650],
    [180, 650]
]

# ============================================================
# TRAFFIC LIGHT
# ============================================================

LIGHT_ROI = (
    50,
    50,
    150,
    250
)

STOP_LINE_Y = 720

# ============================================================
# DATABASE
# ============================================================

EVIDENCE_FOLDER = "violations"

# ============================================================
# INITIALIZE SYSTEM
# ============================================================

print(
    "[INFO] Initializing Traffic Management System..."
)

# ============================================================
# CORE MODULES
# ============================================================

detector = TrafficDetector(
    traffic_model_path="Models/yolo11n.pt",
    emergency_model_path="Models/Emergency_Detection.pt"
)

tracker = TrafficTracker()
state_manager = VehicleStateManager()

# ============================================================
# FEATURE MODULES
# ============================================================

speed_estimator = SpeedEstimator(
    line_a=SPEED_LINE_A,
    line_b=SPEED_LINE_B,
    distance_meters=DISTANCE_METERS,
    speed_limit=SPEED_LIMIT
)

red_light_detector = RedLightViolationDetector(
    light_roi=LIGHT_ROI,
    stop_line_y=STOP_LINE_Y
)

counter = VehicleCounter(
    line_y=COUNTING_LINE_Y
)

density_detector = TrafficDensity(
    roi_points=ROI_POINTS
)

# ============================================================
# LICENSE PLATE RECOGNIZER
# ============================================================

print("[INFO] Loading license plate model...")

license_plate_recognizer = LicensePlateRecognizer(
    plate_model_path=(
        "Models/license_plate_detection.pt"
    ),

    # --------------------------------------------------------
    # Plate Detection
    # --------------------------------------------------------

    confidence=0.25,
    image_size=640,

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    max_readings=5,
    max_attempts=12,

    # Run OCR every 15 calls
    #
    # Video = 30 FPS
    #
    # Approximately 2 attempts/sec
    # --------------------------------------------------------

    process_interval=15,
    ocr_confidence=0.20,

    # CPU
    gpu=False,
    debug=True
)

print("[INFO] License plate model loaded.")

# ============================================================
# CONNECT CV SYSTEM TO AI AGENT
# ============================================================

# Connect the existing CV objects to the AI Agent.
# The Agent uses the same state_manager, detector and tracker.

set_traffic_context(
    state_manager=state_manager,
    detector=detector,
    tracker=tracker,
)

print("[INFO] AI Agent connected to live traffic context.")

# ============================================================
# VIOLATION MANAGER
# ============================================================

violation_manager = ViolationManager(
    db_path=DB_PATH,
    evidence_folder=EVIDENCE_FOLDER
)

# ============================================================
# START AI AGENT
# ============================================================

agent_thread = threading.Thread(
    target=agent_chat_loop,
    daemon=True
)

agent_thread.start()

print("[INFO] AI Agent chat thread started.")
# ============================================================
# HELPER FUNCTIONS
# ============================================================

def update_database_plate(
    vehicle_id,
    violation_type,
    plate
):

    """

    Update an existing violation if it was saved

    before OCR detected the license plate.

    """

    if not plate:
        return

    try:

        conn = sqlite3.connect(DB_PATH)
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

                OR license_plate = 'None'

            )

            """,

            (
                plate,
                vehicle_id,
                violation_type
            )

        )

        conn.commit()

        conn.close()

    except Exception as e:

        print(
            f"[DB ERROR] Could not update plate: {e}"
        )

# ============================================================
# GET EXISTING EVIDENCE PATH
# ============================================================

def get_existing_evidence_path(
    vehicle_id,
    violation_type
):

    try:

        conn = sqlite3.connect(DB_PATH)

        cursor = conn.cursor()

        cursor.execute(

            """

            SELECT evidence_path

            FROM violations

            WHERE vehicle_id = ?

            AND violation_type = ?

            ORDER BY id DESC

            LIMIT 1

            """,

            (
                vehicle_id,
                violation_type
            )

        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return row[0]

    except Exception as e:

        print(

            f"[DB ERROR] {e}"

        )

    return None

# ============================================================
# UPDATE EXISTING EVIDENCE IMAGE
# ============================================================



def update_existing_evidence(

    vehicle_id,

    violation_type,

    evidence_frame

):



    evidence_path = get_existing_evidence_path(

        vehicle_id,

        violation_type

    )



    if not evidence_path:

        return



    try:



        cv2.imwrite(

            evidence_path,

            evidence_frame

        )



        print(

            f"[EVIDENCE] Updated: {evidence_path}"

        )



    except Exception as e:



        print(

            f"[EVIDENCE ERROR] {e}"

        )





# ============================================================

# DRAW VIOLATION EVIDENCE

# ============================================================



def create_violation_evidence(

    raw_frame,

    track,

    vehicle,

    violation_type,

    plate

):

    """

    Creates a completely fresh evidence image.



    IMPORTANT:

    It always starts from raw_frame.

    Therefore no old drawings can accumulate.

    """



    evidence = raw_frame.copy()



    # --------------------------------------------------------

    # Vehicle bbox

    # --------------------------------------------------------



    x1, y1, x2, y2 = map(

        int,

        track["bbox"]

    )



    # Keep bbox inside image



    x1 = max(

        0,

        min(

            x1,

            evidence.shape[1] - 1

        )

    )



    y1 = max(

        0,

        min(

            y1,

            evidence.shape[0] - 1

        )

    )



    x2 = max(

        0,

        min(

            x2,

            evidence.shape[1] - 1

        )

    )



    y2 = max(

        0,

        min(

            y2,

            evidence.shape[0] - 1

        )

    )



    # --------------------------------------------------------

    # Red bbox

    # --------------------------------------------------------



    cv2.rectangle(

        evidence,

        (x1, y1),

        (x2, y2),

        (0, 0, 255),

        3

    )



    # --------------------------------------------------------

    # Information

    # --------------------------------------------------------



    vehicle_id = vehicle.get(

        "id",

        track["id"]

    )



    vehicle_type = vehicle.get(

        "type",

        track["type"]

    )



    speed = vehicle.get(

        "speed"

    )



    if speed is None:

        speed = 0.0



    if plate:

        plate_text = str(plate)

    else:

        plate_text = "Not Detected"



    # --------------------------------------------------------

    # Panel dimensions

    # --------------------------------------------------------



    panel_width = 330

    panel_height = 150



    panel_x = x1



    # Put panel above car

    panel_y = y1 - panel_height - 10



    # If there isn't enough space above

    # put it below the vehicle



    if panel_y < 10:



        panel_y = y2 + 10



    # Keep panel inside frame



    if panel_x + panel_width > evidence.shape[1]:



        panel_x = (

            evidence.shape[1]

            - panel_width

            - 10

        )



    if panel_y + panel_height > evidence.shape[0]:



        panel_y = (

            evidence.shape[0]

            - panel_height

            - 10

        )



    panel_x = max(

        10,

        panel_x

    )



    panel_y = max(

        10,

        panel_y

    )



    # --------------------------------------------------------

    # Black panel

    # --------------------------------------------------------



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



    # --------------------------------------------------------

    # Text

    # --------------------------------------------------------



    text_x = panel_x + 10



    text_y = panel_y + 25



    # Violation



    cv2.putText(

        evidence,



        f"VIOLATION: {violation_type}",



        (

            text_x,

            text_y

        ),



        cv2.FONT_HERSHEY_SIMPLEX,



        0.65,



        (0, 0, 255),



        2

    )



    # Vehicle ID



    cv2.putText(

        evidence,



        f"Vehicle ID: {vehicle_id}",



        (

            text_x,

            text_y + 30

        ),



        cv2.FONT_HERSHEY_SIMPLEX,



        0.55,



        (255, 255, 255),



        2

    )



    # Vehicle type



    cv2.putText(

        evidence,



        f"Type: {vehicle_type}",



        (

            text_x,

            text_y + 55

        ),



        cv2.FONT_HERSHEY_SIMPLEX,



        0.55,



        (255, 255, 255),



        2

    )



    # Speed



    cv2.putText(

        evidence,



        f"Speed: {speed:.1f} km/h",



        (

            text_x,

            text_y + 80

        ),



        cv2.FONT_HERSHEY_SIMPLEX,



        0.55,



        (0, 255, 255),



        2

    )



    # Plate



    cv2.putText(

        evidence,



        f"Plate: {plate_text}",



        (

            text_x,

            text_y + 110

        ),



        cv2.FONT_HERSHEY_SIMPLEX,



        0.55,



        (0, 255, 0),



        2

    )



    return evidence



# ============================================================

# VIDEO

# ============================================================



cap = cv2.VideoCapture(

    VIDEO_PATH

)





if not cap.isOpened():



    print(

        "[ERROR] Could not open video."

    )



    violation_manager.close()



    raise SystemExit





# ============================================================

# FPS

# ============================================================



fps = cap.get(

    cv2.CAP_PROP_FPS

)





if fps <= 0:



    fps = 30.0





print(

    f"[INFO] Video FPS: {fps}"

)



# ============================================================

# FRAME NUMBER

# ============================================================



frame_number = 0


def log_cv_update(message):
    if frame_number == 1 or frame_number % 30 == 0:
        print(message)





# ============================================================

# MAIN LOOP

# ============================================================



while True:



    ret, frame = cap.read()





    if not ret:



        print(

            "[INFO] Video finished."

        )



        break





    frame_number += 1





    # ========================================================

    # IMPORTANT

    #

    # Keep a completely clean copy.

    # Nothing should ever be drawn on raw_frame.

    # ========================================================



    raw_frame = frame.copy()



    # ========================================================

    # 1. DETECTION

    # ========================================================



    detections = detector.detect(

        raw_frame

    )

    log_cv_update("[CV] Detection updated")





    # ========================================================

    # 2. TRACKING

    # ========================================================



    tracks = tracker.update(

        detections,

        raw_frame

    )

    log_cv_update("[CV] Tracking updated")





    traffic_tracks = tracks[

        "traffic"

    ]





    emergency_tracks = tracks[

        "emergency"

    ]





    # ========================================================

    # 3. VEHICLE STATE

    # ========================================================



    state_manager.update(

        tracks

    )





    # ========================================================

    # 4. SPEED ESTIMATION

    # ========================================================



    speed_results = speed_estimator.update(

        traffic_tracks,

        frame_number,

        fps,

        state_manager

    )





    # ========================================================

    # 5. RED LIGHT

    # ========================================================



    red_light_results = red_light_detector.update(

        raw_frame,

        traffic_tracks,

        state_manager

    )





    # ========================================================

    # 6. COUNTING

    # ========================================================



    count_results = counter.update(

        traffic_tracks,

        state_manager

    )





    # ========================================================

    # 7. DENSITY

    # ========================================================



    density_results = density_detector.update(

        raw_frame,

        traffic_tracks

    )



    update_live_results(

        tracks=tracks,

        density=density_results,

        count=count_results,

        speed=speed_results,

        red_light=red_light_results,

    )
    # ========================================================

    # 8. LICENSE PLATE

    # ========================================================



    for track in traffic_tracks:



        vehicle_id = track["id"]





        # ----------------------------------------------------

        # Get state

        # ----------------------------------------------------



        vehicle = state_manager.get_vehicle(

            vehicle_id

        )





        if vehicle is None:

            continue





        # ----------------------------------------------------

        # Violations

        # ----------------------------------------------------



        violations = vehicle.get(

            "violations",

            []

        )





        if not violations:

            continue





        # ====================================================

        # LICENSE PLATE OCR

        # ====================================================



        plate = license_plate_recognizer.process_vehicle(

            raw_frame,

            track

        )





        # ----------------------------------------------------

        # Also get stored result

        # ----------------------------------------------------



        stored_plate = (

            license_plate_recognizer.get_plate(

                vehicle_id

            )

        )





        if stored_plate:



            plate = stored_plate





        # ====================================================

        # UPDATE STATE

        # ====================================================



        if plate:



            state_manager.set_license_plate(

                vehicle_id,

                plate

            )



            print(

                f"[PLATE] Vehicle {vehicle_id}: {plate}"

            )





        # ----------------------------------------------------

        # IMPORTANT:

        #

        # Get vehicle again after OCR.

        # This guarantees the plate is included.

        # ----------------------------------------------------



        vehicle = state_manager.get_vehicle(

            vehicle_id

        )





        if vehicle is None:

            continue





        # ====================================================

        # HANDLE EACH VIOLATION

        # ====================================================



        for violation_type in vehicle.get(

            "violations",

            []

        ):



            # ------------------------------------------------

            # Current plate

            # ------------------------------------------------



            current_plate = vehicle.get(

                "license_plate"

            )





            # =================================================

            # CREATE CLEAN EVIDENCE

            # =================================================



            evidence_frame = create_violation_evidence(



                raw_frame,



                track,



                vehicle,



                violation_type,



                current_plate

            )





            # =================================================

            # SAVE ONLY WHEN PLATE EXISTS

            #

            # This prevents inserting None.

            # =================================================



            if current_plate:



                violation_manager.save_violation(



                    vehicle,



                    violation_type,



                    evidence_frame

                )





                # --------------------------------------------

                # In case violation was already saved before

                # OCR detected the plate

                # --------------------------------------------



                update_database_plate(



                    vehicle_id,



                    violation_type,



                    current_plate

                )





                # --------------------------------------------

                # Replace old evidence with clean evidence

                # containing the plate.

                # --------------------------------------------



                update_existing_evidence(



                    vehicle_id,



                    violation_type,



                    evidence_frame

                )





    # ========================================================

    # 9. DISPLAY

    #

    # IMPORTANT:

    # Start from RAW frame every single time.

    # ========================================================



    display_frame = raw_frame.copy()





    # ========================================================

    # DRAW TRAFFIC VEHICLES

    # ========================================================



    for track in traffic_tracks:



        x1, y1, x2, y2 = map(

            int,

            track["bbox"]

        )





        vehicle_id = track["id"]





        vehicle_type = track["type"]





        # ----------------------------------------------------

        # Vehicle state

        # ----------------------------------------------------



        vehicle = state_manager.get_vehicle(

            vehicle_id

        )





        # ====================================================

        # BOUNDING BOX

        # ====================================================



        cv2.rectangle(



            display_frame,



            (x1, y1),



            (x2, y2),



            (0, 255, 0),



            2

        )





        # ====================================================

        # VEHICLE LABEL

        # ====================================================



        label = (

            f"{vehicle_type} "

            f"ID:{vehicle_id}"

        )





        cv2.putText(



            display_frame,



            label,



            (

                x1,

                max(

                    20,

                    y1 - 10

                )

            ),



            cv2.FONT_HERSHEY_SIMPLEX,



            0.6,



            (0, 255, 0),



            2

        )





        if vehicle is None:

            continue





        # ====================================================

        # SPEED

        # ====================================================



        speed = vehicle.get(

            "speed"

        )





        if speed is not None:



            speed_text = (

                f"Speed: {speed:.1f} km/h"

            )





            cv2.putText(



                display_frame,



                speed_text,



                (

                    x1,

                    min(

                        display_frame.shape[0] - 50,

                        y2 + 20

                    )

                ),



                cv2.FONT_HERSHEY_SIMPLEX,



                0.6,



                (255, 255, 0),



                2

            )





        # ====================================================

        # VIOLATIONS

        # ====================================================



        violations = vehicle.get(

            "violations",

            []

        )





        if violations:



            violation_text = ", ".join(

                violations

            )





            cv2.putText(



                display_frame,



                violation_text,



                (

                    x1,

                    min(

                        display_frame.shape[0] - 30,

                        y2 + 45

                    )

                ),



                cv2.FONT_HERSHEY_SIMPLEX,



                0.6,



                (0, 0, 255),



                2

            )





        # ====================================================

        # LICENSE PLATE

        # ====================================================



        plate = vehicle.get(

            "license_plate"

        )





        if plate:



            cv2.putText(



                display_frame,



                f"Plate: {plate}",



                (

                    x1,

                    min(

                        display_frame.shape[0] - 10,

                        y2 + 70

                    )

                ),



                cv2.FONT_HERSHEY_SIMPLEX,



                0.6,



                (255, 0, 0),



                2

            )





    # ========================================================

    # EMERGENCY VEHICLES

    # ========================================================



    for track in emergency_tracks:



        x1, y1, x2, y2 = map(

            int,

            track["bbox"]

        )





        vehicle_id = track["id"]





        emergency_type = track.get(

            "emergency_type",

            "Emergency"

        )





        # ----------------------------------------------------

        # Bounding box

        # ----------------------------------------------------



        cv2.rectangle(



            display_frame,



            (x1, y1),



            (x2, y2),



            (0, 0, 255),



            2

        )





        # ----------------------------------------------------

        # Label

        # ----------------------------------------------------



        label = (

            f"{emergency_type} "

            f"ID:{vehicle_id}"

        )





        cv2.putText(



            display_frame,



            label,



            (

                x1,

                max(

                    20,

                    y1 - 10

                )

            ),



            cv2.FONT_HERSHEY_SIMPLEX,



            0.6,



            (0, 0, 255),



            2

        )





    # ========================================================

    # SYSTEM INFORMATION

    # ========================================================



    density_level = density_results.get(

        "density",

        "UNKNOWN"

    )





    vehicle_count = density_results.get(

        "vehicle_count",

        0

    )





    total_count = count_results.get(

        "total",

        0

    )





    # ========================================================

    # VEHICLES

    # ========================================================



    cv2.putText(



        display_frame,



        f"Vehicles: {vehicle_count}",



        (20, 30),
        
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ========================================================
    # TOTAL COUNT
    # ========================================================

    cv2.putText(

        display_frame,

        f"Total Count: {total_count}",

        (20, 60),

        cv2.FONT_HERSHEY_SIMPLEX,



        0.7,



        (255, 255, 255),



        2

    )





    # ========================================================

    # DENSITY

    # ========================================================



    cv2.putText(



        display_frame,



        f"Density: {density_level}",



        (20, 90),



        cv2.FONT_HERSHEY_SIMPLEX,



        0.7,



        (255, 255, 255),



        2

    )





    # ========================================================

    # SHOW

    # ========================================================



    cv2.imshow(



        "Smart Traffic Management System",



        display_frame

    )

    # ========================================================

    # QUIT

    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break

# ============================================================

# CLEANUP

# ============================================================

cap.release()

cv2.destroyAllWindows()

violation_manager.close()
print(
    "[INFO] Traffic Management System stopped."
)