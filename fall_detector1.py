import cv2
import time
import math
from collections import deque
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\AI_Projects\Fall_Detection_AI\Models\best.pt"

# YOLO confidence
CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# FALL DETECTION SETTINGS
# ============================================================

# Minimum total movement for downward movement
MOVEMENT_THRESHOLD = 20

# Minimum vertical downward movement
DOWNWARD_MOVEMENT_THRESHOLD = 15

# Body shape:
# width / height
ASPECT_RATIO_THRESHOLD = 2.0

# Strongly horizontal body
STRONG_HORIZONTAL_RATIO = 2.35

# Very horizontal body
VERY_HORIZONTAL_RATIO = 2.60


# ============================================================
# HISTORY / CONFIRMATION
# ============================================================

HISTORY_SIZE = 10

# Number of candidate frames required
CONFIRMATION_FRAMES = 2

# Maximum time a candidate can remain active
CANDIDATE_TIMEOUT = 3.0


# ============================================================
# RECOVERY SETTINGS
# ============================================================

# When fall is detected, we wait for the person to become
# upright again before returning to MONITORING.

# Upright body ratio should generally be below this value
RECOVERY_RATIO_THRESHOLD = 1.70

# Number of consecutive upright frames required
RECOVERY_FRAMES_REQUIRED = 5

# Maximum vertical movement considered stable during recovery
RECOVERY_MOVEMENT_THRESHOLD = 25

# If the person is moving upward strongly, it can help
# confirm that they are getting back up.
RECOVERY_UPWARD_THRESHOLD = 8


# ============================================================
# MISSED DETECTION SETTINGS
# ============================================================

# We don't immediately reset tracking when detection disappears.
MAX_MISSED_FRAMES = 10


# ============================================================
# DISPLAY SETTINGS
# ============================================================

WINDOW_NAME = "Fall Detection"

# Large webcam display
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


# Try to use a larger webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Webcam started!")
print("Press Q to quit.")


# ============================================================
# CREATE LARGE WINDOW
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    WINDOW_WIDTH,
    WINDOW_HEIGHT
)


# ============================================================
# STATE VARIABLES
# ============================================================

previous_center = None
previous_ratio = None

movement_history = deque(
    maxlen=HISTORY_SIZE
)

missed_frames = 0


# ============================================================
# FALL EVENT STATE
# ============================================================

event_active = False

event_start_time = 0


# ============================================================
# FALL CANDIDATE STATE
# ============================================================

fall_candidate = False

candidate_frames = 0

candidate_start_time = 0

candidate_start_ratio = 0


# ============================================================
# DOWNWARD MOVEMENT MEMORY
# ============================================================

downward_motion_detected = False


# ============================================================
# RECOVERY STATE
# ============================================================

recovery_frames = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read frame.")

        break


    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )


    # ========================================================
    # FIND BEST DETECTION
    # ========================================================

    best_box = None

    best_confidence = 0.0


    for result in results:

        if result.boxes is None:

            continue


        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )


            if confidence > best_confidence:

                best_confidence = confidence

                best_box = box


    # ========================================================
    # NO DETECTION
    # ========================================================

    if best_box is None:

        missed_frames += 1


        print(
            f"No detection | "
            f"Missed frames: {missed_frames}"
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We DO NOT immediately clear FALL DETECTED.
        #
        # This is useful when the person is lying on a bed
        # and YOLO temporarily loses the body.
        # ----------------------------------------------------

        if missed_frames >= MAX_MISSED_FRAMES:

            previous_center = None

            previous_ratio = None

            movement_history.clear()

            downward_motion_detected = False

            fall_candidate = False

            candidate_frames = 0

            recovery_frames = 0


        # ----------------------------------------------------
        # Display status
        # ----------------------------------------------------

        if event_active:

            status_text = "FALL DETECTED"

            status_color = (
                0,
                0,
                255
            )

        else:

            status_text = "MONITORING"

            status_color = (
                0,
                255,
                0
            )


        cv2.putText(
            frame,
            status_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            status_color,
            3
        )


        cv2.putText(
            frame,
            f"Missed: {missed_frames}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            WINDOW_NAME,
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


        continue


    # ========================================================
    # DETECTION EXISTS
    # ========================================================

    missed_frames = 0


    # ========================================================
    # GET BOUNDING BOX
    # ========================================================

    x1, y1, x2, y2 = map(
        int,
        best_box.xyxy[0]
    )


    width = max(
        1,
        x2 - x1
    )

    height = max(
        1,
        y2 - y1
    )


    # ========================================================
    # CENTER POINT
    # ========================================================

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2


    current_center = (
        center_x,
        center_y
    )


    # ========================================================
    # BODY ASPECT RATIO
    # ========================================================

    ratio = width / height


    # ========================================================
    # MOVEMENT
    # ========================================================

    movement = 0.0

    vertical_movement = 0.0

    direction = "STABLE"


    if previous_center is not None:

        dx = (
            center_x
            -
            previous_center[0]
        )

        dy = (
            center_y
            -
            previous_center[1]
        )


        movement = math.sqrt(
            dx * dx
            +
            dy * dy
        )


        vertical_movement = dy


        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        if abs(dy) > 5:

            if dy > 0:

                direction = "DOWN"

            else:

                direction = "UP"

        else:

            direction = "STABLE"


    # ========================================================
    # RATIO CHANGE
    # ========================================================

    ratio_change = 0.0


    if previous_ratio is not None:

        ratio_change = (
            ratio
            -
            previous_ratio
        )


    # ========================================================
    # STORE MOVEMENT
    # ========================================================

    movement_history.append(
        movement
    )


    # ========================================================
    # STRONG DOWNWARD MOVEMENT
    # ========================================================

    strong_downward_motion = (

        direction == "DOWN"

        and

        movement >= DOWNWARD_MOVEMENT_THRESHOLD

        and

        vertical_movement >= 8

    )


    # --------------------------------------------------------
    # Remember strong downward movement
    # --------------------------------------------------------

    if strong_downward_motion:

        downward_motion_detected = True


    # ========================================================
    # BODY SHAPE
    # ========================================================

    horizontal_body = (

        ratio >= ASPECT_RATIO_THRESHOLD

    )


    strong_horizontal_body = (

        ratio >= STRONG_HORIZONTAL_RATIO

    )


    very_horizontal_body = (

        ratio >= VERY_HORIZONTAL_RATIO

    )


    # ========================================================
    # FALL CANDIDATE
    # ========================================================

    if not event_active:

        candidate_condition = (

            (
                strong_downward_motion
                or
                downward_motion_detected
            )

            and

            horizontal_body

        )


        if candidate_condition:

            # ------------------------------------------------
            # Start candidate
            # ------------------------------------------------

            if not fall_candidate:

                fall_candidate = True

                candidate_start_time = time.time()

                candidate_frames = 1

                candidate_start_ratio = ratio


                print()
                print("--------------------------------------")
                print("FALL CANDIDATE STARTED")
                print(
                    f"Start Ratio: "
                    f"{candidate_start_ratio:.2f}"
                )
                print("--------------------------------------")


            else:

                candidate_frames += 1


    # ========================================================
    # FALL CANDIDATE PROCESSING
    # ========================================================

    candidate_age = 0.0

    score = 0


    if fall_candidate:

        candidate_age = (
            time.time()
            -
            candidate_start_time
        )


        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        if strong_downward_motion:

            score += 2


        if horizontal_body:

            score += 1


        if strong_horizontal_body:

            score += 1


        if very_horizontal_body:

            score += 1


        # Large ratio increase
        if ratio_change >= 0.20:

            score += 1


        # ----------------------------------------------------
        # Confirmation
        # ----------------------------------------------------

        confirmed = False


        if candidate_frames >= CONFIRMATION_FRAMES:

            confirmed = True


        if score >= 5:

            confirmed = True


        if ratio >= VERY_HORIZONTAL_RATIO:

            confirmed = True


        # ----------------------------------------------------
        # FALL CONFIRMED
        # ----------------------------------------------------

        if confirmed:

            event_active = True

            event_start_time = time.time()


            fall_candidate = False

            candidate_frames = 0

            downward_motion_detected = False

            recovery_frames = 0


            print()
            print("======================================")
            print("          FALL DETECTED!")
            print("======================================")
            print()


        # ----------------------------------------------------
        # Candidate expired
        # ----------------------------------------------------

        elif candidate_age > CANDIDATE_TIMEOUT:

            print()
            print("--------------------------------------")
            print("Fall candidate expired")
            print("--------------------------------------")
            print()


            fall_candidate = False

            candidate_frames = 0

            downward_motion_detected = False


    # ========================================================
    # RECOVERY DETECTION
    # ========================================================

    if event_active:

        # ----------------------------------------------------
        # Person appears upright
        # ----------------------------------------------------

        upright_body = (

            ratio < RECOVERY_RATIO_THRESHOLD

        )


        # ----------------------------------------------------
        # Person moving upward
        # ----------------------------------------------------

        getting_up = (

            direction == "UP"

            and

            abs(vertical_movement)
            >= RECOVERY_UPWARD_THRESHOLD

        )


        # ----------------------------------------------------
        # Recovery condition
        #
        # Either:
        #
        # 1. Body becomes upright
        #
        # OR
        #
        # 2. Person is moving upward strongly
        # ----------------------------------------------------

        recovery_condition = (

            upright_body

            or

            getting_up

        )


        if recovery_condition:

            recovery_frames += 1

        else:

            recovery_frames = 0


        # ----------------------------------------------------
        # Recovery confirmed
        # ----------------------------------------------------

        if recovery_frames >= RECOVERY_FRAMES_REQUIRED:

            event_active = False

            recovery_frames = 0

            downward_motion_detected = False


            print()
            print("======================================")
            print("       PERSON RECOVERED")
            print("       RETURNING TO MONITORING")
            print("======================================")
            print()


    # ========================================================
    # EVENT STATUS
    # ========================================================

    event_timer = 0


    if event_active:

        event_timer = int(
            time.time()
            -
            event_start_time
        )


    # ========================================================
    # HORIZONTAL FRAME
    # ========================================================

    horizontal_frames = 0


    if horizontal_body:

        horizontal_frames = 1


    # ========================================================
    # DRAW BOX
    # ========================================================

    if event_active:

        box_color = (
            0,
            0,
            255
        )

        label = "FALL DETECTED"

    else:

        box_color = (
            0,
            255,
            0
        )

        label = "Monitoring"


    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        box_color,
        3
    )


    # ========================================================
    # BOUNDING BOX LABEL
    # ========================================================

    cv2.putText(
        frame,
        label,
        (
            x1,
            max(
                30,
                y1 - 10
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        box_color,
        2
    )


    # ========================================================
    # MAIN STATUS
    # ========================================================

    if event_active:

        status_text = "FALL DETECTED"

        status_color = (
            0,
            0,
            255
        )

    else:

        status_text = "MONITORING"

        status_color = (
            0,
            255,
            0
        )


    cv2.putText(
        frame,
        status_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        status_color,
        3
    )


    # ========================================================
    # RECOVERY STATUS
    # ========================================================

    if event_active:

        recovery_text = (
            f"Recovery: "
            f"{recovery_frames}/"
            f"{RECOVERY_FRAMES_REQUIRED}"
        )


        cv2.putText(
            frame,
            recovery_text,
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


    # ========================================================
    # DEBUG INFORMATION ON SCREEN
    # ========================================================

    debug_y = 125

    debug_lines = [

        f"Confidence: {best_confidence:.2f}",

        f"Movement: {movement:.1f}px",

        f"Vertical: {vertical_movement:.1f}px",

        f"Direction: {direction}",

        f"Ratio: {ratio:.2f}",

        f"Candidate: {fall_candidate}",

        f"Score: {score}",

        f"Event: {event_active}"

    ]


    for line in debug_lines:

        cv2.putText(
            frame,
            line,
            (30, debug_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        debug_y += 25


    # ========================================================
    # TERMINAL DEBUG
    # ========================================================

    if fall_candidate:

        print(
            f"Candidate | "
            f"Age: {candidate_age:.2f}s | "
            f"Frames: {candidate_frames} | "
            f"Score: {score} | "
            f"StartRatio: {candidate_start_ratio:.2f}"
        )


    print(
        f"Confidence: {best_confidence:.2f} | "
        f"Movement: {movement:.1f}px | "
        f"Vertical: {vertical_movement:.1f}px | "
        f"Direction: {direction} | "
        f"Ratio: {ratio:.2f} | "
        f"RatioChange: {ratio_change:.2f} | "
        f"HorizontalFrames: {horizontal_frames} | "
        f"Candidate: {fall_candidate} | "
        f"Score: {score} | "
        f"EventActive: {event_active} | "
        f"Recovery: {recovery_frames}/{RECOVERY_FRAMES_REQUIRED} | "
        f"EventTimer: {event_timer}"
    )


    # ========================================================
    # UPDATE PREVIOUS VALUES
    # ========================================================

    previous_center = current_center

    previous_ratio = ratio


    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
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

print("Webcam stopped.")