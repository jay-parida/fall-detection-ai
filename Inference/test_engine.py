import cv2
from fall_detection_engine import FallDetectionEngine


# ============================================================
# CREATE DETECTION ENGINE
# ============================================================

print("Starting Fall Detection Engine...")

detector = FallDetectionEngine()


# ============================================================
# START WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


print("Webcam started!")
print("Press Q to quit.")


# ============================================================
# CREATE WINDOW
# ============================================================

window_name = "Stage 6 - Fall Detection Engine"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    1280,
    720
)


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
    # SEND FRAME TO DETECTION ENGINE
    # --------------------------------------------------------

    processed_frame, status = detector.process_frame(
        frame
    )


    # --------------------------------------------------------
    # DISPLAY STATUS
    # --------------------------------------------------------

    print(
        f"Status: {status['status']} | "
        f"Confidence: {status['confidence']:.2f} | "
        f"Movement: {status['movement']:.1f}px | "
        f"Ratio: {status['ratio']:.2f} | "
        f"Recovery: "
        f"{status['recovery_frames']}/"
        f"{status['recovery_required']}"
    )


    # --------------------------------------------------------
    # SHOW FRAME
    # --------------------------------------------------------

    cv2.imshow(
        window_name,
        processed_frame
    )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Stage 6 engine test stopped.")