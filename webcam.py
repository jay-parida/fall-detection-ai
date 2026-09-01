import cv2
from ultralytics import YOLO


# -----------------------------------
# 1. Load trained YOLO model
# -----------------------------------

model_path = r"D:\AI_Projects\Fall_Detection_AI\Models\best.pt"

model = YOLO(model_path)

print("Model loaded successfully!")


# -----------------------------------
# 2. Open laptop webcam
# -----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started!")
print("Press Q to quit.")


# -----------------------------------
# 3. Process webcam frames
# -----------------------------------

while True:

    # Read one frame
    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    # Run YOLO detection
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.50,
        device="cpu",
        verbose=False
    )

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow(
        "Fall Detection - Live Webcam",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------------
# 4. Clean up
# -----------------------------------

cap.release()
cv2.destroyAllWindows()

print("Webcam stopped.")