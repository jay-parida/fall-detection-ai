import cv2
from ultralytics import YOLO


# -----------------------------------
# Load model
# -----------------------------------

model_path = r"D:\AI_Projects\Fall_Detection_AI\Models\best.pt"

model = YOLO(model_path)

print("Model loaded successfully!")


# -----------------------------------
# Open webcam
# -----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started!")
print("Press Q to quit.")


# Previous bounding-box center
previous_center = None


# -----------------------------------
# Live detection
# -----------------------------------

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read frame.")
        break


    # -----------------------------------
    # YOLO detection
    # -----------------------------------

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.50,
        device="cpu",
        verbose=False
    )


    # Get detections
    boxes = results[0].boxes


    if len(boxes) > 0:

        # -----------------------------------
        # Get first bounding box
        # -----------------------------------

        box = boxes[0]


        # Bounding-box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()


        # -----------------------------------
        # Calculate center
        # -----------------------------------

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        current_center = (center_x, center_y)


        # -----------------------------------
        # Calculate movement
        # -----------------------------------

        if previous_center is not None:

            movement = (
                (current_center[0] - previous_center[0]) ** 2
                +
                (current_center[1] - previous_center[1]) ** 2
            ) ** 0.5

        else:

            movement = 0


        previous_center = current_center


        # -----------------------------------
        # Calculate bounding-box dimensions
        # -----------------------------------

        box_width = x2 - x1
        box_height = y2 - y1


        # -----------------------------------
        # Calculate aspect ratio
        # -----------------------------------

        if box_height > 0:
            aspect_ratio = box_width / box_height
        else:
            aspect_ratio = 0


        # -----------------------------------
        # Confidence
        # -----------------------------------

        confidence = float(box.conf[0])


        # -----------------------------------
        # Print information
        # -----------------------------------

        print(
            f"Confidence: {confidence:.2f} | "
            f"Center: {current_center} | "
            f"Movement: {movement:.1f}px | "
            f"Width: {box_width:.0f}px | "
            f"Height: {box_height:.0f}px | "
            f"Ratio: {aspect_ratio:.2f}"
        )


    else:

        previous_center = None

        print("No detection")


    # -----------------------------------
    # Draw YOLO result
    # -----------------------------------

    annotated_frame = results[0].plot()


    # -----------------------------------
    # Show camera
    # -----------------------------------

    cv2.imshow(
        "Fall Detection - Motion Test",
        annotated_frame
    )


    # -----------------------------------
    # Quit
    # -----------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------------
# Cleanup
# -----------------------------------

cap.release()
cv2.destroyAllWindows()

print("Webcam stopped.")