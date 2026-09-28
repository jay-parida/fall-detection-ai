import cv2
from lighting_analyzer import analyze_lighting


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Lighting analyzer started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # Analyze lighting
    result = analyze_lighting(frame)

    # Display information
    text1 = f"Lighting: {result['lighting']}"
    text2 = f"Brightness: {result['brightness']}"
    text3 = f"Contrast: {result['contrast']}"
    text4 = f"Blur Score: {result['blur_score']}"

    cv2.putText(
        frame,
        text1,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        text2,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        text3,
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        text4,
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Lighting Analyzer", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()