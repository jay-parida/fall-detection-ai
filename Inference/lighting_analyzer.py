import cv2


def analyze_lighting(frame):
    """
    Analyze lighting and basic image quality of a camera frame.

    Returns:
        dict containing brightness, contrast, blur score,
        and lighting category.
    """

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mean pixel intensity = brightness
    brightness = float(gray.mean())

    # Standard deviation = contrast
    contrast = float(gray.std())

    # Laplacian variance = blur/sharpness indicator
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Determine lighting category
    if brightness < 50:
        lighting = "VERY_DARK"
    elif brightness < 100:
        lighting = "DARK"
    elif brightness < 180:
        lighting = "NORMAL"
    elif brightness < 230:
        lighting = "BRIGHT"
    else:
        lighting = "VERY_BRIGHT"

    return {
        "lighting": lighting,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
    }