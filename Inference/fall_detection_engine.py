import cv2
import time
import math
from collections import deque
from ultralytics import YOLO


class FallDetectionEngine:

    # ============================================================
    # CONFIGURATION
    # ============================================================

    MODEL_PATH = r"D:\AI_Projects\Fall_Detection_AI\Models\best.pt"

    # YOLO confidence
    CONFIDENCE_THRESHOLD = 0.50

    # ============================================================
    # FALL DETECTION SETTINGS
    # ============================================================

    # Minimum total movement to consider meaningful movement
    MOVEMENT_THRESHOLD = 20

    # Minimum downward movement
    DOWNWARD_MOVEMENT_THRESHOLD = 10

    # Minimum downward component
    DOWNWARD_VERTICAL_THRESHOLD = 8

    # Body becomes somewhat horizontal
    ASPECT_RATIO_THRESHOLD = 1.70

    # Strong horizontal posture
    STRONG_HORIZONTAL_RATIO = 2.10

    # Very horizontal posture
    VERY_HORIZONTAL_RATIO = 2.40

    # ============================================================
    # RATIO CHANGE
    # ============================================================

    # Significant change in body shape
    RATIO_CHANGE_THRESHOLD = 0.15

    # Large body shape change
    STRONG_RATIO_CHANGE = 0.30

    # ============================================================
    # FALL CANDIDATE
    # ============================================================

    CANDIDATE_TIMEOUT = 2.5

    # Number of frames needed for confirmation
    CONFIRMATION_FRAMES = 3

    # ============================================================
    # FALL SCORE
    # ============================================================

    # Score required to confirm fall
    FALL_SCORE_THRESHOLD = 4

    # ============================================================
    # RECOVERY
    # ============================================================

    # Ratio considered reasonably upright
    RECOVERY_RATIO_THRESHOLD = 1.65

    # Strong evidence of standing up
    RECOVERY_UPWARD_THRESHOLD = 8

    # Number of consecutive recovery frames
    RECOVERY_FRAMES_REQUIRED = 8

    # Movement required during recovery
    RECOVERY_MOVEMENT_THRESHOLD = 10

    # ============================================================
    # MISSED DETECTIONS
    # ============================================================

    MAX_MISSED_FRAMES = 10

    # ============================================================
    # DETECTION JUMP PROTECTION
    # ============================================================

    # Very large bbox jumps can be caused by YOLO switching boxes
    MAX_REASONABLE_MOVEMENT = 250

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self):

        print("Loading YOLO model...")

        self.model = YOLO(self.MODEL_PATH)

        print("Model loaded successfully!")

        # --------------------------------------------------------
        # Previous detection
        # --------------------------------------------------------

        self.previous_center = None

        self.previous_ratio = None

        # --------------------------------------------------------
        # Movement history
        # --------------------------------------------------------

        self.movement_history = deque(
            maxlen=10
        )

        self.ratio_history = deque(
            maxlen=10
        )

        # --------------------------------------------------------
        # Missed detections
        # --------------------------------------------------------

        self.missed_frames = 0

        # --------------------------------------------------------
        # Fall event
        # --------------------------------------------------------

        self.event_active = False

        self.event_start_time = 0

        # --------------------------------------------------------
        # Fall candidate
        # --------------------------------------------------------

        self.fall_candidate = False

        self.candidate_frames = 0

        self.candidate_start_time = 0

        self.candidate_start_ratio = 0.0

        # --------------------------------------------------------
        # Candidate evidence
        # --------------------------------------------------------

        self.downward_motion_detected = False

        self.max_candidate_ratio = 0.0

        self.max_candidate_ratio_change = 0.0

        self.candidate_score = 0

        # --------------------------------------------------------
        # Recovery
        # --------------------------------------------------------

        self.recovery_frames = 0

        self.recovery_upward_detected = False

        # --------------------------------------------------------
        # Last status
        # --------------------------------------------------------

        self.last_confidence = 0.0

        self.last_movement = 0.0

        self.last_vertical_movement = 0.0

        self.last_direction = "STABLE"

        self.last_ratio = 0.0

        self.last_ratio_change = 0.0

    # ============================================================
    # RESET TRACKING
    # ============================================================

    def reset_tracking(self):

        self.previous_center = None

        self.previous_ratio = None

        self.movement_history.clear()

        self.ratio_history.clear()

        self.downward_motion_detected = False

        self.fall_candidate = False

        self.candidate_frames = 0

        self.candidate_start_time = 0

        self.candidate_start_ratio = 0.0

        self.max_candidate_ratio = 0.0

        self.max_candidate_ratio_change = 0.0

        self.candidate_score = 0

        self.recovery_frames = 0

        self.recovery_upward_detected = False

    # ============================================================
    # RESET CANDIDATE
    # ============================================================

    def reset_candidate(self):

        self.fall_candidate = False

        self.candidate_frames = 0

        self.candidate_start_time = 0

        self.candidate_start_ratio = 0.0

        self.max_candidate_ratio = 0.0

        self.max_candidate_ratio_change = 0.0

        self.candidate_score = 0

        self.downward_motion_detected = False

    # ============================================================
    # PROCESS FRAME
    # ============================================================

    def process_frame(self, frame):

        # ========================================================
        # YOLO DETECTION
        # ========================================================

        results = self.model(
            frame,
            conf=self.CONFIDENCE_THRESHOLD,
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

            self.missed_frames += 1

            print(
                f"[DETECTOR] No detection | "
                f"Missed frames: {self.missed_frames}"
            )

            # ----------------------------------------------------
            # Short detection gaps
            # ----------------------------------------------------

            if self.missed_frames < self.MAX_MISSED_FRAMES:

                if self.event_active:

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
                    f"Missed: {self.missed_frames}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                return frame, self.get_status()

            # ----------------------------------------------------
            # Too many missed frames
            # ----------------------------------------------------

            self.reset_tracking()

            if self.event_active:

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
                f"Missed: {self.missed_frames}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            return frame, self.get_status()

        # ========================================================
        # DETECTION EXISTS
        # ========================================================

        self.missed_frames = 0

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
        # CENTER
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

        if self.previous_center is not None:

            dx = (
                center_x
                -
                self.previous_center[0]
            )

            dy = (
                center_y
                -
                self.previous_center[1]
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

            if dy > 5:

                direction = "DOWN"

            elif dy < -5:

                direction = "UP"

            else:

                direction = "STABLE"

        # ========================================================
        # PROTECT AGAINST EXTREME BBOX JUMPS
        # ========================================================

        movement_is_reasonable = (

            movement
            <=
            self.MAX_REASONABLE_MOVEMENT
        )

        # ========================================================
        # RATIO CHANGE
        # ========================================================

        ratio_change = 0.0

        if self.previous_ratio is not None:

            ratio_change = (
                ratio
                -
                self.previous_ratio
            )

        # ========================================================
        # HISTORY
        # ========================================================

        self.movement_history.append(
            movement
        )

        self.ratio_history.append(
            ratio
        )

        # ========================================================
        # DOWNWARD MOVEMENT
        # ========================================================

        strong_downward_motion = (

            movement_is_reasonable

            and

            direction == "DOWN"

            and

            movement
            >=
            self.MOVEMENT_THRESHOLD

            and

            vertical_movement
            >=
            self.DOWNWARD_VERTICAL_THRESHOLD
        )

        moderate_downward_motion = (

            movement_is_reasonable

            and

            direction == "DOWN"

            and

            movement
            >=
            self.DOWNWARD_MOVEMENT_THRESHOLD

            and

            vertical_movement
            >=
            self.DOWNWARD_VERTICAL_THRESHOLD
        )

        # ========================================================
        # BODY SHAPE
        # ========================================================

        horizontal_body = (

            ratio
            >=
            self.ASPECT_RATIO_THRESHOLD
        )

        strong_horizontal_body = (

            ratio
            >=
            self.STRONG_HORIZONTAL_RATIO
        )

        very_horizontal_body = (

            ratio
            >=
            self.VERY_HORIZONTAL_RATIO
        )

        # ========================================================
        # RATIO CHANGE
        # ========================================================

        ratio_changed = (

            ratio_change
            >=
            self.RATIO_CHANGE_THRESHOLD
        )

        strong_ratio_change = (

            ratio_change
            >=
            self.STRONG_RATIO_CHANGE
        )

        # ========================================================
        # FALL CANDIDATE
        # ========================================================

        if not self.event_active:

            # ----------------------------------------------------
            # Start candidate when meaningful downward movement
            # happens.
            #
            # IMPORTANT:
            # We DO NOT require ratio >= 2.0 here.
            # ----------------------------------------------------

            start_candidate = (

                strong_downward_motion

                or

                (
                    moderate_downward_motion
                    and
                    ratio_changed
                )
            )

            if start_candidate:

                if not self.fall_candidate:

                    self.fall_candidate = True

                    self.candidate_start_time = time.time()

                    self.candidate_frames = 1

                    self.candidate_start_ratio = ratio

                    self.max_candidate_ratio = ratio

                    self.max_candidate_ratio_change = (
                        ratio_change
                    )

                    self.candidate_score = 0

                    print()

                    print(
                        "--------------------------------------"
                    )

                    print(
                        "[DETECTOR] FALL CANDIDATE STARTED"
                    )

                    print(
                        f"[DETECTOR] "
                        f"Start Ratio: "
                        f"{ratio:.2f}"
                    )

                    print(
                        "--------------------------------------"
                    )

                else:

                    self.candidate_frames += 1

            # ----------------------------------------------------
            # Candidate is active
            # ----------------------------------------------------

            if self.fall_candidate:

                candidate_age = (

                    time.time()
                    -
                    self.candidate_start_time
                )

                # ------------------------------------------------
                # Update candidate information
                # ------------------------------------------------

                if ratio > self.max_candidate_ratio:

                    self.max_candidate_ratio = ratio

                if (
                    ratio_change
                    >
                    self.max_candidate_ratio_change
                ):

                    self.max_candidate_ratio_change = (
                        ratio_change
                    )

                # ------------------------------------------------
                # Remember downward movement
                # ------------------------------------------------

                if strong_downward_motion:

                    self.downward_motion_detected = True

                # ------------------------------------------------
                # Calculate score
                # ------------------------------------------------

                score = 0

                # Strong downward movement
                if strong_downward_motion:

                    score += 2

                # Moderate downward movement
                elif moderate_downward_motion:

                    score += 1

                # Body became horizontal
                if horizontal_body:

                    score += 1

                # Strong horizontal
                if strong_horizontal_body:

                    score += 1

                # Very horizontal
                if very_horizontal_body:

                    score += 1

                # Ratio increased
                if ratio_changed:

                    score += 1

                # Strong ratio increase
                if strong_ratio_change:

                    score += 1

                # Large total shape transition
                total_ratio_change = (

                    ratio
                    -
                    self.candidate_start_ratio
                )

                if total_ratio_change >= 0.30:

                    score += 1

                self.candidate_score = score

                # ------------------------------------------------
                # Confirmation logic
                # ------------------------------------------------

                confirmed = False

                # ------------------------------------------------
                # Pattern 1:
                #
                # Strong downward movement
                # +
                # horizontal posture
                # +
                # multiple frames
                # ------------------------------------------------

                pattern_1 = (

                    self.downward_motion_detected

                    and

                    horizontal_body

                    and

                    self.candidate_frames
                    >=
                    self.CONFIRMATION_FRAMES
                )

                # ------------------------------------------------
                # Pattern 2:
                #
                # Strong downward movement
                # +
                # strong ratio increase
                # +
                # horizontal-ish body
                # ------------------------------------------------

                pattern_2 = (

                    self.downward_motion_detected

                    and

                    total_ratio_change
                    >=
                    0.30

                    and

                    ratio
                    >=
                    self.ASPECT_RATIO_THRESHOLD
                )

                # ------------------------------------------------
                # Pattern 3:
                #
                # Very horizontal body + previous downward
                # movement.
                #
                # Requires candidate first, so a random
                # horizontal box doesn't immediately trigger.
                # ------------------------------------------------

                pattern_3 = (

                    self.downward_motion_detected

                    and

                    very_horizontal_body
                )

                # ------------------------------------------------
                # Score based confirmation
                # ------------------------------------------------

                if (
                    score
                    >=
                    self.FALL_SCORE_THRESHOLD
                ):

                    confirmed = True

                if pattern_1:

                    confirmed = True

                if pattern_2:

                    confirmed = True

                if pattern_3:

                    confirmed = True

                # ------------------------------------------------
                # FALL CONFIRMED
                # ------------------------------------------------

                if confirmed:

                    self.event_active = True

                    self.event_start_time = time.time()

                    print()

                    print(
                        "======================================"
                    )

                    print(
                        "[DETECTOR] FALL DETECTED!"
                    )

                    print(
                        "======================================"
                    )

                    print(
                        f"[DETECTOR] "
                        f"Candidate Frames: "
                        f"{self.candidate_frames}"
                    )

                    print(
                        f"[DETECTOR] "
                        f"Start Ratio: "
                        f"{self.candidate_start_ratio:.2f}"
                    )

                    print(
                        f"[DETECTOR] "
                        f"Current Ratio: "
                        f"{ratio:.2f}"
                    )

                    print(
                        f"[DETECTOR] "
                        f"Total Ratio Change: "
                        f"{total_ratio_change:.2f}"
                    )

                    print(
                        f"[DETECTOR] "
                        f"Score: {score}"
                    )

                    print(
                        "======================================"
                    )

                    print()

                    self.reset_candidate()

                    self.recovery_frames = 0

                    self.recovery_upward_detected = False

                # ------------------------------------------------
                # Candidate expired
                # ------------------------------------------------

                elif candidate_age > self.CANDIDATE_TIMEOUT:

                    print()

                    print(
                        "--------------------------------------"
                    )

                    print(
                        "[DETECTOR] FALL CANDIDATE EXPIRED"
                    )

                    print(
                        f"[DETECTOR] "
                        f"Maximum Ratio: "
                        f"{self.max_candidate_ratio:.2f}"
                    )

                    print(
                        "--------------------------------------"
                    )

                    print()

                    self.reset_candidate()

        # ========================================================
        # RECOVERY
        # ========================================================

        if self.event_active:

            # ----------------------------------------------------
            # Upright body
            # ----------------------------------------------------

            upright_body = (

                ratio
                <
                self.RECOVERY_RATIO_THRESHOLD
            )

            # ----------------------------------------------------
            # Moving upward
            # ----------------------------------------------------

            getting_up = (

                movement_is_reasonable

                and

                direction == "UP"

                and

                abs(vertical_movement)
                >=
                self.RECOVERY_UPWARD_THRESHOLD
            )

            # ----------------------------------------------------
            # Remember upward movement
            # ----------------------------------------------------

            if getting_up:

                self.recovery_upward_detected = True

            # ----------------------------------------------------
            # Recovery should require BOTH:
            #
            # 1. upright posture
            # 2. evidence of getting up OR stable upright body
            #
            # This prevents a random low ratio from immediately
            # clearing the fall.
            # ----------------------------------------------------

            recovery_condition = (

                upright_body

                and

                (
                    self.recovery_upward_detected
                    or
                    movement <= self.RECOVERY_MOVEMENT_THRESHOLD
                )
            )

            if recovery_condition:

                self.recovery_frames += 1

            else:

                # Don't instantly reset if there is small
                # movement variation.
                if not upright_body:

                    self.recovery_frames = max(
                        0,
                        self.recovery_frames - 1
                    )

            # ----------------------------------------------------
            # Recovery confirmed
            # ----------------------------------------------------

            if (
                self.recovery_frames
                >=
                self.RECOVERY_FRAMES_REQUIRED
            ):

                self.event_active = False

                self.recovery_frames = 0

                self.recovery_upward_detected = False

                print()

                print(
                    "======================================"
                )

                print(
                    "[DETECTOR] PERSON RECOVERED"
                )

                print(
                    "[DETECTOR] RETURNING TO MONITORING"
                )

                print(
                    "======================================"
                )

                print()

        # ========================================================
        # EVENT TIMER
        # ========================================================

        event_timer = 0

        if self.event_active:

            event_timer = int(
                time.time()
                -
                self.event_start_time
            )

        # ========================================================
        # HORIZONTAL FRAMES
        # ========================================================

        horizontal_frames = 0

        if horizontal_body:

            horizontal_frames = 1

        # ========================================================
        # DRAW BOX
        # ========================================================

        if self.event_active:

            box_color = (
                0,
                0,
                255
            )

            label = "FALL DETECTED"

        elif self.fall_candidate:

            box_color = (
                0,
                165,
                255
            )

            label = "FALL CANDIDATE"

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

        if self.event_active:

            status_text = "FALL DETECTED"

            status_color = (
                0,
                0,
                255
            )

        elif self.fall_candidate:

            status_text = "FALL CANDIDATE"

            status_color = (
                0,
                165,
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

        if self.event_active:

            recovery_text = (

                f"Recovery: "
                f"{self.recovery_frames}/"
                f"{self.RECOVERY_FRAMES_REQUIRED}"
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
        # DEBUG INFORMATION
        # ========================================================

        debug_y = 125

        debug_lines = [

            f"Confidence: {best_confidence:.2f}",

            f"Movement: {movement:.1f}px",

            f"Vertical: {vertical_movement:.1f}px",

            f"Direction: {direction}",

            f"Ratio: {ratio:.2f}",

            f"RatioChange: {ratio_change:.2f}",

            f"HorizontalFrames: {horizontal_frames}",

            f"Candidate: {self.fall_candidate}",

            f"CandidateFrames: {self.candidate_frames}",

            f"Score: {self.candidate_score}",

            f"EventActive: {self.event_active}",

            f"Recovery: "
            f"{self.recovery_frames}/"
            f"{self.RECOVERY_FRAMES_REQUIRED}",

            f"EventTimer: {event_timer}"
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

            debug_y += 22

        # ========================================================
        # UPDATE PREVIOUS VALUES
        # ========================================================

        self.previous_center = current_center

        self.previous_ratio = ratio

        # ========================================================
        # SAVE LAST STATUS
        # ========================================================

        self.last_confidence = best_confidence

        self.last_movement = movement

        self.last_vertical_movement = vertical_movement

        self.last_direction = direction

        self.last_ratio = ratio

        self.last_ratio_change = ratio_change

        # ========================================================
        # RETURN
        # ========================================================

        return frame, self.get_status(
            confidence=best_confidence,
            movement=movement,
            vertical_movement=vertical_movement,
            direction=direction,
            ratio=ratio,
            ratio_change=ratio_change,
            horizontal_frames=horizontal_frames,
            score=self.candidate_score,
            event_timer=event_timer
        )

    # ============================================================
    # GET STATUS
    # ============================================================

    def get_status(
        self,
        confidence=0.0,
        movement=0.0,
        vertical_movement=0.0,
        direction="STABLE",
        ratio=0.0,
        ratio_change=0.0,
        horizontal_frames=0,
        score=0,
        event_timer=0
    ):

        if self.event_active:

            status = "FALL DETECTED"

        elif self.fall_candidate:

            status = "FALL CANDIDATE"

        else:

            status = "MONITORING"

        return {

            "status":
                status,

            "event_active":
                self.event_active,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "movement":
                round(
                    movement,
                    1
                ),

            "vertical_movement":
                round(
                    vertical_movement,
                    1
                ),

            "direction":
                direction,

            "ratio":
                round(
                    ratio,
                    2
                ),

            "ratio_change":
                round(
                    ratio_change,
                    2
                ),

            "horizontal_frames":
                horizontal_frames,

            "fall_candidate":
                self.fall_candidate,

            "candidate_frames":
                self.candidate_frames,

            "score":
                score,

            "recovery_frames":
                self.recovery_frames,

            "recovery_required":
                self.RECOVERY_FRAMES_REQUIRED,

            "event_timer":
                event_timer,

            "missed_frames":
                self.missed_frames
        }