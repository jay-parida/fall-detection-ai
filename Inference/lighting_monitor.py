import cv2
import os
import threading
import time


# ============================================================
# LIGHTING MONITOR
# ============================================================

class LightingMonitor:

    def __init__(self, frame_path, interval=0.5):

        self.frame_path = frame_path
        self.interval = interval

        self.running = False
        self.thread = None

        self.lock = threading.Lock()

        self.current_lighting = {

            "lighting": "UNKNOWN",

            "brightness": 0.0,

            "contrast": 0.0,

            "blur_score": 0.0
        }


    # ========================================================
    # ANALYZE FRAME
    # ========================================================

    def analyze_frame(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Brightness
        # ----------------------------------------------------

        brightness = float(
            gray.mean()
        )


        # ----------------------------------------------------
        # Contrast
        # ----------------------------------------------------

        contrast = float(
            gray.std()
        )


        # ----------------------------------------------------
        # Blur / sharpness
        # ----------------------------------------------------

        blur_score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )


        # ----------------------------------------------------
        # Lighting category
        # ----------------------------------------------------

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

            "brightness": round(
                brightness,
                2
            ),

            "contrast": round(
                contrast,
                2
            ),

            "blur_score": round(
                blur_score,
                2
            )
        }


    # ========================================================
    # MONITOR LOOP
    # ========================================================

    def monitor_loop(self):

        print(
            "[LIGHTING] Monitor started.",
            flush=True
        )


        while self.running:

            try:

                if os.path.exists(
                    self.frame_path
                ):

                    frame = cv2.imread(
                        self.frame_path
                    )


                    if frame is not None:

                        result = (
                            self.analyze_frame(
                                frame
                            )
                        )


                        with self.lock:

                            self.current_lighting = (
                                result
                            )


            except Exception as e:

                print(
                    f"[LIGHTING] "
                    f"Analysis error: {e}",
                    flush=True
                )


            time.sleep(
                self.interval
            )


        print(
            "[LIGHTING] Monitor stopped.",
            flush=True
        )


    # ========================================================
    # START
    # ========================================================

    def start(self):

        if self.running:

            return


        self.running = True


        self.thread = threading.Thread(

            target=self.monitor_loop,

            daemon=True,

            name="LightingMonitor"

        )


        self.thread.start()


    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.running = False


        if self.thread is not None:

            self.thread.join(
                timeout=2
            )

            self.thread = None


    # ========================================================
    # GET CURRENT RESULT
    # ========================================================

    def get_status(self):

        with self.lock:

            return self.current_lighting.copy()