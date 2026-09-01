import subprocess
import threading
import re
import os
import sys
import time
import uuid


# ============================================================
# FALL DETECTOR PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

DETECTOR_PATH = os.path.join(
    PROJECT_ROOT,
    "fall_detector.py"
)


# ============================================================
# DETECTOR PROCESS
# ============================================================

detector_process = None


# ============================================================
# CURRENT DETECTOR STATUS
# ============================================================

current_status = {

    "status": "STOPPED",

    "event_active": False,

    "confidence": 0.0,

    "movement": 0.0,

    "vertical": 0.0,

    "direction": "STABLE",

    "ratio": 0.0,

    "ratio_change": 0.0,

    "recovery": "0/5",

    "candidate": False,

    "score": 0,

    "event_timer": 0,

    "process_running": False
}


# ============================================================
# CURRENT FALL EVENT
# ============================================================

current_event = {

    "event_id": None,

    "active": False,

    "start_time": None,

    "end_time": None,

    "duration": 0,

    "status": "NO_ACTIVE_EVENT"
}


# ============================================================
# STATUS LOCK
# ============================================================

# IMPORTANT:
#
# RLock allows the same thread to acquire the lock multiple
# times safely.
#
# This prevents the deadlock that could happen when:
#
# read_detector_output()
#       ↓
# parse_status_line()
#       ↓
# start_fall_event()
#
# all use the same lock.
#
status_lock = threading.RLock()


# ============================================================
# RESET STATUS
# ============================================================

def reset_status(status="STOPPED"):

    global current_status

    with status_lock:

        current_status = {

            "status": status,

            "event_active": False,

            "confidence": 0.0,

            "movement": 0.0,

            "vertical": 0.0,

            "direction": "STABLE",

            "ratio": 0.0,

            "ratio_change": 0.0,

            "recovery": "0/5",

            "candidate": False,

            "score": 0,

            "event_timer": 0,

            "process_running": False
        }


# ============================================================
# RESET EVENT
# ============================================================

def reset_event():

    global current_event

    with status_lock:

        current_event = {

            "event_id": None,

            "active": False,

            "start_time": None,

            "end_time": None,

            "duration": 0,

            "status": "NO_ACTIVE_EVENT"
        }


# ============================================================
# START FALL EVENT
# ============================================================

def start_fall_event():

    global current_event

    with status_lock:

        # ----------------------------------------------------
        # Prevent duplicate event
        # ----------------------------------------------------

        if current_event["active"]:

            return


        now = time.time()


        current_event = {

            "event_id":
                str(uuid.uuid4()),

            "active":
                True,

            "start_time":
                now,

            "end_time":
                None,

            "duration":
                0,

            "status":
                "FALL DETECTED"
        }


        print(
            "[EVENT] Fall event started.",
            flush=True
        )


        print(
            f"[EVENT] Event ID: "
            f"{current_event['event_id']}",
            flush=True
        )


# ============================================================
# END FALL EVENT
# ============================================================

def end_fall_event():

    global current_event

    with status_lock:

        # ----------------------------------------------------
        # Nothing to end
        # ----------------------------------------------------

        if not current_event["active"]:

            return


        now = time.time()


        start_time = (
            current_event["start_time"]
        )


        duration = 0


        if start_time is not None:

            duration = round(
                now - start_time,
                2
            )


        current_event["active"] = False

        current_event["end_time"] = now

        current_event["duration"] = duration

        current_event["status"] = "RECOVERED"


        print(
            "[EVENT] Fall event ended.",
            flush=True
        )


        print(
            f"[EVENT] Duration: "
            f"{duration:.2f} seconds",
            flush=True
        )


# ============================================================
# PARSE DETECTOR STATUS LINE
# ============================================================

def parse_status_line(line):

    global current_status


    # --------------------------------------------------------
    # Only parse the main detector status line
    # --------------------------------------------------------

    if "Confidence:" not in line:

        return


    if "EventActive:" not in line:

        return


    # ========================================================
    # CONFIDENCE
    # ========================================================

    match = re.search(
        r"Confidence:\s*([0-9.]+)",
        line
    )

    if match:

        current_status["confidence"] = float(
            match.group(1)
        )


    # ========================================================
    # MOVEMENT
    # ========================================================

    match = re.search(
        r"Movement:\s*([0-9.]+)px",
        line
    )

    if match:

        current_status["movement"] = float(
            match.group(1)
        )


    # ========================================================
    # VERTICAL MOVEMENT
    # ========================================================

    match = re.search(
        r"Vertical:\s*(-?[0-9.]+)px",
        line
    )

    if match:

        current_status["vertical"] = float(
            match.group(1)
        )


    # ========================================================
    # DIRECTION
    # ========================================================

    match = re.search(
        r"Direction:\s*([A-Za-z]+)",
        line
    )

    if match:

        current_status["direction"] = (
            match.group(1)
        )


    # ========================================================
    # RATIO
    # ========================================================

    match = re.search(
        r"Ratio:\s*([0-9.]+)",
        line
    )

    if match:

        current_status["ratio"] = float(
            match.group(1)
        )


    # ========================================================
    # RATIO CHANGE
    # ========================================================

    match = re.search(
        r"RatioChange:\s*(-?[0-9.]+)",
        line
    )

    if match:

        current_status["ratio_change"] = float(
            match.group(1)
        )


    # ========================================================
    # CANDIDATE
    # ========================================================

    match = re.search(
        r"Candidate:\s*(True|False)",
        line
    )

    if match:

        current_status["candidate"] = (
            match.group(1) == "True"
        )


    # ========================================================
    # SCORE
    # ========================================================

    match = re.search(
        r"Score:\s*(\d+)",
        line
    )

    if match:

        current_status["score"] = int(
            match.group(1)
        )


    # ========================================================
    # EVENT ACTIVE
    # ========================================================

    match = re.search(
        r"EventActive:\s*(True|False)",
        line
    )

    if match:

        event_active = (
            match.group(1) == "True"
        )


        current_status["event_active"] = (
            event_active
        )


        if event_active:

            current_status["status"] = (
                "FALL DETECTED"
            )


            # ------------------------------------------------
            # Start event if not already active
            # ------------------------------------------------

            start_fall_event()


        else:

            # ------------------------------------------------
            # IMPORTANT:
            #
            # EventActive=False does NOT automatically mean
            # recovered.
            #
            # Recovery should normally be confirmed by:
            #
            # PERSON RECOVERED
            #
            # from the detector.
            # ------------------------------------------------

            if not current_event["active"]:

                current_status["status"] = (
                    "MONITORING"
                )


    # ========================================================
    # RECOVERY
    # ========================================================

    match = re.search(
        r"Recovery:\s*(\d+/\d+)",
        line
    )

    if match:

        current_status["recovery"] = (
            match.group(1)
        )


    # ========================================================
    # EVENT TIMER
    # ========================================================

    match = re.search(
        r"EventTimer:\s*(\d+)",
        line
    )

    if match:

        current_status["event_timer"] = int(
            match.group(1)
        )


# ============================================================
# READ DETECTOR OUTPUT
# ============================================================

def read_detector_output():

    global detector_process
    global current_status


    process = detector_process


    if process is None:

        return


    try:

        for raw_line in iter(
            process.stdout.readline,
            ""
        ):

            if not raw_line:

                break


            line = raw_line.strip()


            if not line:

                continue


            # ------------------------------------------------
            # Show detector output in backend terminal
            # ------------------------------------------------

            print(
                f"[DETECTOR] {line}",
                flush=True
            )


            # =================================================
            # FALL DETECTED
            # =================================================

            if "FALL DETECTED!" in line:

                with status_lock:

                    current_status["status"] = (
                        "FALL DETECTED"
                    )

                    current_status["event_active"] = (
                        True
                    )


                start_fall_event()


            # =================================================
            # PERSON RECOVERED
            # =================================================

            elif "PERSON RECOVERED" in line:

                with status_lock:

                    current_status["status"] = (
                        "MONITORING"
                    )

                    current_status["event_active"] = (
                        False
                    )

                    current_status["recovery"] = (
                        "0/5"
                    )


                end_fall_event()


            # =================================================
            # NORMAL STATUS LINE
            # =================================================

            with status_lock:

                parse_status_line(line)


    except Exception as e:

        print(
            f"[DETECTOR] Output reader error: {e}",
            flush=True
        )


    finally:

        with status_lock:

            current_status[
                "process_running"
            ] = False


        print(
            "[BACKEND] Detector output reader stopped.",
            flush=True
        )


# ============================================================
# START DETECTOR
# ============================================================

def start_detector():

    global detector_process


    # ========================================================
    # CHECK IF ALREADY RUNNING
    # ========================================================

    if detector_process is not None:

        if detector_process.poll() is None:

            print(
                "[BACKEND] Detector is already running.",
                flush=True
            )

            return False


    # ========================================================
    # CHECK DETECTOR FILE
    # ========================================================

    if not os.path.exists(DETECTOR_PATH):

        print(
            f"[BACKEND] ERROR: Detector not found:"
            f" {DETECTOR_PATH}",
            flush=True
        )

        return False


    print(
        "[BACKEND] Starting Fall Detection Engine...",
        flush=True
    )


    print(
        f"[BACKEND] Detector path: {DETECTOR_PATH}",
        flush=True
    )


    # ========================================================
    # RESET STATUS
    # ========================================================

    reset_status("STARTING")

    reset_event()


    # ========================================================
    # START PROCESS
    # ========================================================

    try:

        detector_process = subprocess.Popen(

            [
                sys.executable,
                "-u",
                DETECTOR_PATH
            ],

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            stdin=subprocess.PIPE,

            text=True,

            bufsize=1

        )


    except Exception as e:

        print(
            f"[BACKEND] Failed to start detector: {e}",
            flush=True
        )

        detector_process = None

        reset_status("STOPPED")

        reset_event()

        return False


    # ========================================================
    # MARK PROCESS RUNNING
    # ========================================================

    with status_lock:

        current_status[
            "process_running"
        ] = True

        current_status[
            "status"
        ] = "MONITORING"


    # ========================================================
    # START OUTPUT READER THREAD
    # ========================================================

    thread = threading.Thread(

        target=read_detector_output,

        daemon=True,

        name="FallDetectorOutputReader"

    )


    thread.start()


    print(
        "[BACKEND] Fall Detection Engine started.",
        flush=True
    )


    return True


# ============================================================
# STOP DETECTOR
# ============================================================

def stop_detector():

    global detector_process


    if detector_process is None:

        print(
            "[BACKEND] Detector is not running.",
            flush=True
        )

        return False


    process = detector_process


    # ========================================================
    # STOP PROCESS
    # ========================================================

    if process.poll() is None:

        print(
            "[BACKEND] Stopping detector...",
            flush=True
        )


        try:

            process.terminate()

            process.wait(
                timeout=5
            )


        except subprocess.TimeoutExpired:

            print(
                "[BACKEND] Force stopping detector...",
                flush=True
            )

            process.kill()

            process.wait()


        except Exception as e:

            print(
                f"[BACKEND] Stop error: {e}",
                flush=True
            )


    # ========================================================
    # CLEAR PROCESS
    # ========================================================

    detector_process = None


    # ========================================================
    # RESET STATUS
    # ========================================================

    reset_status("STOPPED")

    reset_event()


    print(
        "[BACKEND] Detector stopped.",
        flush=True
    )


    return True


# ============================================================
# GET STATUS
# ============================================================

def get_status():

    global detector_process


    with status_lock:

        status = current_status.copy()


        # ----------------------------------------------------
        # Get real process state
        # ----------------------------------------------------

        if detector_process is not None:

            running = (
                detector_process.poll() is None
            )


            status[
                "process_running"
            ] = running


            # ------------------------------------------------
            # Unexpected process termination
            # ------------------------------------------------

            if not running:

                status[
                    "status"
                ] = "STOPPED"

                status[
                    "event_active"
                ] = False


        else:

            status[
                "process_running"
            ] = False


    return status


# ============================================================
# GET FALL EVENT
# ============================================================

def get_event():

    with status_lock:

        event = current_event.copy()


        # ----------------------------------------------------
        # Update active event duration
        # ----------------------------------------------------

        if event["active"]:

            if event["start_time"] is not None:

                event["duration"] = round(
                    time.time()
                    -
                    event["start_time"],
                    2
                )


    return event