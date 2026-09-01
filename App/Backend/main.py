from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

from App.Backend.detector_runner import (
    start_detector,
    stop_detector,
    get_status,
    get_event
)


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Fall Detection AI Backend",
    description="Backend API for the Fall Detection AI system",
    version="1.2.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# CAMERA FRAME PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


CAMERA_FRAME_PATH = os.path.join(
    PROJECT_ROOT,
    "camera_frames",
    "latest.jpg"
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "message":
        "Fall Detection AI Backend is running",

        "status":
        "online"

    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
        "healthy",

        "service":
        "fall-detection-backend"

    }


# ============================================================
# START FALL DETECTOR
# ============================================================

@app.post("/detector/start")
def start():

    started = start_detector()

    current_status = get_status()

    return {

        "success":
        started,

        "detector":
        current_status

    }


# ============================================================
# STOP FALL DETECTOR
# ============================================================

@app.post("/detector/stop")
def stop():

    stopped = stop_detector()

    current_status = get_status()

    return {

        "success":
        stopped,

        "detector":
        current_status

    }


# ============================================================
# GET COMPLETE DETECTOR STATUS
# ============================================================

@app.get("/detector/status")
def status():

    return get_status()


# ============================================================
# GET FALL EVENT
# ============================================================

@app.get("/detector/event")
def detector_event():

    current_status = get_status()

    return {

        "event_active":
        current_status.get(
            "event_active",
            False
        ),

        "status":
        current_status.get(
            "status",
            "MONITORING"
        ),

        "event_timer":
        current_status.get(
            "event_timer",
            0
        ),

        "recovery":
        current_status.get(
            "recovery",
            "0/5"
        )

    }


# ============================================================
# GET LIVE CAMERA FRAME
# ============================================================

@app.get("/camera/frame")
def camera_frame():

    # --------------------------------------------------------
    # Check whether the detector has produced a frame
    # --------------------------------------------------------

    if not os.path.exists(CAMERA_FRAME_PATH):

        return {

            "error":
            "Camera frame not available",

            "message":
            "Start the Fall Detection Engine first."

        }


    # --------------------------------------------------------
    # Return latest annotated camera frame
    # --------------------------------------------------------

    return FileResponse(

        CAMERA_FRAME_PATH,

        media_type="image/jpeg",

        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }

    )