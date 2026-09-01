import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraStarting, setCameraStarting] = useState(false);

  /*
   * This ref is used so the 500ms status polling does not
   * immediately show the old camera frame while the CPU
   * detector is still starting.
   */
  const cameraStartingRef = useRef(false);


  // ============================================================
  // GET CURRENT DETECTOR STATUS
  // ============================================================

  const fetchStatus = async () => {
    try {
      const response = await fetch(
        `${API_URL}/detector/status`
      );

      if (!response.ok) {
        throw new Error("Failed to fetch detector status");
      }

      const data = await response.json();

      setStatus(data);

      /*
       * IMPORTANT:
       *
       * While the detector is starting, do NOT allow the
       * status polling to activate the camera.
       *
       * This prevents the old camera frame from appearing
       * before the new camera feed is ready.
       */
      if (!cameraStartingRef.current) {
        setCameraActive(
          data.process_running === true
        );
      }

    } catch (error) {
      console.error("Status error:", error);

      setStatus({
        status: "BACKEND OFFLINE",
        event_active: false,
        process_running: false,
        confidence: 0,
        movement: 0,
        vertical: 0,
        direction: "STABLE",
        ratio: 0,
        ratio_change: 0,
        recovery: "0/5",
        candidate: false,
        score: 0,
        event_timer: 0,
      });

      setCameraActive(false);
      setCameraReady(false);
      setCameraStarting(false);

      cameraStartingRef.current = false;
    }
  };


  // ============================================================
  // START DETECTOR
  // ============================================================

  const startDetector = async () => {

    setLoading(true);

    /*
     * Immediately hide everything related to the previous
     * camera frame.
     */
    setCameraActive(false);
    setCameraReady(false);
    setCameraStarting(true);

    cameraStartingRef.current = true;

    try {

      const response = await fetch(
        `${API_URL}/detector/start`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error("Failed to start detector");
      }

      const data = await response.json();

      setStatus(data.detector);

      /*
       * The backend may say "running" before the CPU has
       * actually finished initializing the camera + YOLO.
       *
       * Therefore we intentionally wait before displaying
       * the camera.
       *
       * 40 seconds is suitable for the current CPU setup.
       */
      setTimeout(() => {

        cameraStartingRef.current = false;

        setCameraStarting(false);

        setCameraActive(true);

      }, 40000);

    } catch (error) {

      console.error("Start error:", error);

      alert("Could not start the detector.");

      setCameraActive(false);
      setCameraReady(false);
      setCameraStarting(false);

      cameraStartingRef.current = false;
    }

    setLoading(false);
  };


  // ============================================================
  // STOP DETECTOR
  // ============================================================

  const stopDetector = async () => {

    setLoading(true);

    /*
     * Immediately hide the camera.
     */
    setCameraActive(false);
    setCameraReady(false);
    setCameraStarting(false);

    cameraStartingRef.current = false;

    try {

      const response = await fetch(
        `${API_URL}/detector/stop`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error("Failed to stop detector");
      }

      const data = await response.json();

      setStatus(data.detector);

      setCameraActive(false);
      setCameraReady(false);
      setCameraStarting(false);

    } catch (error) {

      console.error("Stop error:", error);

      alert("Could not stop the detector.");
    }

    setLoading(false);
  };


  // ============================================================
  // INITIAL STATUS
  // ============================================================

  useEffect(() => {

    fetchStatus();

  }, []);


  // ============================================================
  // AUTO UPDATE STATUS
  // ============================================================

  useEffect(() => {

    const interval = setInterval(() => {

      fetchStatus();

    }, 500);

    return () => clearInterval(interval);

  }, []);


  // ============================================================
  // STATUS HELPERS
  // ============================================================

  const isFallDetected =
    status?.event_active === true ||
    status?.status === "FALL DETECTED";


  const isRunning =
    status?.process_running === true;


  // ============================================================
  // CAMERA URL
  //
  // Timestamp prevents browser image caching.
  // ============================================================

  const cameraUrl =
    `${API_URL}/camera/frame?t=${Date.now()}`;


  // ============================================================
  // FORMAT EVENT TIMER
  // ============================================================

  const formatTimer = (seconds) => {

    const value = Number(seconds || 0);

    const minutes = Math.floor(
      value / 60
    );

    const remainingSeconds =
      value % 60;

    return `${String(minutes).padStart(2, "0")}:${String(
      remainingSeconds
    ).padStart(2, "0")}`;
  };


  return (
    <div className="app">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="header">

        <div className="brand">

          <div className="brand-icon">
            🛡️
          </div>

          <div>

            <h1>
              Fall Detection AI
            </h1>

            <p>
              Intelligent safety monitoring system
            </p>

          </div>

        </div>


        <div className="header-right">

          <div className="system-status">

            <span
              className={`system-dot ${
                isRunning
                  ? "active"
                  : ""
              }`}
            />

            <div>

              <strong>
                {isRunning
                  ? "System Online"
                  : "System Offline"}
              </strong>

              <span>
                {isRunning
                  ? "AI monitoring active"
                  : "Detector stopped"}
              </span>

            </div>

          </div>

        </div>

      </header>


      {/* ======================================================
          MAIN
      ====================================================== */}

      <main className="dashboard">


        {/* ====================================================
            PAGE INTRO
        ==================================================== */}

        <div className="page-heading">

          <div>

            <span className="eyebrow">
              AI SAFETY MONITOR
            </span>

            <h2>
              Real-Time Monitoring
            </h2>

            <p>
              Monitor the camera feed and AI detection
              activity in real time.
            </p>

          </div>

          <div
            className={`monitor-badge ${
              isFallDetected
                ? "danger"
                : isRunning
                  ? "safe"
                  : "offline"
            }`}
          >

            <span />

            {isFallDetected
              ? "Fall Detected"
              : isRunning
                ? "Monitoring"
                : "Offline"}

          </div>

        </div>


        {/* ====================================================
            FALL ALERT
        ==================================================== */}

        {isFallDetected && (

          <section className="fall-alert">

            <div className="alert-icon">
              ⚠️
            </div>

            <div className="alert-content">

              <span>
                IMMEDIATE ATTENTION
              </span>

              <h3>
                Fall Detected
              </h3>

              <p>
                A possible fall has been detected.
                Please check the person immediately.
              </p>

            </div>

            <div className="event-time">

              <span>
                EVENT TIME
              </span>

              <strong>
                {formatTimer(
                  status?.event_timer
                )}
              </strong>

            </div>

          </section>

        )}


        {/* ====================================================
            CAMERA
        ==================================================== */}

        <section className="camera-card">

          <div className="camera-header">

            <div className="camera-heading">

              <div className="camera-icon">
                📷
              </div>

              <div>

                <h3>
                  Live Camera
                </h3>

                <p>
                  Real-time AI detection feed
                </p>

              </div>

            </div>


            <div
              className={`camera-live ${
                isRunning
                  ? "active"
                  : ""
              }`}
            >

              <span />

              {isRunning
                ? "LIVE"
                : "OFFLINE"}

            </div>

          </div>


          <div className="camera-container">

            {/* =================================================
                CAMERA OFFLINE
            ================================================= */}

            {!cameraActive &&
              !cameraStarting &&
              !loading && (

              <div className="camera-placeholder">

                <div className="placeholder-circle">
                  📷
                </div>

                <h3>
                  Camera Offline
                </h3>

                <p>
                  Start the detector to begin
                  live monitoring.
                </p>

              </div>

            )}


            {/* =================================================
                CAMERA STARTING / CPU WARM-UP
            ================================================= */}

            {cameraStarting && (

              <div className="camera-loading">

                <div className="loading-camera-icon">
                  📷
                </div>

                <div className="loading-spinner" />

                <h3>
                  Starting Camera
                </h3>

                <p>
                  Initializing AI detection engine...
                </p>

                <span>
                  CPU processing may take a few seconds
                </span>

              </div>

            )}


            {/* =================================================
                REAL CAMERA
            ================================================= */}

            {cameraActive && (

              <img
                src={cameraUrl}
                alt="Live Fall Detection Camera"
                className={`camera-feed ${
                  cameraReady
                    ? "camera-visible"
                    : "camera-hidden"
                }`}
                onLoad={() => {

                  /*
                   * The image has successfully loaded.
                   *
                   * Only now do we reveal it.
                   */
                  setCameraReady(true);

                }}
                onError={() => {

                  /*
                   * Keep camera hidden if the frame
                   * has not loaded correctly.
                   */
                  setCameraReady(false);

                }}
              />

            )}


            {/* =================================================
                FALL CAMERA ALERT
            ================================================= */}

            {isFallDetected &&
              cameraReady && (

              <div className="camera-alert">

                <span>
                  ⚠
                </span>

                FALL DETECTED

              </div>

            )}

          </div>

        </section>


        {/* ====================================================
            CONTROL BAR
        ==================================================== */}

        <section className="control-card">

          <div className="control-info">

            <div className="control-icon">
              ⚙️
            </div>

            <div>

              <h3>
                Detection Control
              </h3>

              <p>
                Start or stop the AI monitoring engine.
              </p>

            </div>

          </div>


          <div className="controls">

            <button
              className="start-button"
              onClick={startDetector}
              disabled={
                loading ||
                isRunning
              }
            >

              <span>
                ▶
              </span>

              {loading
                ? "Starting..."
                : "Start Detection"}

            </button>


            <button
              className="stop-button"
              onClick={stopDetector}
              disabled={
                loading ||
                !isRunning
              }
            >

              <span>
                ■
              </span>

              {loading
                ? "Stopping..."
                : "Stop Detection"}

            </button>

          </div>

        </section>


        {/* ====================================================
            METRICS
        ==================================================== */}

        <section className="metrics-grid">


          {/* Confidence */}

          <div className="metric-card">

            <div className="metric-top">

              <span>
                Confidence
              </span>

              <div className="metric-icon">
                ◎
              </div>

            </div>

            <strong>

              {status?.confidence !== undefined

                ? `${(
                    status.confidence * 100
                  ).toFixed(0)}%`

                : "--"}

            </strong>

            <small>
              Detection confidence
            </small>

          </div>


          {/* Movement */}

          <div className="metric-card">

            <div className="metric-top">

              <span>
                Movement
              </span>

              <div className="metric-icon">
                ↕
              </div>

            </div>

            <strong>

              {status?.movement !== undefined

                ? `${Number(
                    status.movement
                  ).toFixed(1)} px`

                : "--"}

            </strong>

            <small>
              Total movement
            </small>

          </div>


          {/* Ratio */}

          <div className="metric-card">

            <div className="metric-top">

              <span>
                Body Ratio
              </span>

              <div className="metric-icon">
                ▱
              </div>

            </div>

            <strong>

              {status?.ratio !== undefined

                ? Number(
                    status.ratio
                  ).toFixed(2)

                : "--"}

            </strong>

            <small>
              Width / height
            </small>

          </div>


          {/* Recovery */}

          <div className="metric-card">

            <div className="metric-top">

              <span>
                Recovery
              </span>

              <div className="metric-icon">
                ✓
              </div>

            </div>

            <strong>

              {status?.recovery ||
                "0/5"}

            </strong>

            <small>
              Stable frames
            </small>

          </div>

        </section>


        {/* ====================================================
            DETECTOR DETAILS
        ==================================================== */}

        <section className="details-card">

          <div className="details-header">

            <div className="details-title">

              <div className="details-icon">
                ✦
              </div>

              <div>

                <h3>
                  Detector Details
                </h3>

                <p>
                  Live information from the AI engine
                </p>

              </div>

            </div>


            <span
              className={`live-indicator ${
                isRunning
                  ? "active"
                  : ""
              }`}
            >

              <span />

              {isRunning
                ? "LIVE DATA"
                : "OFFLINE"}

            </span>

          </div>


          <div className="details-grid">


            <div className="detail-item">

              <span>
                Process
              </span>

              <strong>
                {status?.process_running
                  ? "Running"
                  : "Stopped"}
              </strong>

            </div>


            <div className="detail-item">

              <span>
                Event Active
              </span>

              <strong
                className={
                  isFallDetected
                    ? "danger-text"
                    : "safe-text"
                }
              >

                {status?.event_active
                  ? "YES"
                  : "NO"}

              </strong>

            </div>


            <div className="detail-item">

              <span>
                Detection State
              </span>

              <strong>
                {status?.status ||
                  "--"}
              </strong>

            </div>


            <div className="detail-item">

              <span>
                Direction
              </span>

              <strong>
                {status?.direction ||
                  "STABLE"}
              </strong>

            </div>


            <div className="detail-item">

              <span>
                Vertical Movement
              </span>

              <strong>

                {status?.vertical !== undefined

                  ? `${Number(
                      status.vertical
                    ).toFixed(1)} px`

                  : "--"}

              </strong>

            </div>


            <div className="detail-item">

              <span>
                Ratio Change
              </span>

              <strong>

                {status?.ratio_change !== undefined

                  ? Number(
                      status.ratio_change
                    ).toFixed(2)

                  : "--"}

              </strong>

            </div>


            <div className="detail-item">

              <span>
                Candidate
              </span>

              <strong>
                {status?.candidate
                  ? "YES"
                  : "NO"}
              </strong>

            </div>


            <div className="detail-item">

              <span>
                Event Timer
              </span>

              <strong>

                {formatTimer(
                  status?.event_timer
                )}

              </strong>

            </div>

          </div>

        </section>


        {/* ====================================================
            RECOVERY MESSAGE
        ==================================================== */}

        {!isFallDetected &&
          isRunning && (

            <div className="monitoring-message">

              <div className="monitoring-check">
                ✓
              </div>

              <div>

                <strong>
                  System monitoring normally
                </strong>

                <span>
                  No active fall event detected.
                </span>

              </div>

            </div>

          )}

      </main>


      {/* ======================================================
          FOOTER
      ====================================================== */}

      <footer>

        <div>

          <strong>
            Fall Detection AI
          </strong>

          <span>
            Real-time safety monitoring
          </span>

        </div>

        <span>
          AI Monitoring System • v1.0
        </span>

      </footer>

    </div>
  );
}

export default App;