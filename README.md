\# Fall Detection AI



An AI-powered real-time fall detection system that uses a webcam, YOLO object detection, computer vision, FastAPI, and React to monitor a person and detect potential falls.



\## Overview



Fall Detection AI is a real-time computer vision application designed to monitor a camera feed and identify fall events.



The system combines a trained YOLO model with motion and body-position analysis to distinguish normal monitoring situations from potential falls.



When a fall is detected, the application displays a visual alert and continues monitoring the person's recovery.



\## Features



\- Real-time webcam monitoring

\- YOLO-based fall detection

\- Confidence-based detection

\- Body aspect-ratio analysis

\- Movement analysis

\- Downward movement detection

\- Fall candidate confirmation

\- Fall event detection

\- Recovery monitoring

\- Real-time camera feed in the web dashboard

\- FastAPI backend

\- React + Vite frontend

\- Start/Stop detector controls

\- Live detection metrics

\- Detection status dashboard

\- Fall event timer

\- Recovery progress display

\- CPU-compatible inference



\## Technology Stack



\### AI / Computer Vision



\- Python

\- Ultralytics YOLO

\- OpenCV

\- NumPy

\- PyTorch



\### Backend



\- FastAPI

\- Uvicorn

\- Python



\### Frontend



\- React

\- Vite

\- JavaScript

\- CSS



\## Project Structure



```text

Fall\_Detection\_AI/

│

├── App/

│   ├── Backend/

│   │   ├── detector\_runner.py

│   │   └── main.py

│   │

│   └── Frontend/

│       ├── public/

│       ├── src/

│       │   ├── assets/

│       │   ├── App.css

│       │   ├── App.jsx

│       │   ├── index.css

│       │   └── main.jsx

│       ├── package.json

│       ├── package-lock.json

│       ├── index.html

│       └── vite.config.js

│

├── camera\_frames/

│   └── latest.jpg

│

├── Inference/

│   ├── fall\_detection\_engine.py

│   └── test\_engine.py

│

├── Models/

│   └── best.pt

│

├── Datasets/

│

├── Notebooks/

│

├── fall\_detector.py

├── fall\_detector1.py

├── motion\_test.py

├── webcam.py

├── requirements.txt

└── .gitignore

