# PoseCoach AI – Intelligent Real-Time Posture Recognition & Monitoring System

PoseCoach AI is a real-time AI-powered human posture classification, duration tracking, ergonomic analytics, and biomechanical monitoring web application built with MediaPipe Tasks API, Random Forest (Scikit-Learn), Flask, SQLite, and the Web Speech API.

---

## 🌟 Key Features

- **⚡ Turbo Real-Time Vision**: Sub-35ms round-trip inference using MediaPipe Lite (33 landmarks) and optimized Random Forest classification.
- **🦴 33-Landmark Live Skeleton Visualization**: High-performance canvas overlay with glowing joints and bone connection topology.
- **📐 Biomechanical Angles & Ergonomic Feedback**: Real-time calculation of **Spine Inclination Angle** and **Knee Flexion Angle** with live form assessments (*Optimal Alignment*, *Forward Lean Warning*, *Deep Squat Depth*).
- **🏋️‍♂️ Squat Repetition Counter**: Automated biomechanical finite-state machine tracking knee depth and rep counts with voice announcements.
- **🗣️ Multi-Language Voice Assistant (EN & HI)**: Automated posture announcements in English and Hindi, GC-protected Web Speech API integration, zero-latency Web Audio API chimes, and an interactive voice test button.
- **⏱️ Accurate Posture Duration Tracking**: Wall-clock session tracking (`performance.now()`) with Pause, Resume, Stop, and deduplicated transition timelines.
- **📊 Real Analytics Dashboard**: Interactive Chart.js doughnut and bar charts, continuous sitting/bending streak tracking, and 7-day monitoring trends powered by SQLite.
- **📄 Downloadable Executive Reports**: One-click downloadable CSV raw logs and executive PDF session reports generated via ReportLab.
- **🎨 Modern Dark Navy UI**: Responsive 6-tab interface (Dashboard, Live Vision, Tracker, Analytics, History & Reports, Settings) built for desktop, tablet, and mobile.

---

## 🏗️ Technology Stack

- **Computer Vision**: OpenCV, MediaPipe Tasks API (`pose_landmarker_lite.task`)
- **Machine Learning**: Scikit-Learn Random Forest Classifier (`models/posecoach_rf_model.joblib`), 132-dim normalized feature representation (`hip_center_torso_scale_v1`)
- **Backend**: Python 3.12, Flask, Joblib, NumPy, Pandas
- **Storage**: SQLite3 (`posecoach.db`) with transactional session and transition logging
- **Reporting**: ReportLab (PDF generation) and CSV streaming
- **Frontend**: Vanilla JavaScript (ES6+), Canvas API, Web Speech API, Web Audio API, Chart.js, HTML5, Modern CSS (Glassmorphism Dark Navy Theme)

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.12+
- Git

### 2. Installation
```powershell
# Clone the repository
git clone <your-repo-url>
cd PoseCoach-AI

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Web Application
```powershell
.\.venv\Scripts\python.exe app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 4. Running the Test Suite
```powershell
.\.venv\Scripts\python.exe test_all_features.py
```

---

## 📁 Project Architecture

```
PoseCoach-AI/
├── app.py                  # Flask application & REST API routes
├── database.py             # SQLite database schemas and queries
├── reports.py              # ReportLab PDF & CSV report generators
├── pose_features.py        # 132-dim normalized pose feature extractor
├── extract_features.py     # Dataset feature extraction script
├── train_model.py          # Random Forest training & validation script
├── test_all_features.py    # Automated end-to-end test suite
├── requirements.txt        # Python package dependencies
├── models/
│   ├── posecoach_rf_model.joblib   # Trained Random Forest classifier
│   └── pose_landmarker_lite.task   # MediaPipe Pose Landmarker model
├── templates/
│   └── index.html          # Modern 6-tab responsive interface
└── static/
    ├── style.css           # Dark navy theme styling
    └── script.js           # Full-stack client engine, TTS & biomechanics
```

---

## 📜 Disclaimer
PoseCoach AI is an ergonomic awareness and fitness self-monitoring tool. It is not intended to provide clinical diagnosis or replace professional medical advice.
