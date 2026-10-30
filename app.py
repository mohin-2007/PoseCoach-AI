import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from pathlib import Path
from threading import Lock
from flask import Flask, render_template, request, jsonify, Response, send_file
import io

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pose_features import extract_normalized_features
import database
import reports


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "posecoach_rf_model.joblib"

POSE_MODEL_PATH = PROJECT_DIR / "models" / "pose_landmarker_lite.task"


# ==================================================
# 2. FLASK APPLICATION
# ==================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Initialize SQLite database tables
database.init_db()


# ==================================================
# 3. LOAD TRAINED MODEL
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"ML model not found: {MODEL_PATH}")

if not POSE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found: {POSE_MODEL_PATH}"
    )

model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]
feature_columns = model_bundle["feature_columns"]

visibility_threshold = model_bundle["visibility_threshold"]

if model_bundle.get("feature_version") != "hip_center_torso_scale_v1":
    raise ValueError("Unexpected model feature representation.")

if len(feature_columns) != 132:
    raise ValueError("Expected 132 model features.")

print("Random Forest model loaded successfully!")


# ==================================================
# 4. INITIALIZE MEDIAPIPE
# ==================================================

base_options = python.BaseOptions(
    model_asset_path=str(POSE_MODEL_PATH)
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

landmarker = vision.PoseLandmarker.create_from_options(options)

# Prevent simultaneous calls to the landmarker
landmarker_lock = Lock()


# ==================================================
# 5. HOME PAGE
# ==================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==================================================
# 6. HEALTH CHECK
# ==================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ready",
        "model": "Random Forest",
        "classes": list(model.classes_),
        "features": len(feature_columns)
    })


# ==================================================
# 7. LIVE POSE PREDICTION API
# ==================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    if "frame" not in request.files:
        return jsonify({
            "error": "No camera frame received."
        }), 400

    uploaded_file = request.files["frame"]

    image_bytes = uploaded_file.read()

    if not image_bytes:
        return jsonify({
            "error": "Empty camera frame."
        }), 400

    # Decode JPEG bytes
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return jsonify({
            "error": "Could not decode camera frame."
        }), 400

    height, width = frame.shape[:2]

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect body landmarks
    try:

        with landmarker_lock:
            result = landmarker.detect(mp_image)

    except Exception:
        app.logger.exception("MediaPipe detection failed.")

        return jsonify({
            "error": "Pose detection failed."
        }), 500

    if not result.pose_landmarks:

        return jsonify({
            "detected": False,
            "posture": None,
            "confidence": 0,
            "landmarks": [],
            "message": "No person detected."
        })

    landmarks = result.pose_landmarks[0]

    # Calculate average landmark visibility
    visibility_values = [
        float(point.visibility or 0.0)
        for point in landmarks
    ]

    average_visibility = (
        sum(visibility_values) / len(visibility_values)
    )

    # Build 33 landmark points list for client skeleton rendering
    landmarks_data = [
        {
            "x": round(float(point.x), 4),
            "y": round(float(point.y), 4),
            "z": round(float(point.z), 4),
            "visibility": round(float(point.visibility or 0.0), 3)
        }
        for point in landmarks
    ]

    if average_visibility < visibility_threshold:

        return jsonify({
            "detected": False,
            "posture": None,
            "confidence": 0,
            "landmarks": landmarks_data,
            "message": "Move into full camera view."
        })

    # Extract the same normalized features used during training
    try:

        features = extract_normalized_features(
            landmarks,
            image_width=width,
            image_height=height
        )

    except ValueError:

        return jsonify({
            "detected": False,
            "posture": None,
            "confidence": 0,
            "landmarks": landmarks_data,
            "message": "Could not extract body features."
        })

    # High-speed numpy inference (eliminates pandas per-frame overhead)
    import warnings
    feature_array = np.asarray([features], dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probabilities = model.predict_proba(feature_array)[0]

    best_index = int(np.argmax(probabilities))

    predicted_posture = str(model.classes_[best_index])

    confidence = float(probabilities[best_index] * 100)

    # Class distribution breakdown
    class_distribution = {
        str(cls_name): round(float(prob * 100), 1)
        for cls_name, prob in zip(model.classes_, probabilities)
    }

    return jsonify({
        "detected": True,
        "posture": predicted_posture,
        "confidence": round(confidence, 2),
        "visibility": round(average_visibility, 3),
        "landmarks": landmarks_data,
        "probabilities": class_distribution
    })


# ==================================================
# 8. SESSION MANAGEMENT API
# ==================================================

@app.route("/api/session/start", methods=["POST"])
def session_start():
    """Start a new posture monitoring session."""
    try:
        session = database.create_session()
        return jsonify({
            "success": True,
            "session": session
        })
    except Exception as e:
        app.logger.exception("Failed to start session.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/pause", methods=["POST"])
def session_pause():
    """Pause an active session."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("sessionId")
    if not session_id:
        active = database.get_active_session()
        if active:
            session_id = active["id"]

    if not session_id:
        return jsonify({"error": "No active session to pause"}), 400

    database.update_session_status(int(session_id), "paused")
    return jsonify({"success": True, "status": "paused", "sessionId": session_id})


@app.route("/api/session/resume", methods=["POST"])
def session_resume():
    """Resume a paused session."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("sessionId")
    if not session_id:
        active = database.get_active_session()
        if active:
            session_id = active["id"]

    if not session_id:
        return jsonify({"error": "No paused session to resume"}), 400

    database.update_session_status(int(session_id), "active")
    return jsonify({"success": True, "status": "active", "sessionId": session_id})


@app.route("/api/session/stop", methods=["POST"])
def session_stop():
    """Finalize and save session summary."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("sessionId")

    if not session_id:
        active = database.get_active_session()
        if active:
            session_id = active["id"]

    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400

    try:
        database.save_session_summary(
            session_id=int(session_id),
            total_duration_seconds=float(data.get("totalDurationSeconds", 0.0)),
            standing_seconds=float(data.get("standingSeconds", 0.0)),
            sitting_seconds=float(data.get("sittingSeconds", 0.0)),
            squatting_seconds=float(data.get("squattingSeconds", 0.0)),
            bending_seconds=float(data.get("bendingSeconds", 0.0)),
            total_transitions=int(data.get("totalTransitions", 0)),
            longest_sitting_seconds=float(data.get("longestSittingSeconds", 0.0)),
            longest_bending_seconds=float(data.get("longestBendingSeconds", 0.0)),
            status="completed"
        )
        return jsonify({"success": True, "sessionId": session_id, "status": "completed"})
    except Exception as e:
        app.logger.exception("Failed to save session summary.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/event", methods=["POST"])
def session_event():
    """Record a discrete posture transition event."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("sessionId")
    posture = data.get("posture")
    start_time = data.get("startTime")
    end_time = data.get("endTime")
    duration_seconds = float(data.get("durationSeconds", 0.0))
    avg_confidence = float(data.get("avgConfidence", 0.0))

    if not all([session_id, posture, start_time, end_time]):
        return jsonify({"error": "Missing required event fields"}), 400

    try:
        event_id = database.add_posture_event(
            session_id=int(session_id),
            posture=str(posture),
            start_time=str(start_time),
            end_time=str(end_time),
            duration_seconds=duration_seconds,
            avg_confidence=avg_confidence
        )
        return jsonify({"success": True, "eventId": event_id})
    except Exception as e:
        app.logger.exception("Failed to log posture event.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/active", methods=["GET"])
def session_active():
    """Retrieve active session details if present."""
    active = database.get_active_session()
    return jsonify({"active": active is not None, "session": active})


# ==================================================
# 9. ANALYTICS API
# ==================================================

@app.route("/api/analytics/overview", methods=["GET"])
def analytics_overview():
    """Return aggregated posture statistics across all recorded sessions."""
    try:
        summary = database.get_analytics_summary()
        return jsonify({"success": True, "summary": summary})
    except Exception as e:
        app.logger.exception("Failed to compute analytics summary.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/analytics/trends", methods=["GET"])
def analytics_trends():
    """Return daily trend metrics for trend charts."""
    try:
        days = request.args.get("days", default=7, type=int)
        trends = database.get_daily_trends(limit_days=days)
        return jsonify({"success": True, "trends": trends})
    except Exception as e:
        app.logger.exception("Failed to retrieve trends.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/analytics/sessions", methods=["GET"])
def analytics_sessions():
    """Return list of historical monitoring sessions."""
    try:
        limit = request.args.get("limit", default=30, type=int)
        offset = request.args.get("offset", default=0, type=int)
        sessions = database.get_all_sessions(limit=limit, offset=offset)
        return jsonify({"success": True, "sessions": sessions})
    except Exception as e:
        app.logger.exception("Failed to retrieve sessions list.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/analytics/session/<int:session_id>/events", methods=["GET"])
def analytics_session_events(session_id):
    """Return all posture timeline events for a given session."""
    try:
        events = database.get_session_events(session_id)
        return jsonify({"success": True, "events": events})
    except Exception as e:
        app.logger.exception("Failed to retrieve session events.")
        return jsonify({"error": str(e)}), 500


# ==================================================
# 10. USER SETTINGS API
# ==================================================

@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Fetch user settings."""
    try:
        settings = database.get_user_settings()
        return jsonify({"success": True, "settings": settings})
    except Exception as e:
        app.logger.exception("Failed to get settings.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/settings", methods=["POST"])
def save_settings():
    """Save user settings."""
    data = request.get_json(silent=True) or {}
    try:
        updated = database.update_user_settings(data)
        return jsonify({"success": True, "settings": updated})
    except Exception as e:
        app.logger.exception("Failed to update settings.")
        return jsonify({"error": str(e)}), 500


# ==================================================
# 11. REPORTS DOWNLOAD API
# ==================================================

@app.route("/api/reports/session/<int:session_id>/csv", methods=["GET"])
def download_session_csv(session_id):
    """Download CSV report for a specific session."""
    session = database.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    events = database.get_session_events(session_id)
    csv_content = reports.generate_session_csv(session, events)

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=posecoach_session_{session_id}.csv"
        }
    )


@app.route("/api/reports/session/<int:session_id>/pdf", methods=["GET"])
def download_session_pdf(session_id):
    """Download PDF report for a specific session."""
    session = database.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    events = database.get_session_events(session_id)
    pdf_bytes = reports.generate_session_pdf(session, events)

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"posecoach_session_{session_id}.pdf"
    )


@app.route("/api/reports/all/csv", methods=["GET"])
def download_all_csv():
    """Download CSV report of all sessions."""
    sessions = database.get_all_sessions(limit=500)
    csv_content = reports.generate_all_sessions_csv(sessions)

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=posecoach_all_sessions.csv"
        }
    )


# ==================================================
# 12. RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    try:

        print("\n" + "=" * 45)
        print("       POSECOACH AI WEB SERVER")
        print("=" * 45)

        print("Model ready!")
        print("Website: http://127.0.0.1:5000")
        print("Press CTRL+C to stop the server.")

        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            threaded=True
        )

    finally:

        landmarker.close()
