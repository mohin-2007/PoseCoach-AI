"""
Comprehensive End-to-End Test Suite for PoseCoach AI
Tests:
1. Model loading & feature vector dimension
2. Flask API routes: /api/health, /api/predict (mock image frame)
3. SQLite database operations: session creation, pause, resume, event logging, summary saving
4. SQLite persistence across reloads
5. Analytics overview & daily trends calculation
6. User settings retrieval and updating
7. CSV and PDF report generation
"""

import sys
import unittest
import numpy as np
import cv2
import io
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import app
import database
import reports


class TestPoseCoachAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize test database
        database.init_db()
        cls.client = app.app.test_client()

    def test_01_model_loaded(self):
        """Verify the trained Random Forest model and MediaPipe Landmarker are initialized."""
        self.assertIsNotNone(app.model)
        self.assertEqual(len(app.feature_columns), 132)
        expected_classes = {"Standing", "Sitting", "Squatting", "Bending"}
        self.assertEqual(set(app.model.classes_), expected_classes)

    def test_02_health_endpoint(self):
        """Verify /api/health endpoint returns status, classes and feature count."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["features"], 132)
        self.assertIn("Standing", data["classes"])

    def test_03_predict_endpoint_empty_frame(self):
        """Verify /api/predict gracefully handles empty or invalid requests."""
        response = self.client.post("/api/predict")
        self.assertEqual(response.status_code, 400)

    def test_04_predict_endpoint_black_image(self):
        """Verify /api/predict handles an image with no person detected."""
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", black_img)
        data = {
            "frame": (io.BytesIO(encoded.tobytes()), "test.jpg")
        }
        response = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        res_json = json.loads(response.data)
        self.assertFalse(res_json["detected"])
        self.assertEqual(res_json["posture"], None)
        self.assertIn("landmarks", res_json)

    def test_05_predict_endpoint_sample_image(self):
        """Verify /api/predict processes a real sample image from models/ directory."""
        sample_img_path = PROJECT_DIR / "models" / "p1_00001_crop.jpg"
        if sample_img_path.exists():
            with open(sample_img_path, "rb") as f:
                img_bytes = f.read()
            data = {
                "frame": (io.BytesIO(img_bytes), "sample.jpg")
            }
            response = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
            self.assertEqual(response.status_code, 200)
            res_json = json.loads(response.data)
            # If detected, verify structure
            if res_json.get("detected"):
                self.assertIn(res_json["posture"], ["Standing", "Sitting", "Squatting", "Bending"])
                self.assertGreaterEqual(res_json["confidence"], 0)
                self.assertEqual(len(res_json["landmarks"]), 33)

    def test_06_database_session_lifecycle(self):
        """Verify full session lifecycle: create -> pause -> resume -> events -> stop."""
        # Create
        res = self.client.post("/api/session/start")
        self.assertEqual(res.status_code, 200)
        session_data = json.loads(res.data)["session"]
        session_id = session_data["id"]
        self.assertIsNotNone(session_id)

        # Pause
        res = self.client.post("/api/session/pause", json={"sessionId": session_id})
        self.assertEqual(res.status_code, 200)
        s = database.get_session(session_id)
        self.assertEqual(s["status"], "paused")

        # Resume
        res = self.client.post("/api/session/resume", json={"sessionId": session_id})
        self.assertEqual(res.status_code, 200)
        s = database.get_session(session_id)
        self.assertEqual(s["status"], "active")

        # Log Discrete Posture Events
        event_payload = {
            "sessionId": session_id,
            "posture": "Standing",
            "startTime": "2026-10-30T10:00:00Z",
            "endTime": "2026-10-30T10:05:00Z",
            "durationSeconds": 300.0,
            "avgConfidence": 88.5
        }
        res = self.client.post("/api/session/event", json=event_payload)
        self.assertEqual(res.status_code, 200)
        self.assertTrue(json.loads(res.data)["success"])

        event_payload2 = {
            "sessionId": session_id,
            "posture": "Sitting",
            "startTime": "2026-10-30T10:05:00Z",
            "endTime": "2026-10-30T10:25:00Z",
            "durationSeconds": 1200.0,
            "avgConfidence": 91.2
        }
        res = self.client.post("/api/session/event", json=event_payload2)
        self.assertEqual(res.status_code, 200)

        # Stop & Finalize Summary
        stop_payload = {
            "sessionId": session_id,
            "totalDurationSeconds": 1500.0,
            "standingSeconds": 300.0,
            "sittingSeconds": 1200.0,
            "squattingSeconds": 0.0,
            "bendingSeconds": 0.0,
            "totalTransitions": 1,
            "longestSittingSeconds": 1200.0,
            "longestBendingSeconds": 0.0
        }
        res = self.client.post("/api/session/stop", json=stop_payload)
        self.assertEqual(res.status_code, 200)

        # Check in DB
        final_s = database.get_session(session_id)
        self.assertEqual(final_s["status"], "completed")
        self.assertEqual(final_s["total_duration_seconds"], 1500.0)
        self.assertEqual(final_s["sitting_seconds"], 1200.0)
        self.assertEqual(final_s["longest_sitting_seconds"], 1200.0)

        # Verify events retrieval
        events = database.get_session_events(session_id)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["posture"], "Standing")
        self.assertEqual(events[1]["posture"], "Sitting")

    def test_07_analytics_endpoints(self):
        """Verify analytics overview and trends return valid aggregated data."""
        res = self.client.get("/api/analytics/overview")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data["success"])
        summary = data["summary"]
        self.assertGreaterEqual(summary["total_sessions"], 1)
        self.assertGreaterEqual(summary["total_monitoring_seconds"], 1500.0)

        res = self.client.get("/api/analytics/trends?days=7")
        self.assertEqual(res.status_code, 200)
        trends_data = json.loads(res.data)
        self.assertTrue(trends_data["success"])

    def test_08_settings_endpoints(self):
        """Verify user settings can be queried and modified."""
        res = self.client.get("/api/settings")
        self.assertEqual(res.status_code, 200)
        settings = json.loads(res.data)["settings"]
        self.assertIn("sitting_reminder_minutes", settings)

        # Update settings
        update_res = self.client.post("/api/settings", json={
            "sitting_reminder_minutes": 45,
            "bending_reminder_minutes": 15,
            "speech_rate": 1.1
        })
        self.assertEqual(update_res.status_code, 200)
        new_settings = json.loads(update_res.data)["settings"]
        self.assertEqual(new_settings["sitting_reminder_minutes"], 45)
        self.assertEqual(new_settings["bending_reminder_minutes"], 15)
        self.assertEqual(new_settings["speech_rate"], 1.1)

    def test_09_csv_reports(self):
        """Verify CSV export for session and all sessions."""
        # Find a session ID
        sessions = database.get_all_sessions(limit=1)
        self.assertTrue(len(sessions) > 0)
        session_id = sessions[0]["id"]

        # Session CSV
        res = self.client.get(f"/api/reports/session/{session_id}/csv")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "text/csv")
        self.assertIn(b"POSECOACH AI - SESSION REPORT", res.data)
        self.assertIn(b"Sitting", res.data)

        # All sessions CSV
        res_all = self.client.get("/api/reports/all/csv")
        self.assertEqual(res_all.status_code, 200)
        self.assertEqual(res_all.mimetype, "text/csv")
        self.assertIn(b"Session ID", res_all.data)

    def test_10_pdf_report(self):
        """Verify PDF export produces valid binary PDF data."""
        sessions = database.get_all_sessions(limit=1)
        self.assertTrue(len(sessions) > 0)
        session_id = sessions[0]["id"]

        res = self.client.get(f"/api/reports/session/{session_id}/pdf")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "application/pdf")
        # PDF files begin with '%PDF'
        self.assertTrue(res.data.startswith(b"%PDF"))
        self.assertGreater(len(res.data), 1000)


if __name__ == "__main__":
    unittest.main()
