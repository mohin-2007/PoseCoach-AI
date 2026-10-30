
import cv2
import time
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from pathlib import Path
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pose_features import extract_normalized_features


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "posecoach_rf_model.joblib"

POSE_MODEL_PATH = PROJECT_DIR / "models" / "pose_landmarker_lite.task"


# ==================================================
# 2. CHECK REQUIRED FILES
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError("Trained ML model not found!")

if not POSE_MODEL_PATH.exists():
    raise FileNotFoundError("MediaPipe pose model not found!")


# ==================================================
# 3. LOAD TRAINED MACHINE LEARNING MODEL
# ==================================================

print("Loading PoseCoach AI model...")

model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]

feature_columns = model_bundle["feature_columns"]

visibility_threshold = model_bundle["visibility_threshold"]

if model_bundle.get("feature_version") != "hip_center_torso_scale_v1":
    raise ValueError("Unexpected feature representation!")

if len(feature_columns) != 132:
    raise ValueError("Incorrect number of ML features!")

print("Random Forest model loaded successfully!")


# ==================================================
# 4. MEDIAPIPE CONFIGURATION
# ==================================================

base_options = python.BaseOptions(
    model_asset_path=str(POSE_MODEL_PATH)
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)


# ==================================================
# 5. SKELETON CONNECTIONS
# ==================================================

CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 31), (27, 29),
    (28, 30), (28, 32)
]


def draw_pose(frame, landmarks):

    height, width = frame.shape[:2]

    points = []

    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Draw skeleton connections
    for start, end in CONNECTIONS:

        if (
            landmarks[start].visibility > 0.4
            and landmarks[end].visibility > 0.4
        ):

            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )

    # Draw landmarks
    for index, landmark in enumerate(landmarks):

        if landmark.visibility > 0.4:

            cv2.circle(
                frame,
                points[index],
                4,
                (0, 0, 255),
                -1
            )


# ==================================================
# 6. START WEBCAM
# ==================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Cannot access webcam. Check camera permissions."
    )

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nWebcam started!")
print("Press Q to quit.")

# Smooth predictions across recent frames
probability_history = deque(maxlen=7)

start_time = time.perf_counter()

# ==================================================
# 7. REAL-TIME POSE CLASSIFICATION
# ==================================================

try:

    with vision.PoseLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            height, width = frame.shape[:2]

            # Mirror the camera for natural interaction
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Generate increasing video timestamps
            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

            # Detect pose
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            prediction_text = "No pose detected"
            confidence_text = ""

            # ======================================
            # 8. EXTRACT AND CLASSIFY POSE
            # ======================================

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]

                draw_pose(frame, landmarks)

                average_visibility = sum(
                    float(point.visibility or 0.0)
                    for point in landmarks
                ) / len(landmarks)

                if average_visibility >= visibility_threshold:

                    try:

                        features = extract_normalized_features(
                            landmarks,
                            image_width=width,
                            image_height=height
                        )

                        # Use exactly the training feature columns
                        feature_data = pd.DataFrame(
                            [features],
                            columns=feature_columns
                        )

                        probabilities = model.predict_proba(
                            feature_data
                        )[0]

                        probability_history.append(probabilities)

                        # Smooth predictions over recent frames
                        average_probabilities = np.mean(
                            probability_history,
                            axis=0
                        )

                        best_index = np.argmax(
                            average_probabilities
                        )

                        predicted_class = model.classes_[best_index]

                        confidence = (
                            average_probabilities[best_index] * 100
                        )

                        prediction_text = str(predicted_class)

                        confidence_text = (
                            f"Confidence: {confidence:.1f}%"
                        )

                    except (ValueError, TypeError) as error:

                        probability_history.clear()

                        prediction_text = "Feature extraction error"

                        print("Prediction error:", error)

                else:

                    probability_history.clear()

                    prediction_text = "Move into full view"

            else:

                probability_history.clear()

            # ======================================
            # 9. DISPLAY PREDICTIONS
            # ======================================

            cv2.rectangle(
                frame,
                (10, 10),
                (390, 110),
                (30, 30, 30),
                -1
            )

            cv2.putText(
                frame,
                "POSECOACH AI",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Posture: {prediction_text}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                confidence_text,
                (20, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "PoseCoach AI - Live Posture Detection",
                frame
            )

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


finally:

    camera.release()

    cv2.destroyAllWindows()

    print("\nPoseCoach AI webcam testing stopped.")
