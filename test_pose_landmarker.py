
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "pose_landmarker_lite.task"
IMAGE_PATH = PROJECT_DIR / "models" / "p1_00001_crop.jpg"

# Check files
if not MODEL_PATH.exists():
    raise FileNotFoundError("Pose model not found!")

if not IMAGE_PATH.exists():
    raise FileNotFoundError("Test crop image not found!")

# Configure Pose Landmarker
base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

# Detect pose
with vision.PoseLandmarker.create_from_options(options) as landmarker:

    image = mp.Image.create_from_file(str(IMAGE_PATH))

    result = landmarker.detect(image)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        print("POSE DETECTED SUCCESSFULLY!")
        print("Total landmarks:", len(landmarks))

        print("\nFirst 5 landmarks:")
        for point in landmarks[:5]:
            print(
                "X:", round(point.x, 4),
                "Y:", round(point.y, 4),
                "Z:", round(point.z, 4),
                "Visibility:", round(point.visibility, 4)
            )

    else:
        print("No pose detected in this image.")

print("\nPOSE TEST COMPLETED")
