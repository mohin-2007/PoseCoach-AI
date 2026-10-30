
import json
import random
import cv2
import mediapipe as mp
import pandas as pd

from pathlib import Path
from collections import Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Import our common normalization function
from pose_features import extract_normalized_features


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset" / "hvnsh7rwz7-1"

ANNOTATIONS_DIR = DATASET_DIR / "Annotations" / "annotations"
IMAGES_DIR = DATASET_DIR / "JPEGImages" / "JPEGImages"
IMAGESETS_DIR = DATASET_DIR / "ImageSets"

MODEL_PATH = PROJECT_DIR / "models" / "pose_landmarker_lite.task"

OUTPUT_PATH = PROJECT_DIR / "dataset" / "sample_features.csv"


# ==================================================
# 2. CONFIGURATION
# ==================================================

LABEL_MAP = {
    "stand": "Standing",
    "sit": "Sitting",
    "squat": "Squatting",
    "bendover": "Bending"
}

MAX_IMAGES = 1000

RANDOM_SEED = 42

MIN_AVG_VISIBILITY = 0.50


# ==================================================
# 3. CHECK REQUIRED FILES
# ==================================================

required_paths = [
    MODEL_PATH,
    IMAGESETS_DIR / "train.txt",
    ANNOTATIONS_DIR,
    IMAGES_DIR
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"Required path not found: {path}"
        )


# ==================================================
# 4. LOAD TRAINING IMAGE IDS
# ==================================================

train_file = IMAGESETS_DIR / "train.txt"

with open(train_file, "r") as file:
    image_ids = [
        line.strip()
        for line in file
        if line.strip()
    ]

random.Random(RANDOM_SEED).shuffle(image_ids)

image_ids = image_ids[:MAX_IMAGES]

print("=" * 55)
print("POSECOACH AI - NORMALIZED FEATURE EXTRACTION")
print("=" * 55)

print("Training images available:", len(
    (IMAGESETS_DIR / "train.txt").read_text().splitlines()
))

print("Images selected:", len(image_ids))
print("Random seed:", RANDOM_SEED)
print("Minimum visibility:", MIN_AVG_VISIBILITY)


# ==================================================
# 5. INITIALIZE COUNTERS
# ==================================================

all_rows = []

processed = 0
target_persons = 0

pose_not_detected = 0
low_visibility = 0
invalid_features = 0

missing_files = 0
invalid_images = 0
invalid_bboxes = 0

class_counts = Counter()


# ==================================================
# 6. CREATE MEDIAPIPE POSE LANDMARKER
# ==================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)


# ==================================================
# 7. EXTRACT NORMALIZED FEATURES
# ==================================================

with vision.PoseLandmarker.create_from_options(options) as landmarker:

    for image_number, image_id in enumerate(image_ids, start=1):

        if image_number == 1 or image_number % 100 == 0:
            print(
                f"\nProgress: {image_number}/{len(image_ids)}"
            )

        image_path = IMAGES_DIR / f"{image_id}.jpg"
        annotation_path = ANNOTATIONS_DIR / f"{image_id}.json"

        # ------------------------------------------
        # CHECK FILES
        # ------------------------------------------

        if not image_path.exists() or not annotation_path.exists():
            missing_files += 1
            continue

        # ------------------------------------------
        # READ IMAGE
        # ------------------------------------------

        image = cv2.imread(str(image_path))

        if image is None:
            invalid_images += 1
            continue

        # ------------------------------------------
        # READ ANNOTATION
        # ------------------------------------------

        try:
            with open(annotation_path, "r") as file:
                annotation = json.load(file)

        except (json.JSONDecodeError, OSError):
            missing_files += 1
            continue

        height, width = image.shape[:2]

        # ------------------------------------------
        # PROCESS EACH PERSON
        # ------------------------------------------

        for person_index, person in enumerate(
            annotation.get("persons", [])
        ):

            actions = person.get("actions", {})

            active_labels = [
                label
                for label, value in actions.items()
                if value == 1 and label in LABEL_MAP
            ]

            if len(active_labels) != 1:
                continue

            target_persons += 1

            label = LABEL_MAP[active_labels[0]]

            # --------------------------------------
            # GET BOUNDING BOX
            # --------------------------------------

            bbox = person.get("bndbox", {})

            try:
                x1 = max(0, int(bbox["xmin"]))
                y1 = max(0, int(bbox["ymin"]))
                x2 = min(width, int(bbox["xmax"]))
                y2 = min(height, int(bbox["ymax"]))

            except (KeyError, TypeError, ValueError):
                invalid_bboxes += 1
                continue

            if x2 <= x1 or y2 <= y1:
                invalid_bboxes += 1
                continue

            # --------------------------------------
            # CROP PERSON
            # --------------------------------------

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                invalid_bboxes += 1
                continue

            rgb_crop = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_crop
            )

            # --------------------------------------
            # DETECT POSE
            # --------------------------------------

            result = landmarker.detect(mp_image)

            if not result.pose_landmarks:
                pose_not_detected += 1
                continue

            landmarks = result.pose_landmarks[0]

            # --------------------------------------
            # CHECK VISIBILITY
            # --------------------------------------

            visibility_values = [
                float(point.visibility or 0.0)
                for point in landmarks
            ]

            average_visibility = (
                sum(visibility_values) / len(visibility_values)
            )

            if average_visibility < MIN_AVG_VISIBILITY:
                low_visibility += 1
                continue

            # --------------------------------------
            # NORMALIZE LANDMARK FEATURES
            # --------------------------------------

            try:
                features = extract_normalized_features(
                    landmarks,
                    image_width=crop.shape[1],
                    image_height=crop.shape[0]
                )

            except ValueError as error:
                print(
                    f"Invalid features: {image_id} | {error}"
                )
                invalid_features += 1
                continue

            if len(features) != 132:
                invalid_features += 1
                continue

            # --------------------------------------
            # CREATE DATASET ROW
            # --------------------------------------

            row = {
                "image_id": image_id,
                "person_index": person_index,
                "label": label,
                "average_visibility": average_visibility
            }

            for index, value in enumerate(features):
                row[f"feature_{index}"] = value

            all_rows.append(row)

            processed += 1
            class_counts[label] += 1


# ==================================================
# 8. SAVE FEATURE DATASET
# ==================================================

print("\n" + "=" * 55)
print("FEATURE EXTRACTION SUMMARY")
print("=" * 55)

print("Images selected:", len(image_ids))
print("Target persons found:", target_persons)
print("Accepted samples:", processed)
print("Pose not detected:", pose_not_detected)
print("Low visibility:", low_visibility)
print("Invalid features:", invalid_features)
print("Invalid bounding boxes:", invalid_bboxes)
print("Missing files:", missing_files)
print("Unreadable images:", invalid_images)


# ==================================================
# 9. EXPORT CSV
# ==================================================

if all_rows:

    df = pd.DataFrame(all_rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nNORMALIZED FEATURE EXTRACTION SUCCESSFUL!")

    print("Dataset shape:", df.shape)

    print("\nClass distribution:")

    print(
        df["label"].value_counts().reindex(
            ["Standing", "Sitting", "Squatting", "Bending"],
            fill_value=0
        ).to_string()
    )

    print("\nAverage visibility:")
    print(round(df["average_visibility"].mean(), 3))

    print("\nCSV saved at:")
    print(OUTPUT_PATH)

else:
    print("\nNO FEATURES EXTRACTED.")
    print("Please check the dataset and MediaPipe model.")


print("\n" + "=" * 55)
print("TEST COMPLETED")
print("=" * 55)
