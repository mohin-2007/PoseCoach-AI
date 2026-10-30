
import json
import cv2
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset" / "hvnsh7rwz7-1"
ANNOTATIONS_DIR = DATASET_DIR / "Annotations" / "annotations"
IMAGES_DIR = DATASET_DIR / "JPEGImages" / "JPEGImages"

FAILED_IDS = [
    "p1_00012",
    "p1_00035",
    "p1_00048",
    "p1_00079",
    "p1_00081"
]

OUTPUT_DIR = PROJECT_DIR / "dataset" / "failed_crops"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for image_id in FAILED_IDS:

    image_path = IMAGES_DIR / f"{image_id}.jpg"
    annotation_path = ANNOTATIONS_DIR / f"{image_id}.json"

    image = cv2.imread(str(image_path))

    if image is None:
        print("Image not found:", image_id)
        continue

    with open(annotation_path, "r") as file:
        annotation = json.load(file)

    height, width = image.shape[:2]

    for index, person in enumerate(annotation.get("persons", [])):

        bbox = person.get("bndbox", {})

        x1 = max(0, int(bbox.get("xmin", 0)))
        y1 = max(0, int(bbox.get("ymin", 0)))
        x2 = min(width, int(bbox.get("xmax", 0)))
        y2 = min(height, int(bbox.get("ymax", 0)))

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        output_path = OUTPUT_DIR / f"{image_id}_{index}.jpg"

        cv2.imwrite(str(output_path), crop)

        print("Saved:", output_path)

print("\nFAILED CROP INSPECTION COMPLETED")
