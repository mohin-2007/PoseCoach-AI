
import json
from pathlib import Path

import cv2
import mediapipe as mp

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset" / "hvnsh7rwz7-1"

ANNOTATIONS_DIR = DATASET_DIR / "Annotations" / "annotations"
IMAGES_DIR = DATASET_DIR / "JPEGImages" / "JPEGImages"
OUTPUT_DIR = PROJECT_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 55)
    print("       POSECOACH AI - PREPROCESSING TEST")
    print("=" * 55)

    # Check MediaPipe
    print("\nMEDIAPIPE INFORMATION")
    print("Version:", mp.__version__)
    print("Legacy solutions available:", hasattr(mp, "solutions"))

    # Test sample images
    sample_ids = [
        "p1_00001",
        "p1_00002",
        "p1_00005",
        "p1_00006",
        "p1_00008"
    ]

    print("\nIMAGE AND BOUNDING BOX TEST")

    for image_id in sample_ids:
        image_path = IMAGES_DIR / f"{image_id}.jpg"
        annotation_path = ANNOTATIONS_DIR / f"{image_id}.json"

        if not image_path.exists() or not annotation_path.exists():
            print(f"{image_id}: FILE MISSING")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"{image_id}: IMAGE COULD NOT BE READ")
            continue

        with open(annotation_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        print(f"\nImage: {image_id}.jpg")
        print("Image dimensions:", image.shape)

        persons = data.get("persons", [])
        print("Persons annotated:", len(persons))

        if not persons:
            print("No person annotation found.")
            continue

        # Test the first person's bounding box
        bbox = persons[0].get("bndbox", {})

        xmin = int(bbox.get("xmin", 0))
        ymin = int(bbox.get("ymin", 0))
        xmax = int(bbox.get("xmax", 0))
        ymax = int(bbox.get("ymax", 0))

        height, width = image.shape[:2]

        # Keep coordinates within image boundaries
        xmin = max(0, min(xmin, width))
        xmax = max(0, min(xmax, width))
        ymin = max(0, min(ymin, height))
        ymax = max(0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            print("Invalid bounding box.")
            continue

        cropped_person = image[ymin:ymax, xmin:xmax]

        if cropped_person.size == 0:
            print("Empty person crop.")
            continue

        # Save one sample crop for visual inspection
        output_path = OUTPUT_DIR / f"{image_id}_crop.jpg"
        cv2.imwrite(str(output_path), cropped_person)

        actions = persons[0].get("actions", {})
        active_labels = [
            label for label, value in actions.items()
            if value == 1
        ]

        print("Bounding box:", (xmin, ymin, xmax, ymax))
        print("Active label:", active_labels)
        print("Crop saved:", output_path.name)

    print("\nPREPROCESSING TEST COMPLETED")


if __name__ == "__main__":
    main()
