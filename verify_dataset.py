
import json
from pathlib import Path
from collections import Counter

# Project and dataset paths
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset" / "hvnsh7rwz7-1"

ANNOTATIONS_DIR = DATASET_DIR / "Annotations" / "annotations"
IMAGES_DIR = DATASET_DIR / "JPEGImages" / "JPEGImages"
IMAGESETS_DIR = DATASET_DIR / "ImageSets"

def main():
    print("=" * 55)
    print("       POSECOACH AI - DATASET VERIFICATION")
    print("=" * 55)

    # Check folders
    for folder in [ANNOTATIONS_DIR, IMAGES_DIR, IMAGESETS_DIR]:
        if not folder.exists():
            print(f"\nERROR: Folder not found: {folder}")
            return

    # Collect files
    json_files = list(ANNOTATIONS_DIR.glob("*.json"))
    image_files = list(IMAGES_DIR.glob("*.jpg"))

    print(f"\nTotal JSON annotations : {len(json_files)}")
    print(f"Total JPG images       : {len(image_files)}")

    # Create image filename lookup
    image_names = {file.name for file in image_files}

    label_counts = Counter()
    missing_images = []
    invalid_json = []
    missing_filename = []
    multiple_labels = 0
    no_labels = 0

    # Inspect annotations
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            filename = data.get("filename")

            if not filename:
                missing_filename.append(json_file.name)
                continue

            if filename not in image_names:
                missing_images.append(filename)

            for person in data.get("persons", []):
                actions = person.get("actions", {})

                active_labels = [
                    label for label, value in actions.items()
                    if value == 1
                ]

                if len(active_labels) == 0:
                    no_labels += 1
                elif len(active_labels) > 1:
                    multiple_labels += 1

                for label in active_labels:
                    label_counts[label] += 1

        except (json.JSONDecodeError, OSError, TypeError) as error:
            invalid_json.append((json_file.name, str(error)))

    # Check split files
    print("\nIMAGE SPLIT FILES")
    for split in ["train.txt", "val.txt", "trainval.txt", "test.txt"]:
        split_path = IMAGESETS_DIR / split

        if split_path.exists():
            with open(split_path, "r", encoding="utf-8") as file:
                lines = [line.strip() for line in file if line.strip()]
            print(f"{split:15} : {len(lines)} entries")
        else:
            print(f"{split:15} : NOT FOUND")

    # Display labels
    print("\nACTION LABEL COUNTS")
    for label, count in sorted(label_counts.items()):
        print(f"{label:15} : {count}")

    # Display verification results
    print("\nVERIFICATION RESULTS")
    print(f"Missing images          : {len(missing_images)}")
    print(f"Missing filenames       : {len(missing_filename)}")
    print(f"Invalid JSON files      : {len(invalid_json)}")
    print(f"Persons with no labels  : {no_labels}")
    print(f"Persons with multiple labels: {multiple_labels}")

    print("\n" + "=" * 55)
    print("             VERIFICATION COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()
