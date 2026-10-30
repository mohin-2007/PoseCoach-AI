
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_DIR / "dataset" / "sample_features.csv"

MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "posecoach_rf_model.joblib"

MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"

REPORT_PATH = MODEL_DIR / "classification_report.txt"


# ==================================================
# 2. CONFIGURATION
# ==================================================

LABELS = [
    "Standing",
    "Sitting",
    "Squatting",
    "Bending"
]

RANDOM_STATE = 42
N_SPLITS = 5


# ==================================================
# 3. LOAD DATASET
# ==================================================

print("=" * 55)
print("POSECOACH AI - NORMALIZED MODEL TRAINING")
print("=" * 55)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==================================================
# 4. VALIDATE DATASET
# ==================================================

required_columns = ["image_id", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

feature_columns = [
    column for column in df.columns
    if column.startswith("feature_")
]

feature_columns.sort(
    key=lambda column: int(column.split("_")[1])
)

if len(feature_columns) != 132:
    raise ValueError(
        f"Expected 132 features, found {len(feature_columns)}"
    )

X = df[feature_columns].copy()

y = df["label"].copy()

groups = df["image_id"].copy()

if X.isnull().any().any() or y.isnull().any():
    raise ValueError("Dataset contains missing values.")

if not X.apply(
    lambda column: pd.api.types.is_numeric_dtype(column)
).all():
    raise ValueError("All features must be numerical.")

if not set(y.unique()).issubset(set(LABELS)):
    raise ValueError("Unexpected posture label found.")

if not all(
    pd.api.types.is_numeric_dtype(X[column])
    for column in feature_columns
):
    raise ValueError("Invalid feature data types.")

print("\nFeatures:", len(feature_columns))
print("Unique images:", groups.nunique())

print("\nClass distribution:")
print(y.value_counts().reindex(LABELS, fill_value=0))


# ==================================================
# 5. GROUP-BASED TRAINING AND VALIDATION SPLIT
# ==================================================

print("\nPreparing group-based split...")

splitter = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

train_indices, validation_indices = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_indices]
X_validation = X.iloc[validation_indices]

y_train = y.iloc[train_indices]
y_validation = y.iloc[validation_indices]

train_groups = set(groups.iloc[train_indices])
validation_groups = set(groups.iloc[validation_indices])

# Prevent images from appearing in both sets
if train_groups.intersection(validation_groups):
    raise RuntimeError("Data leakage detected!")

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_validation))

print("\nTraining class distribution:")
print(y_train.value_counts().reindex(LABELS, fill_value=0))

print("\nValidation class distribution:")
print(y_validation.value_counts().reindex(LABELS, fill_value=0))


# ==================================================
# 6. CREATE RANDOM FOREST MODEL
# ==================================================

print("\nCreating Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# ==================================================
# 7. TRAIN MODEL
# ==================================================

print("\nTraining started...")

model.fit(X_train, y_train)

print("Training completed!")


# ==================================================
# 8. VALIDATION PREDICTIONS
# ==================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_validation)


# ==================================================
# 9. EVALUATE MODEL
# ==================================================

accuracy = accuracy_score(y_validation, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_validation,
    y_pred
)

report = classification_report(
    y_validation,
    y_pred,
    labels=LABELS,
    zero_division=0
)

cm = confusion_matrix(
    y_validation,
    y_pred,
    labels=LABELS
)

print("\n" + "=" * 55)
print("MODEL EVALUATION RESULTS")
print("=" * 55)

print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

print(
    f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(cm)


# ==================================================
# 10. SAVE CONFUSION MATRIX
# ==================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

matrix_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABELS
)

matrix_display.plot(
    cmap="Blues",
    xticks_rotation=30
)

plt.title("PoseCoach AI - Normalized Features")

plt.tight_layout()

plt.savefig(
    MATRIX_PATH,
    dpi=300
)

plt.close()

print("\nConfusion matrix saved:", MATRIX_PATH)


# ==================================================
# 11. SAVE EVALUATION REPORT
# ==================================================

with open(REPORT_PATH, "w") as file:

    file.write("POSECOACH AI - NORMALIZED MODEL EVALUATION\n")

    file.write("=" * 50 + "\n\n")

    file.write(
        f"Validation Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%\n\n"
    )

    file.write(report)

    file.write("\nConfusion Matrix:\n")

    file.write(str(cm))


# ==================================================
# 12. SAVE MODEL AND METADATA
# ==================================================

model_bundle = {
    "model": model,
    "feature_columns": feature_columns,
    "labels": LABELS,
    "visibility_threshold": 0.50,
    "feature_version": "hip_center_torso_scale_v1"
}

joblib.dump(
    model_bundle,
    MODEL_PATH
)

print("\nTrained model saved:", MODEL_PATH)

print("Evaluation report saved:", REPORT_PATH)


# ==================================================
# 13. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 55)
print("TRAINING COMPLETED")
print("=" * 55)

print("Algorithm: Random Forest")
print("Feature representation: Normalized landmarks")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_validation))
print("Features:", len(feature_columns))

print(f"Validation accuracy: {accuracy * 100:.2f}%")
print(f"Balanced accuracy: {balanced_accuracy * 100:.2f}%")

print("\nPOSECOACH AI MODEL READY FOR TESTING")

print("=" * 55)
