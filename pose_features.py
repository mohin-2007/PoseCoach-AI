
import math


def extract_normalized_features(landmarks, image_width, image_height):
    """
    Convert 33 MediaPipe pose landmarks into
    132 body-relative numerical features.

    Each landmark contains:
    X, Y, Z and visibility.

    Total: 33 * 4 = 132 features.
    """

    if len(landmarks) != 33:
        raise ValueError(
            f"Expected 33 landmarks, received {len(landmarks)}"
        )

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Invalid image dimensions")

    # Convert normalized coordinates into pixel coordinates
    x_coords = [
        float(point.x) * image_width
        for point in landmarks
    ]

    y_coords = [
        float(point.y) * image_height
        for point in landmarks
    ]

    z_coords = [
        float(point.z) * image_width
        for point in landmarks
    ]

    # MediaPipe landmark indices:
    # 11 = Left shoulder
    # 12 = Right shoulder
    # 23 = Left hip
    # 24 = Right hip

    # Calculate shoulder centre
    shoulder_x = (x_coords[11] + x_coords[12]) / 2
    shoulder_y = (y_coords[11] + y_coords[12]) / 2

    # Calculate hip centre
    hip_x = (x_coords[23] + x_coords[24]) / 2
    hip_y = (y_coords[23] + y_coords[24]) / 2
    hip_z = (z_coords[23] + z_coords[24]) / 2

    # Calculate torso length for scale normalization
    torso_scale = math.hypot(
        shoulder_x - hip_x,
        shoulder_y - hip_y
    )

    if not math.isfinite(torso_scale) or torso_scale < 1e-6:
        raise ValueError("Invalid torso scale")

    features = []

    # Normalize all 33 landmarks
    for index, point in enumerate(landmarks):

        normalized_x = (
            x_coords[index] - hip_x
        ) / torso_scale

        normalized_y = (
            y_coords[index] - hip_y
        ) / torso_scale

        normalized_z = (
            z_coords[index] - hip_z
        ) / torso_scale

        visibility = float(point.visibility or 0.0)

        values = [
            normalized_x,
            normalized_y,
            normalized_z,
            visibility
        ]

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite landmark feature")

        features.extend(values)

    if len(features) != 132:
        raise ValueError("Feature extraction failed")

    return features
