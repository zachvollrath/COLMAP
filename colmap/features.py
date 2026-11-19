"""SIFT feature extraction for the correspondence-search stage (Section 2.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from colmap.geometry import calibration_matrix
from colmap.scene_graph import Camera, ImageNode


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def default_focal(width: int, height: int) -> float:
    """Prior focal length used when a photo has no calibration metadata."""
    return 1.2 * max(width, height)


def extract_sift(image_path: str | Path, max_features: int = 8000) -> tuple[np.ndarray, np.ndarray, int, int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Reading images requires opencv. Install the images extra: pip install opencv-python-headless"
        ) from exc
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if not keypoints or descriptors is None:
        return np.zeros((0, 2)), np.zeros((0, 128), dtype=np.float32), width, height
    points = np.array([kp.pt for kp in keypoints], dtype=float)
    return points, descriptors.astype(np.float32), width, height


def load_image_folder(folder: str | Path, focal: float | None = None) -> tuple[dict[int, Camera], dict[int, ImageNode]]:
    paths = sorted(
        path for path in Path(folder).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    cameras: dict[int, Camera] = {}
    images: dict[int, ImageNode] = {}
    for image_id, path in enumerate(paths, start=1):
        keypoints, descriptors, width, height = extract_sift(path)
        focal_px = float(focal) if focal is not None else default_focal(width, height)
        cameras[image_id] = Camera(image_id, width, height, focal_px, 0.0, focal_px)
        images[image_id] = ImageNode(image_id, path.name, image_id, keypoints, descriptors)
    return cameras, images


def camera_matrix(camera: Camera) -> np.ndarray:
    return calibration_matrix(camera.focal, camera.width, camera.height)
