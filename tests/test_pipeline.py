"""
tests/test_pipeline.py
Basic validation tests for the core pipeline stages. Run with:
    python -m pytest tests/
"""

import os
import sys

import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing import remove_noise, power_law_transform, histogram_equalization, flip_image, reduce_contrast
from feature_extraction import detect_edges, detect_corners, detect_corners_harris, extract_feature_vector
from segmentation import kmeans_segmentation, _simple_kmedoids
from detection_tracking import CentroidTracker
from utils import validate_image_path


@pytest.fixture
def sample_image():
    """A synthetic 100x100 test image with a white square on black background."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (75, 75), (255, 255, 255), -1)
    return img


def test_remove_noise_shape_preserved(sample_image):
    result = remove_noise(sample_image)
    assert result.shape == sample_image.shape


def test_power_law_transform_range(sample_image):
    result = power_law_transform(sample_image, gamma=1.5)
    assert result.dtype == np.uint8
    assert result.min() >= 0 and result.max() <= 255


def test_histogram_equalization_runs(sample_image):
    result = histogram_equalization(sample_image)
    assert result.shape == sample_image.shape


def test_detect_edges_finds_square_border(sample_image):
    edges = detect_edges(sample_image)
    assert edges.sum() > 0  # the square border should produce edge pixels


def test_detect_corners_finds_points(sample_image):
    corners = detect_corners(sample_image)
    assert len(corners) > 0  # a square has 4 corners


def test_detect_corners_harris_finds_points(sample_image):
    corners = detect_corners_harris(sample_image)
    assert len(corners) > 0  # Harris should also flag the square's corners


def test_flip_image_horizontal_matches_numpy(sample_image):
    flipped = flip_image(sample_image, mode="horizontal")
    assert np.array_equal(flipped, sample_image[:, ::-1])


def test_reduce_contrast_moves_pixels_toward_gray(sample_image):
    result = reduce_contrast(sample_image, factor=0.0)
    assert np.all(result == 128)  # factor=0 should collapse fully to mid-gray


def test_feature_vector_length_consistent(sample_image):
    vec = extract_feature_vector(sample_image)
    assert vec.shape[0] == 5 + 64  # 5 scalar features + 4x4x4 color histogram


def test_kmeans_segmentation_output_shape(sample_image):
    segmented, labels = kmeans_segmentation(sample_image, k=2)
    assert segmented.shape == sample_image.shape
    assert labels.shape == sample_image.shape[:2]


def test_simple_kmedoids_returns_k_medoids():
    data = np.random.rand(50, 3)
    labels, medoids = _simple_kmedoids(data, k=3)
    assert medoids.shape[0] == 3
    assert len(np.unique(labels)) <= 3


def test_centroid_tracker_assigns_and_maintains_ids():
    tracker = CentroidTracker(max_disappeared=2)
    objects = tracker.update([(10, 10), (50, 50)])
    assert len(objects) == 2

    # small movement should keep the same IDs
    ids_before = set(objects.keys())
    objects = tracker.update([(12, 11), (52, 49)])
    assert set(objects.keys()) == ids_before


def test_validate_image_path_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        validate_image_path("data/does_not_exist.png")
