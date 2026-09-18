"""
feature_extraction.py
Module 2a: Feature Detection

Implements edge detection (Canny), corner detection (both Harris
and Shi-Tomasi, matching the two corner-detection sessions in the
syllabus), and the Hough Line Transform. The Hough Circle Transform
is also included as a self-directed extension beyond the taught
syllabus (only Hough Line Transform was covered in the course) —
see the project report, section 7.5, for the rationale.
These features are later combined into a feature vector for
classification and used to help identify sign shape.
"""

import cv2
import numpy as np

from utils import timed_stage


@timed_stage("Canny Edge Detection")
def detect_edges(image, low_thresh=50, high_thresh=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Canny(gray, low_thresh, high_thresh)


@timed_stage("Harris Corner Detection")
def detect_corners_harris(image, block_size=2, ksize=3, k=0.04, thresh_ratio=0.01):
    """Classic Harris corner detector. Returns (row, col) coordinates of corners."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = np.float32(gray)
    response = cv2.cornerHarris(gray, block_size, ksize, k)
    response = cv2.dilate(response, None)
    ys, xs = np.where(response > thresh_ratio * response.max())
    return np.column_stack([xs, ys])


@timed_stage("Shi-Tomasi Corner Detection")
def detect_corners(image, max_corners=50, quality=0.01, min_distance=10):
    """Shi-Tomasi corner detection (goodFeaturesToTrack) — an improvement on Harris."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality, min_distance)
    return corners.reshape(-1, 2) if corners is not None else np.empty((0, 2))


@timed_stage("Hough Line Transform")
def detect_lines(edge_image, threshold=80):
    lines = cv2.HoughLinesP(
        edge_image, 1, np.pi / 180, threshold, minLineLength=20, maxLineGap=5
    )
    return lines if lines is not None else np.empty((0, 1, 4))


@timed_stage("Hough Circle Transform")
def detect_circles(image, dp=1.2, min_dist=30):
    """
    NOTE: this is a self-directed extension, not part of the taught
    syllabus (only the Hough Line Transform was covered). Included
    because sign shape (circular vs. triangular) is a useful feature,
    but kept as a separate, optional function rather than assumed
    course content.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=100, param2=30, minRadius=10, maxRadius=0,
    )
    return circles[0] if circles is not None else np.empty((0, 3))


def extract_feature_vector(image, include_circles=True):
    """
    Build a compact numeric feature vector for a sign image: edge
    density, Harris corner count, Shi-Tomasi corner count, line
    count, (optionally) circle count, and a coarse color histogram.
    Used as input to PCA + classifiers.
    """
    edges = detect_edges(image)
    harris_corners = detect_corners_harris(image)
    st_corners = detect_corners(image)
    lines = detect_lines(edges)
    circle_count = len(detect_circles(image)) if include_circles else 0

    edge_density = np.sum(edges > 0) / edges.size
    harris_count = len(harris_corners)
    st_count = len(st_corners)
    line_count = len(lines)

    hist = cv2.calcHist([image], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3)
    hist = cv2.normalize(hist, hist).flatten()

    return np.concatenate(
        [[edge_density, harris_count, st_count, line_count, circle_count], hist]
    )
