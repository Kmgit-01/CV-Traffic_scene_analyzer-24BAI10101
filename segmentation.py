"""
segmentation.py
Module 2b: Segmentation & Dimensionality Reduction

Implements watershed segmentation, K-means / K-medoids color
clustering (to isolate the sign region from the background),
and PCA for reducing extracted feature vectors before
classification.
"""

import cv2
import numpy as np
from sklearn.decomposition import PCA

from utils import timed_stage


def _simple_kmedoids(data, k, max_iter=15, random_state=42):
    """
    A lightweight K-Medoids (PAM-style) implementation, avoiding an
    external dependency. Medoids are actual data points, unlike
    K-means centroids which are averages.
    """
    rng = np.random.default_rng(random_state)
    n = data.shape[0]
    medoid_idx = rng.choice(n, k, replace=False)
    medoids = data[medoid_idx]

    for _ in range(max_iter):
        dists = np.linalg.norm(data[:, None, :] - medoids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        new_medoids = medoids.copy()
        for cluster in range(k):
            members = data[labels == cluster]
            if len(members) == 0:
                continue
            # choose the member minimizing total distance to all other members
            intra_dists = np.linalg.norm(members[:, None, :] - members[None, :, :], axis=2)
            new_medoids[cluster] = members[np.argmin(intra_dists.sum(axis=1))]

        if np.allclose(new_medoids, medoids):
            break
        medoids = new_medoids

    dists = np.linalg.norm(data[:, None, :] - medoids[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    return labels, medoids


@timed_stage("Watershed Segmentation")
def watershed_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)
    segmented = image.copy()
    segmented[markers == -1] = [0, 0, 255]
    return segmented, markers


@timed_stage("K-Means Clustering")
def kmeans_segmentation(image, k=3):
    """Cluster pixels by color to separate sign region from background."""
    pixels = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(image.shape)
    return segmented, labels.reshape(image.shape[:2])


@timed_stage("K-Medoids Clustering")
def kmedoids_segmentation(image, k=3, sample_size=2000):
    """
    K-Medoids clustering on a downsampled pixel set (K-Medoids does not
    scale to full-resolution images the way K-means does).
    """
    pixels = image.reshape((-1, 3)).astype(np.float32)
    if len(pixels) > sample_size:
        idx = np.random.choice(len(pixels), sample_size, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    _, medoids = _simple_kmedoids(sample, k)
    dists = np.linalg.norm(pixels[:, None, :] - medoids[None, :, :], axis=2)
    labels_full = np.argmin(dists, axis=1)
    centers = np.uint8(medoids)
    segmented = centers[labels_full].reshape(image.shape)
    return segmented, labels_full.reshape(image.shape[:2])


@timed_stage("PCA Reduction")
def reduce_dimensions(feature_vectors, n_components=10):
    """Reduce a matrix of feature vectors (rows = samples) with PCA."""
    n_components = min(n_components, feature_vectors.shape[0], feature_vectors.shape[1])
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(feature_vectors)
    return reduced, pca
