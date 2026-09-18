"""
preprocessing.py
Module 1: Preprocessing & Enhancement

Implements classical enhancement operations used before feature
extraction: noise removal, intensity transforms (log / power-law),
morphological operations, and histogram equalization.
"""

import cv2
import numpy as np

from utils import timed_stage


@timed_stage("Noise Removal")
def remove_noise(image, method="gaussian", ksize=5):
    """Denoise an image using a Gaussian or median filter."""
    if method == "gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif method == "median":
        return cv2.medianBlur(image, ksize)
    else:
        raise ValueError(f"Unknown denoising method: {method}")


@timed_stage("Log Transform")
def log_transform(image):
    """Apply a log transform to compress the dynamic range."""
    img = image.astype(np.float32)
    c = 255 / np.log(1 + np.max(img))
    log_img = c * np.log(1 + img)
    return np.uint8(np.clip(log_img, 0, 255))


@timed_stage("Power-Law Transform")
def power_law_transform(image, gamma=1.5):
    """Apply gamma (power-law) correction."""
    normalized = image / 255.0
    corrected = np.power(normalized, gamma)
    return np.uint8(corrected * 255)


@timed_stage("Morphological Operation")
def morphological_op(binary_image, op="opening", ksize=5, iterations=1):
    """Apply a morphological operation to a binary/grayscale image."""
    kernel = np.ones((ksize, ksize), np.uint8)
    ops = {
        "erosion": cv2.MORPH_ERODE,
        "dilation": cv2.MORPH_DILATE,
        "opening": cv2.MORPH_OPEN,
        "closing": cv2.MORPH_CLOSE,
    }
    if op not in ops:
        raise ValueError(f"Unknown morphological operation: {op}")
    return cv2.morphologyEx(binary_image, ops[op], kernel, iterations=iterations)


@timed_stage("Histogram Equalization")
def histogram_equalization(image):
    """Equalize the histogram of a grayscale or color image (via the Y channel)."""
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


@timed_stage("Image Flip")
def flip_image(image, mode="horizontal"):
    """
    Flip an image, matching the Image Representation session.
    mode: 'horizontal', 'vertical', or 'both'.
    """
    flip_codes = {"horizontal": 1, "vertical": 0, "both": -1}
    if mode not in flip_codes:
        raise ValueError(f"Unknown flip mode: {mode}")
    return cv2.flip(image, flip_codes[mode])


@timed_stage("Contrast Reduction")
def reduce_contrast(image, factor=0.5):
    """
    Reduce contrast by blending the image toward mid-gray (128).
    factor=1.0 keeps the original image; factor=0.0 collapses to flat gray.
    Matches the Image Representation session's contrast-reduction topic.
    """
    factor = float(np.clip(factor, 0.0, 1.0))
    gray_fill = np.full_like(image, 128)
    blended = cv2.addWeighted(image, factor, gray_fill, 1 - factor, 0)
    return blended


def preprocess_pipeline(image, gamma=1.2, denoise_method="gaussian"):
    """Run the full preprocessing chain and return the cleaned image."""
    denoised = remove_noise(image, method=denoise_method)
    enhanced = power_law_transform(denoised, gamma=gamma)
    equalized = histogram_equalization(enhanced)
    return equalized
