"""
utils.py
Shared helper functions used across the pipeline: logging setup,
input validation, and small I/O helpers.
"""

import os
import logging
import time
from functools import wraps

import cv2


def setup_logger(log_file="outputs/run.log"):
    """Configure a logger that writes to both console and a log file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("cvpipe")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # avoid duplicate handlers on repeated calls

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logger()


def timed_stage(stage_name):
    """Decorator that logs how long a pipeline stage took to run."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = (time.time() - start) * 1000
            logger.info(f"[{stage_name}] completed in {elapsed:.2f} ms")
            return result
        return wrapper
    return decorator


def validate_image_path(path):
    """Raise a clear error if the image path is missing or unreadable."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"File exists but could not be read as an image: {path}")
    return img


def validate_video_path(path):
    """Raise a clear error if the video path is missing or unreadable."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"File exists but could not be opened as a video: {path}")
    return cap


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
