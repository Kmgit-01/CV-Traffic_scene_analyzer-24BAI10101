"""
detection_tracking.py
Module 3b: Object Detection & Tracking

Detects moving objects (e.g., vehicles) in a video stream using
background subtraction + contour filtering, then tracks them
frame-to-frame with a lightweight centroid tracker so each object
keeps a stable ID and trajectory across frames.
"""

from collections import OrderedDict

import cv2
import numpy as np

from utils import timed_stage


class CentroidTracker:
    """
    Assigns and maintains a unique ID for each detected object by
    matching new detections to existing tracked centroids that are
    closest in Euclidean distance. Objects not matched for
    `max_disappeared` frames are dropped.
    """

    def __init__(self, max_disappeared=15, max_distance=60):
        self.next_id = 0
        self.objects = OrderedDict()       # id -> centroid
        self.disappeared = OrderedDict()   # id -> frames since last seen
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids):
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for c in centroids:
                self.register(c)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))
        input_centroids = np.array(centroids)

        dists = np.linalg.norm(
            object_centroids[:, None, :] - input_centroids[None, :, :], axis=2
        )

        rows = dists.min(axis=1).argsort()
        cols = dists.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dists[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects


@timed_stage("Vehicle Detection")
def detect_objects(frame, bg_subtractor, min_area=500):
    """Detect moving objects in a frame via background subtraction."""
    mask = bg_subtractor.apply(frame)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, centroids = [], []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
        centroids.append((x + w // 2, y + h // 2))
    return boxes, centroids


def process_video(video_path, output_path="outputs/tracked_output.mp4"):
    """Run detection + tracking over an entire video and save the annotated result."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    tracker = CentroidTracker()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        boxes, centroids = detect_objects(frame, bg_subtractor)
        tracked = tracker.update(centroids)

        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for object_id, centroid in tracked.items():
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {object_id}", (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)

    cap.release()
    writer.release()
    return frame_count, output_path
