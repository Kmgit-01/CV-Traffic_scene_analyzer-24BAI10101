# Problem Statement

Manual monitoring of road scenes for two very different tasks —
reading traffic signs correctly and keeping track of vehicles moving
through a scene — is repetitive and error-prone at scale. This
project builds a single, modular computer vision pipeline that
handles both tasks using classical image processing and machine
learning techniques, without relying on deep learning.

## Scope of the Project

The project covers:

- Enhancing and cleaning raw traffic sign images before analysis
- Extracting shape- and texture-based features from signs using
  edge detection, corner detection, and the Hough transform
- Reducing the dimensionality of extracted features with PCA and
  grouping similar regions with K-means / K-medoids clustering and
  watershed segmentation
- Classifying a sign's category using KNN and Naive Bayes,
  comparing the two, and keeping the better performer
- Detecting moving vehicles in a video feed and tracking each one
  with a persistent ID across frames using centroid-based tracking

It does not cover real-time deployment on embedded hardware, deep
learning-based detectors, or large-scale multi-camera tracking —
these are noted as future enhancements in the project report.

## Target Users

- Students and instructors evaluating classical computer vision
  techniques on a concrete, end-to-end use case
- Developers who want a lightweight, dependency-light starting point
  for sign recognition or simple traffic analytics without a GPU or
  deep learning framework

## High-Level Features

1. **Image enhancement module** — noise removal, gamma correction,
   histogram equalization
2. **Feature detection & segmentation module** — Canny edges,
   Shi-Tomasi corners, Hough lines/circles, K-means/K-medoids
   clustering, watershed segmentation, PCA reduction
3. **Classification & tracking module** — KNN/Naive Bayes sign
   classification and centroid-based multi-object vehicle tracking
   in video
4. Command-line interface for every stage, with logging and basic
   input validation throughout
