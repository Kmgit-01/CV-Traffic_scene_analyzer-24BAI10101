"""
train_classifier.py
Offline training entry point.

Expects a dataset folder structured as:
    data/train/<class_name>/*.png
(this matches how GTSRB-style datasets are typically organized).

Extracts features for every image, reduces them with PCA, trains
KNN + Naive Bayes, keeps whichever scores higher on the held-out
split, and saves it to models/classifier.pkl for use by main.py.
"""

import argparse
import os

import numpy as np

from preprocessing import preprocess_pipeline
from feature_extraction import extract_feature_vector
from segmentation import reduce_dimensions
from classifier import train_classifiers, save_model
from utils import validate_image_path, ensure_dir, logger


def build_dataset(data_dir):
    features, labels, label_map = [], [], {}
    class_names = sorted(os.listdir(data_dir))

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        label_map[idx] = class_name

        for fname in os.listdir(class_dir):
            path = os.path.join(class_dir, fname)
            try:
                img = validate_image_path(path)
            except (FileNotFoundError, ValueError):
                continue
            processed = preprocess_pipeline(img)
            features.append(extract_feature_vector(processed))
            labels.append(idx)

    return np.array(features), np.array(labels), label_map


def main():
    parser = argparse.ArgumentParser(description="Train the traffic sign classifier")
    parser.add_argument("--data", default="data/train", help="Path to labeled training folder")
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--k", type=int, default=5, help="k for KNN")
    args = parser.parse_args()

    logger.info(f"Building dataset from {args.data} ...")
    X, y, label_map = build_dataset(args.data)
    logger.info(f"Loaded {len(X)} samples across {len(label_map)} classes")

    X_reduced, pca = reduce_dimensions(X, n_components=args.pca_components)
    results = train_classifiers(X_reduced, y, k_neighbors=args.k)

    best_name = max(results, key=lambda name: results[name]["accuracy"])
    best_model = results[best_name]["model"]
    logger.info(f"Selected {best_name} as the final model "
                f"(accuracy={results[best_name]['accuracy']:.4f})")

    ensure_dir("models")
    save_model(best_model, pca, label_map, path="models/classifier.pkl")
    logger.info("Saved trained model to models/classifier.pkl")


if __name__ == "__main__":
    main()
