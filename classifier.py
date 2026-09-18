"""
classifier.py
Module 3a: Classification

Trains and evaluates KNN and Naive Bayes classifiers on the
PCA-reduced feature vectors produced by feature_extraction.py
and segmentation.py, then exposes a simple predict() interface
used by main.py for single-image inference.
"""

import pickle

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from utils import timed_stage, logger


@timed_stage("Train Classifiers")
def train_classifiers(X, y, k_neighbors=5, test_size=0.2):
    """
    Train both a KNN and a Gaussian Naive Bayes classifier on the same
    train/test split and return both models plus their accuracy scores,
    so the two can be compared (per the report's model-selection rationale).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_preds)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_preds = nb.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_preds)

    logger.info(f"KNN accuracy: {knn_acc:.4f}")
    logger.info(f"Naive Bayes accuracy: {nb_acc:.4f}")
    logger.info("\nKNN report:\n" + classification_report(y_test, knn_preds))
    logger.info("\nNaive Bayes report:\n" + classification_report(y_test, nb_preds))

    return {
        "knn": {"model": knn, "accuracy": knn_acc},
        "naive_bayes": {"model": nb, "accuracy": nb_acc},
    }


def save_model(model, pca, label_map, path="models/classifier.pkl"):
    with open(path, "wb") as f:
        pickle.dump({"model": model, "pca": pca, "label_map": label_map}, f)


def load_model(path="models/classifier.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_sign(feature_vector, bundle):
    """Predict a sign class label for a single feature vector."""
    reduced = bundle["pca"].transform(feature_vector.reshape(1, -1))
    pred_idx = bundle["model"].predict(reduced)[0]
    return bundle["label_map"].get(pred_idx, str(pred_idx))
