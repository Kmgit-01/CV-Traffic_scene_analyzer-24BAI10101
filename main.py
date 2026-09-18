"""
main.py
CLI entry point for the Traffic Scene Analyzer.

Usage:
    python main.py classify --img path/to/sign.png
    python main.py track    --video path/to/clip.mp4
    python main.py enhance  --img path/to/image.png --out outputs/enhanced.png
"""

import argparse
import os

import cv2

from preprocessing import preprocess_pipeline, flip_image, reduce_contrast
from feature_extraction import extract_feature_vector, detect_edges, detect_corners, detect_corners_harris
from segmentation import kmeans_segmentation, watershed_segmentation
from classifier import load_model, predict_sign
from detection_tracking import process_video
from utils import validate_image_path, validate_video_path, ensure_dir, logger


def cmd_enhance(args):
    img = validate_image_path(args.img)
    result = preprocess_pipeline(img, gamma=args.gamma)
    ensure_dir(os.path.dirname(args.out) or ".")
    cv2.imwrite(args.out, result)
    logger.info(f"Enhanced image written to {args.out}")


def cmd_represent(args):
    """Image Representation demo: flipping and contrast reduction."""
    img = validate_image_path(args.img)
    result = img
    if args.flip:
        result = flip_image(result, mode=args.flip)
    if args.contrast is not None:
        result = reduce_contrast(result, factor=args.contrast)
    ensure_dir(os.path.dirname(args.out) or ".")
    cv2.imwrite(args.out, result)
    logger.info(f"Represented image (flip={args.flip}, contrast={args.contrast}) written to {args.out}")


def cmd_segment(args):
    img = validate_image_path(args.img)
    processed = preprocess_pipeline(img)
    kmeans_result, _ = kmeans_segmentation(processed, k=args.k)
    watershed_result, _ = watershed_segmentation(processed)

    ensure_dir("outputs")
    cv2.imwrite("outputs/segmentation_kmeans.png", kmeans_result)
    cv2.imwrite("outputs/segmentation_watershed.png", watershed_result)
    logger.info("Segmentation results written to outputs/")


def cmd_classify(args):
    img = validate_image_path(args.img)
    processed = preprocess_pipeline(img)

    edges = detect_edges(processed)
    harris_corners = detect_corners_harris(processed)
    st_corners = detect_corners(processed)
    logger.info(f"Detected {int((edges > 0).sum())} edge pixels, "
                f"{len(harris_corners)} Harris corners, {len(st_corners)} Shi-Tomasi corners")

    features = extract_feature_vector(processed)
    bundle = load_model(args.model)
    label = predict_sign(features, bundle)

    print(f"\nPredicted sign class: {label}\n")
    logger.info(f"Prediction for {args.img}: {label}")


def cmd_track(args):
    cap = validate_video_path(args.video)
    cap.release()  # process_video reopens it; this just validates the file first
    ensure_dir("outputs")
    frame_count, out_path = process_video(args.video, output_path=args.out)
    print(f"\nProcessed {frame_count} frames. Annotated video saved to {out_path}\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Traffic Scene Analyzer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_enhance = sub.add_parser("enhance", help="Run preprocessing/enhancement on an image")
    p_enhance.add_argument("--img", required=True)
    p_enhance.add_argument("--out", default="outputs/enhanced.png")
    p_enhance.add_argument("--gamma", type=float, default=1.2)
    p_enhance.set_defaults(func=cmd_enhance)

    p_represent = sub.add_parser("represent", help="Flip and/or reduce contrast of an image")
    p_represent.add_argument("--img", required=True)
    p_represent.add_argument("--out", default="outputs/represented.png")
    p_represent.add_argument("--flip", choices=["horizontal", "vertical", "both"], default=None)
    p_represent.add_argument("--contrast", type=float, default=None,
                              help="Contrast factor in [0,1]; 1.0 = unchanged, 0.0 = flat gray")
    p_represent.set_defaults(func=cmd_represent)

    p_segment = sub.add_parser("segment", help="Run segmentation on an image")
    p_segment.add_argument("--img", required=True)
    p_segment.add_argument("--k", type=int, default=3)
    p_segment.set_defaults(func=cmd_segment)

    p_classify = sub.add_parser("classify", help="Classify a traffic sign image")
    p_classify.add_argument("--img", required=True)
    p_classify.add_argument("--model", default="models/classifier.pkl")
    p_classify.set_defaults(func=cmd_classify)

    p_track = sub.add_parser("track", help="Detect and track vehicles in a video")
    p_track.add_argument("--video", required=True)
    p_track.add_argument("--out", default="outputs/tracked_output.mp4")
    p_track.set_defaults(func=cmd_track)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
