"""
Main CLI Pipeline for Multi-Label Thoracic Disease Classification with DenseNet121+GRU and Score-CAM.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

from config import CLASS_NAMES, THRESHOLD, AUROC_SCORES
from utils import (
    load_and_preprocess_image,
    build_densenet_gru_model,
    generate_scorecam,
    save_visual_report,
    generate_synthetic_chest_xray
)


MODEL_PATH = os.path.join("models", "final_densenet_gru.keras")
DEFAULT_IMAGE_PATH = os.path.join("data", "test_images", "sample_xray.jpg")
OUTPUT_DIR = os.path.join("data", "outputs")


def load_or_initialize_model():
    """
    Loads trained model if present; otherwise builds and initializes DenseNet121+GRU architecture.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"[+] Loading trained DenseNet121+GRU model from: {MODEL_PATH}")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            print(f"[!] Warning: Could not load existing model file ({e}). Rebuilding architecture...")

    print(f"[*] Pre-trained weights file '{MODEL_PATH}' not found.")
    print(f"[*] Instantiating DenseNet121 + GRU hybrid model architecture with ImageNet backbone...")
    model = build_densenet_gru_model(num_classes=len(CLASS_NAMES))
    
    # Save initialized architecture to disk
    try:
        model.save(MODEL_PATH)
        print(f"[+] Initialized model architecture saved to: {MODEL_PATH}")
    except Exception as e:
        print(f"[!] Note: Skipping direct disk save ({e}).")

    return model


def print_diagnosis_table(probabilities):
    """
    Prints a formatted, publication-ready multi-label classification diagnostic table.
    """
    print("\n" + "=" * 80)
    print("        THORACIC MULTI-LABEL CLASSIFICATION DIAGNOSIS REPORT")
    print("=" * 80)
    header = f"{'#':<3} | {'Disease Category':<22} | {'Probability':<12} | {'Threshold':<10} | {'Status':<14} | {'AUROC'}"
    print(header)
    print("-" * 80)

    positive_findings = []

    for idx, name in enumerate(CLASS_NAMES):
        prob = probabilities[idx]
        auroc = AUROC_SCORES.get(name, 0.0)
        is_positive = prob >= THRESHOLD
        
        status_str = "POSITIVE [!]" if is_positive else "NEGATIVE [-]"
        prob_str = f"{prob * 100:>6.2f} %"
        thresh_str = f"{THRESHOLD * 100:.0f} %"
        auroc_str = f"{auroc:.4f}"

        # Color/highlighting cue in text
        marker = ">>" if is_positive else "  "
        print(f"{marker}{idx+1:<2} | {name:<22} | {prob_str:<12} | {thresh_str:<10} | {status_str:<14} | {auroc_str}")

        if is_positive:
            positive_findings.append((idx, name, prob, auroc))

    print("=" * 80)
    print(f"Total Diseases Evaluated: {len(CLASS_NAMES)}")
    print(f"Positive Detections (Prob >= {THRESHOLD*100:.0f}%): {len(positive_findings)}")
    if positive_findings:
        print("Detected Pathologies: " + ", ".join([f"{name} ({prob*100:.1f}%)" for _, name, prob, _ in positive_findings]))
    else:
        print("Detected Pathologies: None (All classes below threshold)")
    print("=" * 80 + "\n")

    return positive_findings


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Label Thoracic Disease Classification with DenseNet121+GRU & Score-CAM"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=DEFAULT_IMAGE_PATH,
        help=f"Path to input chest X-Ray image (default: {DEFAULT_IMAGE_PATH})"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of activation channels to use for Score-CAM computation (default: 20)"
    )
    args = parser.parse_args()

    # Ensure required directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DEFAULT_IMAGE_PATH), exist_ok=True)

    # If default test image doesn't exist, create it automatically
    if not os.path.exists(args.image_path):
        if args.image_path == DEFAULT_IMAGE_PATH:
            print(f"[*] Synthetic test image not found at '{args.image_path}'. Generating synthetic X-ray...")
            generate_synthetic_chest_xray(args.image_path)
            print(f"[+] Synthetic X-ray successfully created at '{args.image_path}'.")
        else:
            print(f"[!] Error: Image not found at specified path: '{args.image_path}'")
            sys.exit(1)

    print(f"[*] Processing Input Chest X-Ray: {args.image_path}")
    img_tensor, img_display = load_and_preprocess_image(args.image_path)

    # Load / initialize model
    model = load_or_initialize_model()

    # Model inference
    print("[*] Running multi-label prediction with DenseNet121+GRU...")
    raw_predictions = model.predict(img_tensor, verbose=0)[0]

    # Display diagnostic table
    positive_findings = print_diagnosis_table(raw_predictions)

    # Determine classes to visualize with Score-CAM
    target_visualizations = []
    if positive_findings:
        target_visualizations = positive_findings
    else:
        # If no finding is above threshold, visualize the highest probability class
        top_idx = int(np.argmax(raw_predictions))
        top_name = CLASS_NAMES[top_idx]
        top_prob = float(raw_predictions[top_idx])
        top_auroc = AUROC_SCORES.get(top_name, 0.0)
        print(f"[*] No classes exceeded threshold {THRESHOLD:.2f}. Explaining highest confidence class: {top_name} ({top_prob*100:.2f}%)")
        target_visualizations = [(top_idx, top_name, top_prob, top_auroc)]

    # Generate Score-CAM visualizations
    print("\n" + "-" * 80)
    print("                 GENERATING SCORE-CAM EXPLAINABILITY MAPS")
    print("-" * 80)
    saved_reports = []

    for class_idx, class_name, prob, auroc in target_visualizations:
        print(f"[*] Computing Score-CAM for class: '{class_name}' (Index: {class_idx}, Prob: {prob*100:.2f}%)...")
        heatmap = generate_scorecam(model, img_tensor, class_idx, top_k=args.top_k)
        
        output_filename = f"scorecam_{class_name.lower().replace(' ', '_')}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        save_visual_report(img_display, heatmap, class_name, prob, output_path, auroc=auroc)
        print(f"[+] Visual Diagnostic Report saved -> {output_path}")
        saved_reports.append(output_path)

    print("-" * 80)
    print(f"[OK] Pipeline complete! Generated {len(saved_reports)} visual diagnostic reports in '{OUTPUT_DIR}'.\n")


if __name__ == "__main__":
    main()
