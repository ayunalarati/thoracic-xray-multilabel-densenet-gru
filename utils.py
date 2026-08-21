"""
Utility functions for Multi-Label Thoracic Disease Classification with DenseNet121+GRU and Score-CAM.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from config import CLASS_NAMES, AUROC_SCORES, THRESHOLD


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Load an image from file, convert to RGB, resize, and normalize to [0, 1].

    Returns:
        img_tensor (np.ndarray): Tensor of shape (1, 224, 224, 3), normalized [0, 1].
        img_display (np.ndarray): Image array of shape (224, 224, 3), float32 [0, 1].
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at path: {img_path}")

    # Read image using OpenCV
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image file: {img_path}")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize to target size
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    img_display = img_resized.astype(np.float32) / 255.0

    # Expand batch dimension
    img_tensor = np.expand_dims(img_display, axis=0)

    return img_tensor, img_display


def build_densenet_gru_model(num_classes=15, input_shape=(224, 224, 3)):
    """
    Constructs the DenseNet121 + GRU hybrid model architecture:
    - Backbone: DenseNet121 (pre-trained on ImageNet)
    - Sequential/Recurrent Head: Reshape 7x7 spatial grid -> (49, 1024) -> GRU -> Dense Sigmoid
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # DenseNet121 Feature Extractor
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    # Target feature maps from last conv/relu: shape (batch, 7, 7, 1024)
    features = base_model.output

    # Reshape spatial features into sequence format: (batch, 49, 1024)
    feature_dim = features.shape[-1]  # 1024
    grid_dim = features.shape[1] * features.shape[2]  # 7 * 7 = 49
    x = layers.Reshape((grid_dim, feature_dim), name="spatial_to_sequence")(features)

    # Recurrent Temporal / Spatial Sequence Modeling with GRU
    x = layers.GRU(256, return_sequences=False, name="gru_reasoning")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_intermediate")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)

    # Multi-label classification head (Sigmoid activations for independent label probabilities)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="multi_label_predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="DenseNet121_GRU_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )
    return model


def find_target_conv_layer(model):
    """
    Finds the last Conv2D or ReLU feature layer in DenseNet121 for CAM extraction.
    """
    # Priority search for DenseNet121 terminal conv layers
    preferred_layers = ['conv5_block16_2_conv', 'conv5_block16_2_relu', 'relu']
    for name in preferred_layers:
        for layer in model.layers:
            if layer.name == name:
                return layer

    # Fallback search for any layer that outputs 4D tensor before reshaping
    for layer in reversed(model.layers):
        if isinstance(layer, (layers.Conv2D, layers.Activation)) or 'conv' in layer.name.lower() or 'relu' in layer.name.lower():
            if len(layer.output_shape) == 4:
                return layer

    # Secondary fallback to base model if nested
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, layers.Conv2D) or 'conv' in sub_layer.name.lower():
                    return sub_layer

    raise ValueError("Could not automatically locate a target 4D Conv layer in the model.")


def generate_scorecam(model, img_tensor, class_idx, top_k=20):
    """
    Gradient-free Score-CAM implementation (Wang et al., 2020).
    
    Args:
        model: Trained Keras multi-label classification model.
        img_tensor (np.ndarray): Preprocessed image tensor of shape (1, 224, 224, 3).
        class_idx (int): Index of the disease class to explain.
        top_k (int): Number of top activation maps to evaluate for efficiency.

    Returns:
        heatmap (np.ndarray): Normalized 2D Score-CAM heatmap of shape (224, 224) in [0, 1].
    """
    target_layer = find_target_conv_layer(model)
    
    # Sub-model extracting activation maps from target conv layer
    activation_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=target_layer.output
    )

    # 1. Forward pass to extract feature activation maps
    activations = activation_model.predict(img_tensor, verbose=0)[0]  # (H, W, C) e.g., (7, 7, 1024)
    h, w, c = activations.shape
    input_h, input_w = img_tensor.shape[1], img_tensor.shape[2]

    # 2. Select top_k activation channels by activation variance / variance energy
    channel_energies = np.array([np.var(activations[:, :, i]) for i in range(c)])
    top_indices = np.argsort(channel_energies)[::-1][:min(top_k, c)]

    weights = []
    upsampled_masks = []

    # 3. Upsample and normalize each selected activation channel
    for idx in top_indices:
        act_map = activations[:, :, idx]
        
        # Upsample to input resolution (224, 224)
        act_upsampled = cv2.resize(act_map, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        
        # Min-Max Normalization to [0, 1]
        act_min, act_max = np.min(act_upsampled), np.max(act_upsampled)
        if act_max - act_min > 1e-8:
            norm_mask = (act_upsampled - act_min) / (act_max - act_min)
        else:
            norm_mask = np.zeros_like(act_upsampled)

        upsampled_masks.append(norm_mask)

        # 4. Mask input image with the normalized activation map
        masked_img = img_tensor * np.expand_dims(np.expand_dims(norm_mask, axis=0), axis=-1)

        # 5. Forward pass on masked image to compute target class score
        score_preds = model.predict(masked_img, verbose=0)
        target_score = float(score_preds[0, class_idx])
        weights.append(target_score)

    # 6. Linear combination of activation maps weighted by forward prediction score
    weights = np.array(weights)
    upsampled_masks = np.array(upsampled_masks)

    # Apply softmax on weights to stabilize contribution
    if np.sum(weights) > 0:
        exp_weights = np.exp(weights - np.max(weights))
        norm_weights = exp_weights / np.sum(exp_weights)
    else:
        norm_weights = np.ones_like(weights) / len(weights)

    score_cam = np.zeros((input_h, input_w), dtype=np.float32)
    for i in range(len(top_indices)):
        score_cam += norm_weights[i] * upsampled_masks[i]

    # 7. Apply ReLU (keep positive influences only)
    score_cam = np.maximum(score_cam, 0)

    # 8. Final Min-Max normalization to [0, 1]
    cam_min, cam_max = np.min(score_cam), np.max(score_cam)
    if cam_max - cam_min > 1e-8:
        score_cam = (score_cam - cam_min) / (cam_max - cam_min)
    else:
        score_cam = np.zeros_like(score_cam)

    return score_cam


def save_visual_report(img_np, heatmap, disease_name, prob, output_path, auroc=None):
    """
    Generates and saves a publication-quality 3-panel visualization:
    1. Original Chest X-Ray
    2. Score-CAM Heatmap
    3. Blended Overlay
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert heatmap to uint8 colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet_colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_colormap, cv2.COLOR_BGR2RGB) / 255.0

    # Ensure img_np is float [0, 1]
    if img_np.max() > 1.0:
        img_np = img_np / 255.0

    # Alpha blending: Overlay = 0.55 * original + 0.45 * heatmap
    overlay = 0.55 * img_np + 0.45 * jet_rgb
    overlay = np.clip(overlay, 0.0, 1.0)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    fig.patch.set_facecolor('#0f172a')  # Modern dark background

    # 1. Original X-Ray
    axes[0].imshow(img_np)
    axes[0].set_title("Input Chest X-Ray", fontsize=12, fontweight='bold', color='white', pad=10)
    axes[0].axis('off')

    # 2. Score-CAM Heatmap
    im1 = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title(f"Score-CAM Activation\n({disease_name})", fontsize=12, fontweight='bold', color='white', pad=10)
    axes[1].axis('off')
    cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # 3. Superimposed Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f"Diagnostic Overlay\nProb: {prob*100:.1f}% (Threshold: {THRESHOLD*100:.0f}%)", 
                      fontsize=12, fontweight='bold', color='#38bdf8', pad=10)
    axes[2].axis('off')

    # Main Super Title
    auroc_str = f" | Baseline AUROC: {auroc:.4f}" if auroc is not None else ""
    fig.suptitle(
        f"Thoracic AI Diagnostic Report: {disease_name.upper()}{auroc_str}",
        fontsize=14, fontweight='bold', color='#f8fafc', y=0.98
    )

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def generate_synthetic_chest_xray(output_path):
    """
    Creates an anatomically textured synthetic chest X-Ray image for testing.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_size = 512
    xray = np.zeros((img_size, img_size), dtype=np.float32)

    # 1. Background soft gradient
    y, x = np.mgrid[0:img_size, 0:img_size]
    center_y, center_x = img_size // 2, img_size // 2
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    xray += 30 + 20 * np.sin(y / 50.0)

    # 2. Thoracic cage / Outer body contour
    body_mask = ((x - center_x)**2 / (200**2) + (y - center_y)**2 / (230**2)) <= 1.0
    xray[body_mask] += 60.0

    # 3. Bilateral Lung Fields (darker, air-filled regions)
    # Left Lung
    left_lung_mask = (((x - 170)**2 / (65**2) + (y - 250)**2 / (130**2)) <= 1.0)
    xray[left_lung_mask] = np.clip(xray[left_lung_mask] - 50.0, 15, 255)
    
    # Right Lung
    right_lung_mask = (((x - 340)**2 / (65**2) + (y - 250)**2 / (130**2)) <= 1.0)
    xray[right_lung_mask] = np.clip(xray[right_lung_mask] - 50.0, 15, 255)

    # 4. Mediastinum & Cardiac Silhouette (white/dense heart in left lower quadrant)
    heart_mask = (((x - 220)**2 / (55**2) + (y - 300)**2 / (55**2)) <= 1.0)
    xray[heart_mask] += 90.0

    # 5. Clavicles and Ribs (horizontal elliptical density lines)
    for rib_y in range(160, 420, 35):
        rib_line = np.exp(-((y - rib_y - 0.05 * (x - center_x)**2 / 10)**2) / 12.0)
        xray += rib_line * 25.0

    # Clavicles (top arches)
    left_clavicle = np.exp(-((y - 130 - 0.15 * (x - 180))**2) / 8.0) * (x < 250) * (x > 100)
    right_clavicle = np.exp(-((y - 130 + 0.15 * (x - 330))**2) / 8.0) * (x > 260) * (x < 410)
    xray += (left_clavicle + right_clavicle) * 45.0

    # 6. Simulate a focal opacity / consolidation infiltrate in the right lower lung
    infiltrate = np.exp(-(((x - 340)**2 + (y - 310)**2) / (30.0**2))) * 70.0
    xray += infiltrate

    # 7. Add realistic anatomical Poisson/Gaussian noise & smooth
    noise = np.random.normal(0, 4.0, (img_size, img_size))
    xray = np.clip(xray + noise, 0, 255).astype(np.uint8)
    xray_blurred = cv2.GaussianBlur(xray, (3, 3), 0)

    # Convert to 3-channel RGB image and save
    xray_rgb = cv2.cvtColor(xray_blurred, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(output_path, xray_rgb)
    return output_path
