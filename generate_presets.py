"""
Generator for realistic synthetic sample chest X-rays with specific pathological findings.
"""

import os
import cv2
import numpy as np


def create_base_xray(size=512):
    """Generates standard normal thoracic anatomy base."""
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size // 2, size // 2
    
    # Background gradient
    xray = 25.0 + 15.0 * np.sin(y / 40.0)
    
    # Thorax contour
    thorax = ((x - cx)**2 / (195**2) + (y - cy)**2 / (225**2)) <= 1.0
    xray[thorax] += 65.0

    # Lungs (left & right air spaces - darker)
    left_lung = (((x - 165)**2 / (65**2) + (y - 245)**2 / (125**2)) <= 1.0)
    right_lung = (((x - 345)**2 / (65**2) + (y - 245)**2 / (125**2)) <= 1.0)
    xray[left_lung] = np.clip(xray[left_lung] - 55.0, 18, 255)
    xray[right_lung] = np.clip(xray[right_lung] - 55.0, 18, 255)

    # Mediastinum and Heart
    heart = (((x - 220)**2 / (58**2) + (y - 295)**2 / (55**2)) <= 1.0)
    xray[heart] += 90.0

    # Spine column
    spine = (np.abs(x - cx) < 18) & (y > 90) & (y < 460)
    xray[spine] += 30.0

    # Ribs
    for ry in range(160, 420, 32):
        rib = np.exp(-((y - ry - 0.04 * (x - cx)**2 / 10)**2) / 10.0)
        xray += rib * 22.0

    # Clavicles
    left_clav = np.exp(-((y - 130 - 0.14 * (x - 175))**2) / 7.0) * (x < 250) * (x > 100)
    right_clav = np.exp(-((y - 130 + 0.14 * (x - 335))**2) / 7.0) * (x > 260) * (x < 410)
    xray += (left_clav + right_clav) * 40.0

    return xray, (x, y, cx, cy)


def generate_preset_effusion(path):
    xray, (x, y, cx, cy) = create_base_xray()
    # Pleural effusion: dense fluid meniscus blunting right costophrenic angle
    fluid = np.exp(-(((x - 365)**2 / (50**2)) + ((y - 350)**2 / (35**2)))) * 85.0
    fluid_base = (x > 310) & (y > 340) & (y < 390)
    xray += fluid
    xray[fluid_base] += 40.0
    save_final_xray(xray, path)


def generate_preset_hernia(path):
    xray, (x, y, cx, cy) = create_base_xray()
    # Hiatal hernia: retrocardiac / supra-diaphragmatic gas-fluid mass
    hernia = np.exp(-(((x - 240)**2 / (40**2)) + ((y - 275)**2 / (40**2)))) * 80.0
    xray += hernia
    save_final_xray(xray, path)


def generate_preset_pneumothorax(path):
    xray, (x, y, cx, cy) = create_base_xray()
    # Pneumothorax: apical visceral pleural line and hyperlucency
    pno_zone = (x > 330) & (y > 140) & (y < 260)
    xray[pno_zone] = np.clip(xray[pno_zone] - 30.0, 10, 255)
    line = np.exp(-((x - 325)**2) / 3.0) * (y > 140) * (y < 260) * 35.0
    xray += line
    save_final_xray(xray, path)


def generate_preset_normal(path):
    xray, _ = create_base_xray()
    save_final_xray(xray, path)


def save_final_xray(xray, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    noise = np.random.normal(0, 3.5, xray.shape)
    xray_clipped = np.clip(xray + noise, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(xray_clipped, (3, 3), 0)
    rgb = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(path, rgb)


def generate_all_presets():
    out_dir = os.path.join("data", "test_images")
    generate_preset_effusion(os.path.join(out_dir, "sample_effusion.jpg"))
    generate_preset_hernia(os.path.join(out_dir, "sample_hernia.jpg"))
    generate_preset_pneumothorax(os.path.join(out_dir, "sample_pneumothorax.jpg"))
    generate_preset_normal(os.path.join(out_dir, "sample_normal.jpg"))
    print("[+] All sample preset images created successfully in data/test_images/")


if __name__ == "__main__":
    generate_all_presets()
