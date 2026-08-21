import os
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr

from config import CLASS_NAMES
from utils import build_densenet_gru_model

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "final_densenet_gru.keras"))
MODEL = None

def get_model():
    """Singleton pattern to keep DenseNet121+GRU loaded in memory."""
    global MODEL
    if MODEL is None:
        if os.path.exists(MODEL_PATH):
            print(f"[+] Loading existing model from: {MODEL_PATH}")
            try:
                MODEL = tf.keras.models.load_model(MODEL_PATH)
            except Exception as e:
                print(f"[!] Load failed ({e}). Building fresh DenseNet121+GRU model...")
                MODEL = build_densenet_gru_model(num_classes=len(CLASS_NAMES))
                MODEL.save(MODEL_PATH)
        else:
            print("[*] Instantiating DenseNet121+GRU model architecture...")
            MODEL = build_densenet_gru_model(num_classes=len(CLASS_NAMES))
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            MODEL.save(MODEL_PATH)
    return MODEL

def preprocess_image(image):
    """
    Ubah ukuran (resize) menjadi 224x224, normalisasi piksel (0-1), 
    pastikan format menjadi 3 channel (RGB) jika inputnya grayscale, 
    lalu tambahkan batch dimension.
    """
    if image is None:
        return None
    
    # Memastikan format 3 channel (RGB) jika grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Menghapus alpha channel jika format RGBA
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
    # Resize menjadi 224x224
    img_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Normalisasi piksel (0-1)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Menambahkan batch dimension
    img_tensor = np.expand_dims(img_normalized, axis=0)
    
    return img_tensor

def predict_xray(image):
    """
    Fungsi utama untuk menerima gambar dari Gradio dan mengembalikan
    dictionary probabilitas per kelas untuk komponen gr.Label.
    """
    if image is None:
        return {}
    
    img_tensor = preprocess_image(image)
    model = get_model()
    
    # Prediksi model
    raw_preds = model.predict(img_tensor, verbose=0)[0]
    
    # Petakan hasil prediksi ke nama kelas (15 kelas)
    results = {}
    for idx, name in enumerate(CLASS_NAMES):
        results[name] = float(raw_preds[idx])
        
    return results

# --- Konfigurasi Antarmuka Gradio ---
title = "Sistem Klasifikasi X-Ray Toraks (DenseNet-GRU)"
description = """
Aplikasi **CITRA X-RAY** merupakan aplikasi web berbasis AI untuk klasifikasi multi-label 15 penyakit toraks pada citra rontgen dada (*Chest X-Ray*). 
Aplikasi ini dirancang sebagai **alat bantu pendukung keputusan klinis (*Clinical Decision Support System*)** bagi tenaga medis, khususnya radiolog, 
untuk memperoleh hasil analisis citra rontgen secara lebih cepat dan akurat.

**Penyakit yang dideteksi:** Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, No Finding, Nodule, Pleural Thickening, Pneumonia, dan Pneumothorax.
"""

iface = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(type="numpy", label="Upload Citra X-Ray (PNG/JPG)"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 Prediksi Penyakit"),
    title=title,
    description=description,
    allow_flagging="never",
    examples=[
        ["data/test_images/sample_effusion.jpg"],
        ["data/test_images/sample_normal.jpg"],
        ["data/test_images/sample_pneumothorax.jpg"]
    ]
)

if __name__ == "__main__":
    print("[*] Memuat model dan memulai server Gradio...")
    get_model()
    # Menggunakan port 7860 yang umumnya dipakai oleh Gradio (atau HF Spaces)
    iface.launch(server_name="0.0.0.0", server_port=7860)
