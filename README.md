# thoracic-xray-multilabel-densenet-gru
Proyek penelitian ini berfokus pada pengembangan sistem **klasifikasi multi-label untuk 15 kelas penyakit toraks** menggunakan citra radiografi X-Ray dada dari dataset **NIH ChestX-ray14**.


# Klasifikasi Multi-Label Penyakit Toraks pada Citra X-Ray Menggunakan Arsitektur CNN+RNN (DenseNet121+GRU) dengan Visualisasi Score-CAM

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.style=for-the-badge)](#license)

## 📌 Deskripsi Proyek
Proyek penelitian ini berfokus pada pengembangan sistem **klasifikasi multi-label untuk 15 kelas penyakit toraks** menggunakan citra radiografi X-Ray dada dari dataset **NIH ChestX-ray14**. 

Model mengadaptasi pendekatan hibrida **CNN+RNN (DenseNet121 + GRU)** yang menggabungkan kemampuan ekstraksi fitur spasial tingkat tinggi dari CNN dengan pemrosesan sekuensial rekuren GRU. Untuk mengatasi ketidakseimbangan kelas (*class imbalance*) yang ekstrim, digunakan **Weighted Binary Cross-Entropy Loss**. Keterbacaan dan interpretabilitas model medis dijamin melalui metode visualisasi peta panas (**Score-CAM**).

---

## 🎓 Informasi Peneliti
* **Penyusun:** Ayu Nalarati
* **NIM:** 2222105048
* **Program Studi:** Teknik Informatika
* **Fakultas:** Fakultas Teknik
* **Instansi:** Universitas Cendekia Abditama (Tangerang, 2026)

---

## 🔑 Fitur Utama
* **Arsitektur Hybrid CNN-RNN:** DenseNet121 sebagai *backbone feature extractor* dipadukan dengan 256 unit GRU untuk mempelajari ketergantungan antar-region fitur spasial.
* **Penanganan Class Imbalance:** Implementasi *Weighted Loss Function* yang dinamis (pembobotan maksimum hingga 10,0) untuk meningkatkan sensitivitas pada kelas langka seperti Hernia, Fibrosis, dan Edema.
* **Explainable AI (XAI) via Score-CAM:** Menghasilkan heatmap visualisasi berbasis skor untuk mengidentifikasi *Region of Interest* (ROI) medis tanpa terganggu batasan alur gradien RNN.

---

## 📊 Ringkasan Hasil Evaluasi

Model **DenseNet121 + GRU** berhasil mencapai **Mean AUROC sebesar 0,7111** (peningkatan **+6,79%** dibandingkan baseline DenseNet121 murni sebesar 0,6659).

### Perbandingan AUROC per Kelas (Baseline vs Model Usulan)
| Kelas Penyakit | DenseNet121 (Baseline) | DenseNet121 + GRU (Usulan) | $\Delta$ AUROC |
| :--- | :---: | :---: | :---: |
| **Hernia** | 0,6062 | **0,8540** | 🟢 **+0,2478** |
| **Effusion** | 0,7911 | **0,7933** | 🟢 +0,0022 |
| **Edema** | 0,7468 | **0,7925** | 🟢 +0,0457 |
| **Consolidation** | 0,7530 | **0,7628** | 🟢 +0,0098 |
| **Pneumothorax** | 0,7252 | **0,7603** | 🟢 +0,0351 |
| **Emphysema** | 0,6500 | **0,7397** | 🟢 **+0,0897** |
| **Cardiomegaly** | 0,6597 | **0,7196** | 🟢 +0,0599 |
| **Atelectasis** | 0,7038 | **0,7028** | 🔴 -0,0010 |
| **No Finding** | 0,6995 | **0,6974** | 🔴 -0,0021 |
| **Fibrosis** | 0,5559 | **0,6917** | 🟢 **+0,1358** |
| **Pleural Thickening** | 0,6317 | **0,6841** | 🟢 +0,0524 |
| **Pneumonia** | 0,7030 | **0,6607** | 🔴 -0,0423 |
| **Mass** | 0,5643 | **0,6229** | 🟢 +0,0586 |
| **Infiltration** | 0,6212 | **0,6068** | 🔴 -0,0144 |
| **Nodule** | 0,5769 | **0,5786** | 🟢 +0,0017 |
| **MEAN AUROC** | **0,6659** | **0,7111** | 🟢 **+0,0452** |

---

## 🏗️ Arsitektur Model

```text
Input Image (224 x 224 x 3)
       │
       ▼
DenseNet121 Backbone (Outputs 7 x 7 x 1024)
       │
       ▼
Reshape Layer ──> Sekuens 1D (49 x 1024)
       │
       ▼
GRU Layer (256 Units)
       │
       ▼
Dense + ReLU (256) ──> Dropout (0.5)
       │
       ▼
Dense + ReLU (128) ──> Dropout (0.25)
       │
       ▼
Output Sigmoid Layer (15 Classes Multi-Label)
