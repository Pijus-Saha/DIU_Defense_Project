# PharmaScan Pro: AI-Powered Medicine Recognition

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.3-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Open_Source-green.svg)](#)

An intelligent, automated software solution designed to digitize illegible handwritten medical prescriptions, reducing the high risk of medication dispensing errors in healthcare systems. By employing a robust **Hybrid Deep Learning Approach** coupled with a **Heuristic Post-Processing Safety Net**, PharmaScan Pro extracts character sequences from static cursive script images and maps them securely to a verified pharmaceutical database.

This project was developed as a Final Year Design Project for a Bachelor of Science degree in Computer Science and Engineering at Daffodil International University.

---

## 📌 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#%EF%B8%8F-system-architecture)
- [Model Evaluation & Benchmarks](#-model-evaluation--benchmarks)
- [The Heuristic Safety Net](#-the-heuristic-safety-net)
- [Tech Stack & Environment Setup](#-tech-stack--environment-setup)
- [Functional Requirements](#-functional-requirements)
- [Future Roadmap](#-future-roadmap)
- [Contributors & Team](#-contributors--team)

---

## 🚀 Key Features

* **Offline Handwriting Recognition (HTR):** Resolves the complex spatial and sequential problem of analyzing unstructured cursive medication scripts from static images.
* **8-Model Deep Learning Benchmark:** Rigorous evaluation framework implementing standard CNN-RNN architectures, compound scaled structures, and advanced self-attention models (Vision Transformers).
* **Zero-Hallucination Post-Processing:** Combines stochastic machine learning models with a deterministic rule-based database matching loop using the Levenshtein Distance algorithm.
* **Clinical UI (PharmaScan Pro Desktop Application):** Built using a streamlined, "click-free" human-computer interaction (HCI) pattern via Tkinter, providing rapid visual validation with color-coded confidence indicators.
* **Custom Dataset:** Leverages a specifically compiled local dataset comprising 3,900 curated and labeled images representing local handwriting irregularities and clinical vocabulary.

---

## 🗺️ System Architecture

The core framework operates as a pipeline funnel broken into four isolated processing blocks:

```
[Input Image] ──> [Image Preprocessing] ──> [Visual Recognition] ──> [Hybrid Post-Processing] ──> [GUI Output Rendering]
                       (Adaptive                 (EfficientNet +              (Levenshtein Distance +       (PharmaScan Pro UI)
                     Thresholding)                    GRU)                         Database Loop)
```

1.  **Image Preprocessing Pipeline:** Grayscale conversion and **Adaptive Thresholding** remove shadow gradients, variations in pen thickness, and background paper noise. Final images are systematically scaled to a uniform tensor format of $32 \times 128$ pixels.
2.  **Visual Recognition Engine (CNN-RNN Backbone):**
    * **Spatial Feature Extraction:** Custom Convolutional Neural Networks (CNNs) identify physical cursive forms, loops, and micro-strokes.
    * **Sequence Modeling:** Recurrent layers map visual intervals sequentially to character distributions utilizing the **Connectionist Temporal Classification (CTC) Loss Function**.
3.  **Hybrid Post-Processing:** Translates raw output strings into validated database records, filtering out deep learning classification inconsistencies.

---

## 📊 Model Evaluation & Benchmarks

Eight deep architectures were evaluated against an isolated test sample pool of 780 text lines (20% hold-out evaluation schema) to establish optimal edge runtime metrics:

| Model Architecture | Character Error Rate (CER) | Raw Accuracy (AI Only) | System Accuracy (Hybrid DB-Corrected) |
| :--- | :---: | :---: | :---: |
| **EfficientNet_GRU** | 0.1403 | 57.95% | **91.92% (Overall Champion)** |
| **DenseNet_GRU** | 0.1403 | 59.36% | 91.28% |
| **Transformer (ViT)** | **0.1193** | 70.90% | 90.13% (Best NLP Middle-Ground) |
| **ResNet_GRU** | 0.1195 | **72.31% (Best Raw AI)** | 90.00% |
| **MobileNet_GRU** | 0.1739 | 52.44% | 88.33% (Mobile Optimization King) |
| **VGG_BILSTM** | 0.1433 | 65.51% | 87.05% |
| **GRU Baseline** | 0.2405 | 36.41% | 83.97% |
| **CRNN (Baseline LSTM)**| 0.2796 | 28.46% | 78.33% |

### 📈 Major Insights from the Efficiency Frontier:
* **The Sweet Spot:** `EfficientNet_GRU` achieves the maximum accuracy frontier (~92%) with a production latency profile of approximately **7ms per character block**.
* **The Vision Transformer Surprise:** The pure attention mechanism (`ViT`) demonstrated a highly structured error pattern, achieving the maximum Neural Sequence-to-Sequence correction performance (**74.7%** translation match).
* **Mobile Deployment Ready:** `MobileNet_GRU` operates with an ultra-lightweight **1.5 MB footprint** and execution latencies below **1.5ms**, maintaining a viable performance baseline of **88.33%**.

---

## 🔒 The Heuristic Safety Net

Standard deep learning or generative NLP architectures (such as Seq2Seq translation models) are highly prone to **AI Hallucinations**, generating fictional medication terms that sound linguistically authentic. In healthcare settings, this unpredictability introduces critical risk vectors.

PharmaScan Pro utilizes a **Deterministic Heuristic Correction System** that completely overrides neural hallucinations. By strictly referencing an isolated pharmaceutical databank (`MEDICINE_DB`), the system bounds all final corrections within authentic, registered pharmaceutical vocabulary, ensuring that the software acts as a mathematically reliable safety tool.

---

## ⚙️ Tech Stack & Environment Setup

### Hardware Development Profile
* **CPU:** Intel Core i7 12700H (12th Generation)
* **GPU:** NVIDIA GeForce RTX 3050 (4GB Dedicated VRAM)
* **Memory:** 16GB DDR5 RAM
* **Storage:** 512GB NVMe SSD

### System Requirements & Tools
* **Operating System:** Windows 11 (64-bit) / Linux Equivalent
* **Runtime:** Python 3.9
* **Core AI Libraries:** PyTorch (v1.12) with CUDA 11.3 support
* **Vision Toolkit:** OpenCV (cv2)
* **Data Structures:** Pandas, NumPy
* **Visualization:** Matplotlib
* **Interface Layer:** Tkinter


## 📋 Functional Requirements

* **Multi-Format Ingestion:** Seamless input supporting standard format encodings (`.jpg`, `.png`).
* **Real-Time Quantized Confidence Scoring:** Interleaved progress bar indicators that color-code prediction safety states (e.g., Green for High Confidence, Red for Low Confidence values).
* **History Scanned Logging:** An active memory structure allowing local operators to query and parse previously analyzed prescription rows.

---

## 🗺️ Future Roadmap

* **Automated Full-Page Document Parsing:** Integrating advanced edge detection frameworks (`YOLOv4 / Faster R-CNN`) to parse unstructured medical pages into atomic line elements without manual cropping boundaries.
* **National Database Expansion API Integration:** Linking the local heuristic table layer to cloud gateways managed by official regulatory bodies (such as the FDA or local pharmaceutical authorities), immediately scaling the system's runtime vocabulary to **20,000+** medication names.
* **Offline Global Edge AI Engine Compilation:** Porting compressed model paths directly into `TensorFlow Lite (TFLite)` and `ONNX` file profiles to facilitate fully offline cross-platform deployments on iOS and Android devices.
* **Advanced Prescription NER Integration:** Upgrading recurrent sequences to parse structured entity types, automatically separating compound dosages (e.g., `500mg`) and usage frequency sequences (e.g., `1+0+1`).


## 👥 Contributors & Team

* **Research & Core Engineering:** Pijus Saha (Student ID: 221-15-5809), Department of Computer Science and Engineering, Daffodil International University.
* **Academic Supervision:** Ms. Sharun Akter Khushbu, Assistant Professor, Department of CSE, Daffodil International University.
* **Co-Supervision:** Mr. Md Assaduzzaman, Assistant Professor, Department of CSE, Daffodil International University.
* **Board of Examiners:** Ms. Nazmun Nessa Moon (Chairman), Amit Chakraborty Chhoton (Internal), Ms. Taslima Akhter (Internal), Dr. Mohammed Nasir Uddin (External, Jagannath University).
