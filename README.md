# 📝 Handwritten Prescription Analysis: Data Annotation Tool

This component of the repository contains the **Manual Dataset Annotation & Cropping Utility**, which serves as the initial data-ingestion step in the "Handwritten Prescription Analysis" pipeline. This tool allows you to convert raw, full-page prescription images into a structured, labeled dataset of individual word segments for training CRNN/LSTM networks.

---

## 🚀 Key Features

* **Intelligent ROI Selection:** Implements a graphical user interface to manually isolate handwritten medicine names from complex backgrounds.
* **Zero-Loss Cropping:** Annotations are performed on a scaled preview for optimal usability, while the actual cropping logic targets the **original high-resolution source** via calculated scale factors.
* **Automated File Management:** * Generates unique file identifiers using `UUID` to guarantee no accidental overwrites.
    * Automatically structures and provisions target output directories.
    * Standardizes naming convention: `[label]_[unique_id].jpg`.

---

## 🛠️ Technical Implementation

### **Coordinate Remapping Logic**
To preserve the highest data resolution for training quality, the script calculates a linear ratio between the raw source dimensions and the target UI viewport window size:

$$Scale\ Factor = \frac{Original\ Image\ Height}{Display\ Height}$$

When a bounding box is drawn on-screen, the extracted coordinates $(x, y, w, h)$ are mapped back to the raw source coordinate grid to output full-resolution crops.

### **Workflow Processing Step**
1. **Load:** Scans targeted raw graphics files from the source directory.
2. **Scale:** Implements an aspect-ratio-preserving downscale to standard user-display dimensions.
3. **Select:** Polls OpenCV's native ROI selection module for bounding box array inputs.
4. **Label:** Records text annotations typed via standard CLI input.
5. **Save:** Cuts and writes the isolated data array out to disk with a hashed filename.

---

## 💻 Usage Instructions

### **1. Prerequisites**
Ensure you have Python installed along with the OpenCV library:
```bash
pip install opencv-python
