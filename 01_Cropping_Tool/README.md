# Computer Vision Dataset Annotation and Region of Interest (ROI) Extractor

An automated, high-throughput region-of-interest (ROI) extraction and grounding utility built for multi-modal Computer Vision and Deep Learning pipelines. This tool standardizes the initial phase of dataset engineering—specifically targeting tasks like Handwritten Prescription Analysis, Text Recognition (OCR), and Localized Object Classification—by allowing developers to extract sub-regions from raw high-resolution images, map spatial scale transformations, and map ground-truth labels directly into standardized filenames using non-colliding Universally Unique Identifiers (UUIDs).

---

## 🚀 Key Features

* **Dynamic Spatial Scaling:** Automatically maps pixel coordinates selected on scaled display windows back to original, high-resolution native image coordinate spaces using calculated scale factors ($S = H_{original} / H_{display}$).
* **Interactive ROI Grounding:** Integrates an active loop using OpenCV's standard Region of Interest (`selectROI`) API for manual bounding box generation via interactive mouse/keyboard interrupts.
* **Collision-Free Filename Serialization:** Appends an 8-character hardware-independent `uuid4` cryptographic salt token to prevent namespace collision on shared labels.
* **Real-time Preview Engine:** Displays immediate sub-matrix crops via synchronous GUI rendering before committing disk writes.
* **Automatic Multi-format IO Mapping:** Parses local filesystem directories using specific MIME-type sub-strings (`.png`, `.jpg`, `.jpeg`) with fault-tolerant exception handling for corrupted headers.

---

## 🛠 Architecture & Tech Stack

The architecture separates the physical display limits of client machines from the ground-truth high-fidelity image structures. It operates as a localized hardware-accelerated processing loop.
