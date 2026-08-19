# Automated Visual Triage System for Skin Conditions 🔬

A modern web application designed for dermatological assessment, built for the **Foundations of Computer Vision (CSE 3176)** Mini Project. This repository contains a premium User Interface and a high-performance computer vision backend for analyzing skin lesions.

## 👨‍💻 Team Details
- **Rajit Mohan Shrivastava** (240962378)
- **Prayatshu Misra** (240962386)
- **Eknoor Singh Chhabra** (240962412)
- **Branch:** AIML-B

## 🧠 What is this project? (For Non-Tech Readers)

Imagine walking into a clinic with a weird spot on your skin. The doctor takes a picture of it, but maybe the lighting in the room is too dim, or the picture is slightly blurry. This can make it hard for computer systems (or even doctors!) to analyze it perfectly.

This project is like a **smart photo assistant** for dermatologists. When you upload a picture of a skin condition, it does the following automatically:

1. **Fixes the Lighting (Enhancement):** It adjusts the brightness and contrast so the picture looks clear, no matter what room it was taken in.
2. **Cleans the Image (Noise Reduction):** It smooths out weird pixelation and draws a sharp outline around the edges of the skin spot.
3. **Isolates the Spot (Segmentation):** It acts like magic scissors, cutting out the infected area (the lesion) from the healthy skin so we only focus on what matters.
4. **Calculates a "Fingerprint" (Feature Analysis):** It looks closely at the texture of the spot (is it bumpy? smooth? rough?) and creates a unique mathematical fingerprint (a graph) for it.

By the time our system is done, the raw, messy photo is transformed into a clean, mathematical "profile" of the skin issue, ready to be sent to an AI for a fast and accurate diagnosis!

## 🚀 Future Scope & Real-World Use Cases

While this project is a foundational computer vision pipeline, it serves as the crucial **preprocessing engine** for real-world medical AI. In modern clinical settings, pure Deep Learning (like CNNs) is often a "black box" that doctors hesitate to trust. Our pipeline provides **interpretable, mathematical feature extraction** which can be extended in the following ways:

### 1. Automated "ABCD" Melanoma Detection
The outputs of this pipeline map perfectly to the dermatological **ABCD rule** for skin cancer:
*   **Asymmetry (A):** By finding the center of our **K-Means segmentation mask**, we can calculate if the two halves of the lesion match.
*   **Border (B):** Our **Canny Edge Detection** output can be mathematically analyzed to see if the border is ragged, notched, or blurred.
*   **Color (C):** By analyzing the isolated K-Means cluster, we can calculate the variance in RGB channels to detect dangerous multi-color pigmentation.
*   **Diameter (D):** By counting the pixels of our segmented mask (and calibrating with a known reference), we can automatically warn if the lesion is >6mm.

### 2. Machine Learning Classification (SVM / Random Forest)
Because our pipeline outputs a 1D array of numbers (the **LBP Histogram Signature**), this data can be directly fed into a classic machine learning classifier (like a Support Vector Machine). Instead of just showing the histogram, the system could confidently output: *"87% probability of Benign Keratosis based on texture signature."*

### 3. Tele-Dermatology Triage API
Because the backend is already decoupled using **FastAPI**, this system can immediately be plugged into a Mobile App (Flutter/React Native). Patients in remote areas could take photos with their phones, send them to this API, and the system would automatically discard bad photos, extract the features, and flag high-risk patients for a human doctor to review first.

## 🌟 Technical Pipeline (CV Labs 1-5)
1. **Phase I - Preprocessing:** Spatial domain intensity transformations (Histogram Equalization & Contrast Stretching) to normalize clinical photo brightness.
2. **Phase II - Noise Reduction:** Gaussian blur filter to suppress high-frequency artifacts and Canny Edge Detection to outline structural boundaries.
3. **Phase III - Segmentation:** Color-based K-Means clustering to partition the image and isolate the lesion from surrounding healthy tissue.
4. **Phase IV - Feature Analysis:** Quantitative texture profiling using the Local Binary Pattern (LBP) descriptor.

## ⚙️ Getting Started

The project features a **FastAPI backend** paired with a stunning **Clinical Dashboard Frontend**.

### Option 1: Run the Modern Web Dashboard (Recommended for Expert Level)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI server:
   ```bash
   python -m uvicorn api:app --reload
   ```
3. Open your browser and go to: [http://127.0.0.1:8000](http://127.0.0.1:8000)

### Option 2: Jupyter Notebook
A complete interactive notebook is provided for step-by-step execution.
1. Open `SkinTriage_Notebook.ipynb` in VSCode or Jupyter.
2. Run all cells to see the pipeline in action.

## 🛠️ Built With
- **FastAPI & Uvicorn** (High-performance API Server)
- **Vanilla HTML/CSS/JS** (Clinical Frontend UI)
- **OpenCV & Scikit-Image** (Computer Vision backend)
- **Chart.js** (Dynamic Data Visualization)
