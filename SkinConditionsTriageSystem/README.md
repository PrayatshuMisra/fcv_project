\# ClinicalVision 2.0: Automated Visual Triage System for Skin Lesions 🔬

python -m uvicorn api:app --reload --port 8000
http://localhost:8000


\[!\[Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python\&logoColor=white)](https://www.python.org/)

\[!\[FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)

\[!\[OpenCV](https://img.shields.io/badge/OpenCV-4.11-5C3EE8?logo=opencv\&logoColor=white)](https://opencv.org/)

\[!\[Course](https://img.shields.io/badge/Course-CSE%203176%20FCV%20Lab-0284C7)](https://manipal.edu/mit.html)



A high-performance, explainable computer vision pipeline and clinical triage dashboard for dermatological lesion assessment, engineered strictly using foundational computer vision principles from \*\*CSE 3176: Foundations of Computer Vision (FCV) Lab (Labs 1 through 5)\*\* at \*\*Manipal Institute of Technology (MIT), Manipal\*\*.



\---



\## 👨‍💻 Project Team \& Academic Credentials

\- \*\*Rajit Mohan Shrivastava\*\* (Reg. No. 240962378, Roll No. 24) — AIML-B

\- \*\*Prayatshu Misra\*\* (Reg. No. 240962386, Roll No. 32) — AIML-B

\- \*\*Eknoor Singh Chhabra\*\* (Reg. No. 240962412, Roll No. 58) — AIML-B

\- \*\*Faculty In-Charge / Examiner:\*\* Dr. Roopalakshmi R, Associate Professor, School of Computer Engineering, MIT Manipal.

\- \*\*Academic Year:\*\* 2026–2027 (Semester 5)



\---



\## 📑 Deliverables \& Submission Index

\- 📄 \*\*Official Submission Word Report:\*\* \[`FCV\_MiniProject\_Report.docx`](file:///d:/study/MIT%20PROJECTS/Sem5/FCV/SkinConditionsTriageSystem/FCV\_MiniProject\_Report.docx) \*(Strictly formatted to template: Times New Roman, 7 Chapters, IEEE References, Certificate)\*

\- 📖 \*\*Complete Markdown Report:\*\* \[`REPORT.md`](file:///d:/study/MIT%20PROJECTS/Sem5/FCV/SkinConditionsTriageSystem/REPORT.md)

\- 🎓 \*\*Viva Voce Defense Guide:\*\* \[`VIVA\_PREPARATION.md`](file:///d:/study/MIT%20PROJECTS/Sem5/FCV/SkinConditionsTriageSystem/VIVA\_PREPARATION.md) \*(Top 10 examiner questions \& mathematical derivations)\*

\- 📓 \*\*Interactive Jupyter Notebook:\*\* \[`SkinTriage\_Notebook.ipynb`](file:///d:/study/MIT%20PROJECTS/Sem5/FCV/SkinConditionsTriageSystem/SkinTriage\_Notebook.ipynb) \*(Step-by-step Labs 1–5 pedagogical walkthrough)\*

\- 🌐 \*\*Interactive Web Triage UI:\*\* \[`static/index.html`](file:///d:/study/MIT%20PROJECTS/Sem5/FCV/SkinConditionsTriageSystem/static/index.html) \*(FastAPI + Glassmorphism Clinical Dashboard)\*



\---



\## 🌟 Technical Pipeline \& Syllabus Mapping (Labs 1–5)



```mermaid

flowchart LR

&#x20;   A\["Raw Clinical Photo"] --> B\["Phase 0: DullRazor<br/>Hair Inpainting (Lab 3)"]

&#x20;   B --> C\["Phase 1: Multi-Color Space<br/>\& CLAHE (Labs 1 \& 2)"]

&#x20;   C --> D\["Phase 2: Bilateral<br/>Edge-Preserving Filter (Lab 3)"]

&#x20;   D --> E\["Phase 3: Canny \& Sobel<br/>Boundary Detection (Lab 4)"]

&#x20;   E --> F\["Phase 4: 3D K-Means<br/>\& Lesion Masking (Labs 4 \& 5)"]

&#x20;   F --> G\["Phase 5: LBP \& GLCM<br/>Texture Fingerprints (Lab 5)"]

&#x20;   G --> H\["Phase 6: Stolz ABCD<br/>TDS Diagnostic Scoring"]

```



\### 1. Lab 1 — Image Basics \& Multi-Color Space Transformations

\- \*\*Decoupled Color Models:\*\* Decouples luminance from chromatic components by converting raw RGB to \*\*CIE LAB\*\* and \*\*HSV\*\*.

\- \*\*Diagnostic Chromophores:\*\*

&#x20; - $a^\*$ channel isolates vascular erythema (redness/inflammation).

&#x20; - $b^\*$ channel isolates melanin accumulation (deep pigment).

&#x20; - HSV Hue isolates chromophore angle, while Saturation captures pigment density.



\### 2. Lab 2 — Spatial Domain Intensity Transformations

\- \*\*CLAHE (Clinical Benchmark):\*\* Applies Contrast Limited Adaptive Histogram Equalization (`clipLimit=2.0`, $8 \\times 8$ grid) on the $L^\*$ channel to amplify subtle lesion details without washing out uniform skin or blowing up background noise.

\- \*\*Comparative Baseline:\*\* Evaluates Min-Max Contrast Stretching, Global Histogram Equalization, and Power-Law Gamma Correction ($\\gamma \\in \[0.4, 2.4]$).



\### 3. Lab 3 — Spatial Filtering \& Artifact Suppression

\- \*\*DullRazor Morphological Filter:\*\* Applies a $17 \\times 17$ Black Top-Hat morphological transform to isolate occluding dark hair strands, followed by Telea fast-marching inpainting (`cv2.inpaint`) to reconstruct clean dermal skin.

\- \*\*Bilateral Filtering:\*\* Combines a spatial Gaussian kernel ($\\sigma\_s=75$) with a radiometric intensity kernel ($\\sigma\_c=75$) to suppress epidermal pores and minor blemishes without smearing sharp lesion margins.



\### 4. Lab 4 — Edge Detection \& Morphological Geometry

\- \*\*Canny Edge Detector:\*\* Incorporates Gaussian pre-smoothing, Sobel first-order spatial derivatives ($G\_x, G\_y$), Non-Maximum Suppression (NMS), and hysteresis dual-thresholding ($T\_1=100, T\_2=200$).

\- \*\*Morphological Geometry:\*\* Applies morphological opening and closing to clean binary masks; traces primary contours (`cv2.findContours`); computes Convex Hull envelopes (`cv2.convexHull`), minimum area rotated bounding boxes (`cv2.minAreaRect`), and spatial moments (`cv2.moments`) to locate lesion centroids.



\### 5. Lab 5 — Unsupervised Segmentation, Texture Descriptors \& Clinical TDS

\- \*\*3D K-Means Color Segmentation:\*\* Partitions pixel vectors in 3D color space. Employs a robust \*\*border-sampling contrast localization algorithm\*\* to identify the lesion cluster without heuristic center-pixel assumptions.

\- \*\*Uniform Local Binary Patterns (LBP):\*\* Computes rotation-invariant LBP ($P=24, R=3$) and generates a normalized 26-bin histogram capturing micro-surface roughness.

\- \*\*Haralick GLCM Descriptors:\*\* Computes second-order spatial statistics across distances $\[1, 3]$ and angles $\[0^\\circ, 45^\\circ, 90^\\circ, 135^\\circ]$ (Contrast, Dissimilarity, Homogeneity, Energy, Correlation).

\- \*\*Stolz ABCD Rule \& Total Dermoscopy Score (TDS):\*\*

&#x20; $$\\text{TDS} = 1.3 \\times A + 0.1 \\times B + 0.5 \\times C + 0.5 \\times D$$

&#x20; - \*\*Asymmetry ($A$, 0–2):\*\* 2-axis folding overlap error along principal inertia axes.

&#x20; - \*\*Border Irregularity ($B$, 0–8):\*\* Radial 8-octant notch analysis and compactness ($P^2 / 4\\pi A$).

&#x20; - \*\*Color Variegation ($C$, 1–6):\*\* Detection of 6 clinical colors (white, red, light brown, dark brown, slate blue, black).

&#x20; - \*\*Diameter ($D$, 1–5):\*\* Calibrated millimeter size ($>6\\text{ mm}$ rule).

&#x20; - \*\*Stratification:\*\* Benign ($\\text{TDS} < 4.75$), Suspicious ($4.75 \\le \\text{TDS} < 5.45$), High Risk / Malignant ($\\text{TDS} \\ge 5.45$).



\---



\## 📊 Experimental Results on Clinical Cases



| Clinical Case | Asymmetry (A) | Border (B) | Color (C) | Diameter (D) | Stolz TDS | Risk Stratification | Clinical Protocol |

| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |

| \*\*Malignant Melanoma\*\* | 2 / 2 | 7 / 8 | 5 / 6 | 5 / 5 | \*\*6.60\*\* | \*\*High Risk / Malignant\*\* | Urgent Dermatological Biopsy |

| \*\*Benign Melanocytic Nevus\*\* | 0 / 2 | 1 / 8 | 2 / 6 | 2 / 5 | \*\*2.00\*\* | \*\*Low Risk / Likely Benign\*\* | Routine Annual Surveillance |

| \*\*Basal Cell Carcinoma\*\* | 1 / 2 | 5 / 8 | 3 / 6 | 3 / 5 | \*\*4.80\*\* | \*\*Suspicious / Moderate\*\* | Specialist Evaluation (3 Months) |

| \*\*Seborrheic Keratosis\*\* | 1 / 2 | 4 / 8 | 2 / 6 | 3 / 5 | \*\*4.20\*\* | \*\*Low Risk / Benign Plaque\*\* | Routine Clinical Monitoring |



\---



\## 🚀 Quickstart Guide



\### 1. Environment Installation

Ensure Python 3.11+ is installed. Install required dependencies:

```bash

pip install -r requirements.txt

```



\### 2. Launch the Web Clinical Dashboard

Start the high-performance FastAPI server:

```bash

python -m uvicorn api:app --reload --port 8000

```

Open your web browser and navigate to:

👉 \*\*\[http://127.0.0.1:8000](http://127.0.0.1:8000)\*\*



\### 3. Test Built-in Clinical Cases (1-Click Evaluation)

Click any of the pre-loaded clinical sample buttons at the top of the dashboard:

\- \*\*Malignant Melanoma\*\* \*(High Risk)\*

\- \*\*Benign Melanocytic Nevus\*\* \*(Low Risk)\*

\- \*\*Basal Cell Carcinoma\*\* \*(BCC)\*

\- \*\*Seborrheic Keratosis\*\* \*(Benign Plaque)\*



\### 4. Export Printable Medical Dossier

After analyzing any lesion, click the \*\*"Export Clinical Dossier"\*\* button at the top right to generate a complete, formatted medical triage sheet ready for printing or saving as a PDF.



\### 5. Interactive Jupyter Notebook

Open `SkinTriage\_Notebook.ipynb` in VS Code or Jupyter Lab and run all cells for an interactive, step-by-step mathematical demonstration of Labs 1 through 5.



\---



\## 📁 Repository Structure

```

SkinConditionsTriageSystem/

├── api.py                                 # FastAPI application \& REST endpoints

├── utils.py                               # Core CV algorithms (Labs 1-5 \& Stolz TDS)

├── generate\_report.py                     # Automated Word report generator (python-docx)

├── generate\_samples.py                    # Synthetic clinical sample generator

├── build\_notebook.py                      # Pedagogical Jupyter notebook generator

├── FCV\_MiniProject\_Report.docx            # Official 7-chapter submission report

├── REPORT.md                              # Comprehensive Markdown report

├── VIVA\_PREPARATION.md                    # Examiner viva voce defense guide

├── SkinTriage\_Notebook.ipynb              # Interactive Lab 1-5 Jupyter notebook

├── requirements.txt                       # Project dependencies

├── static/

│   ├── index.html                         # Clinical dashboard UI

│   ├── style.css                          # Medical glassmorphic stylesheet

│   ├── script.js                          # Frontend logic \& Chart.js rendering

│   └── samples/                           # Validated clinical sample images

│       ├── melanoma.jpg

│       ├── benign\_nevus.jpg

│       ├── basal\_cell.jpg

│       └── seborrheic\_keratosis.jpg

```



