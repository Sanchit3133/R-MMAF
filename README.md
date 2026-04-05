# 🚁 R-MMAF: Robust Multi-Modal Affective-Aware Framework

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SAHI-Sliced%20Inference-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Gradio-Live%20Dashboard-FF6B6B?style=for-the-badge&logo=gradio"/>
  <a href="https://colab.research.google.com/drive/1ZBiJVToNZZD8nBJtH17pI7x4hd0L59cU?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height:28px"/>
  </a>
</p>

> **An end-to-end cyber-physical AI safety system** combining adversarial drone detection with panic-aware crowd evacuation modelling — designed for real-world, all-weather public venue security.

---

## 📌 The Problem

Imagine an unauthorised, potentially weaponised drone breaching the airspace of a packed outdoor concert. Current security systems fail at **three critical points**:

| Failure Point | What Goes Wrong |
|---|---|
| 🌫️ **Optical Blindness** | Standard cameras fail in fog, rain, or night — the drone simply cannot be seen |
| 🎯 **Tracking Loss** | Algorithms lose the drone when it moves erratically against a cluttered skyline |
| 😱 **Panic Blindspot** | Evacuation software treats humans like calm, rational atoms — ignoring how fear causes herding, freezing, and fatal bottlenecks |

**R-MMAF closes all three gaps in a single, integrated pipeline.**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     R-MMAF Safety Loop                          │
│                                                                 │
│  [RGB Camera] ──┐                                               │
│                 ├──▶ EADW Fusion ──▶ SAHI+YOLOv8 ──▶ TA-GRU   │
│  [Thermal IR] ──┘   (weighted by      (sliced        (temporal  │
│                      environment)      inference)     tracking)  │
│                                              │                  │
│                                              ▼                  │
│                                    BHSFM Crowd Simulation       │
│                                    (panic-aware evacuation)     │
│                                              │                  │
│                                              ▼                  │
│                                    Gradio Command Dashboard     │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Core Modules

### 1. 🌡️ EADW — Environment-Aware Dynamic Weighting (`src/eadw.py`)
Instead of blindly mixing RGB and Thermal IR feeds, EADW constructs an **environmental awareness vector** from fog density, precipitation, and illumination. It dynamically shifts sensor trust — when fog rolls in, the system instantly increases thermal weight to maintain drone lock.

```python
from src.eadw import EnvironmentSimulator, calculate_eadw_weights

midnight_fog = EnvironmentSimulator(fog_density=0.9, illumination=0.1)
visual_w, thermal_w = calculate_eadw_weights(midnight_fog)
# → Visual: 0.10  |  Thermal: 0.90  (system trusts IR almost entirely)
```

| Scenario | Visual Weight | Thermal Weight |
|---|---|---|
| Sunny Day | 1.00 | 0.00 |
| Midnight Fog | 0.10 | 0.90 |
| Heavy Storm | 0.10 | 0.90 |
| Overcast Night | 0.05 | 0.95 |

---

### 2. 🔍 Sliced Drone Detection — YOLOv8 + SAHI (`src/detection.py`)
Standard convolution downsampling deletes the tiny pixels of a distant drone. SAHI's **Simple Slicing Convolution** tiles the image into 320×320 overlapping patches, runs YOLOv8 on each, then merges detections with NMS — catching objects that full-resolution inference would miss.

```python
from src.detection import build_detection_model, run_sliced_inference

model  = build_detection_model("yolov8n.pt", confidence_threshold=0.25)
result = run_sliced_inference("aerial_feed.jpg", model)
result.export_visuals(export_dir="output/", file_name="detected")
```

---

### 3. 🧠 TA-GRU — Temporal Attention Gated Recurrent Unit
Gives the tracker **short-term memory** of the drone's trajectory. Temporal attention weights measure similarity between the drone's current features and past movements — enabling trajectory prediction even when the target passes behind a building.

---

### 4. 👥 BHSFM — Behavioural Heterogeneity Social Force Model (`src/bhsfm.py`)
A complete redesign of crowd simulation physics. Introduces a **panic coefficient** that alters agents' desired speeds and directions based on threat level — accurately modelling irrational behaviours like herding and freezing. Forked from PySocialForce with custom override of the self-driven force equation.

```python
from src.bhsfm import identify_leader
import numpy as np

positions  = np.random.rand(50, 2) * 100
velocities = np.random.rand(50, 2) * 2

leader_idx = identify_leader(positions, velocities)
# → Index of the frontrunner driving the crowd's direction
```

---

### 5. 🖥️ Live Dashboard — Gradio Command Center (`src/dashboard.py`)
Real-time dual-sensor UI with three alert tiers:

| Alert | Trigger | Action |
|---|---|---|
| 🔴 **RED** | Mechanical object detected (aircraft/drone) | RF jamming protocol initiated |
| 🟡 **YELLOW** | Biological signature detected (bird) | Thermal cross-reference to confirm/clear |
| 🟢 **GREEN** | Airspace clear | Standard sweep active |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/R-MMAF.git
cd R-MMAF

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the live dashboard
python src/dashboard.py
```

Or open directly in Google Colab (no setup needed):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZBiJVToNZZD8nBJtH17pI7x4hd0L59cU?usp=sharing)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Detection Model | YOLOv8 (Ultralytics) |
| Sliced Inference | SAHI |
| Temporal Tracking | Custom TA-GRU (PyTorch) |
| Sensor Fusion | Custom EADW Algorithm |
| Crowd Simulation | PySocialForce (custom fork) |
| Deployment | TensorRT INT8 (NVIDIA GPU) |
| Dashboard | Gradio |
| Language | Python 3.10+ |

---

## 📁 Repository Structure

```
R-MMAF/
├── README.md
├── requirements.txt
├── notebooks/
│   └── R_MMAF_Research.ipynb      ← Full Colab research notebook
├── src/
│   ├── eadw.py                    ← Sensor fusion (EADW module)
│   ├── detection.py               ← YOLOv8 + SAHI sliced inference
│   ├── bhsfm.py                   ← Panic-aware crowd simulation
│   └── dashboard.py               ← Gradio live dashboard
└── results/
    └── sample_outputs/            ← Detection visualisations
```

---

## 🔬 Research Context

This project is part of ongoing research into **cyber-physical safety systems for smart cities and public venues**. The R-MMAF addresses a critical gap in existing infrastructure: current systems handle aerial threats and crowd management in complete isolation. R-MMAF is the first framework to close the full loop — from drone detection under adverse weather to panic-aware, psychologically-realistic crowd evacuation.

**Key novel contributions:**
- EADW dynamic weighting eliminates weather-based sensor blind spots
- TA-GRU reduces computational cost of high-FPS aerial tracking
- BHSFM panic coefficient models real human crisis behaviour vs. physics-only SFM

---

## 👤 Author

**Sanchit Agarwal**
M.Tech CSE — IIIT Guwahati
Patent & Research Associate — Ennoble IP

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/sanchit-agarwal-4309b21a0/)
## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
