"""
R-MMAF Tactical Command Center — Gradio Dashboard
===================================================
Live dual-sensor (RGB + Thermal IR) threat detection UI.

Classifies aerial objects into three alert tiers:
  🔴 RED    — Confirmed mechanical threat (aircraft / drone proxy)
  🟡 YELLOW — Biological signature (bird — investigating)
  🟢 GREEN  — Airspace secure

Run this file directly to launch the dashboard:
    python dashboard.py

Author: Sanchit Agarwal
"""

import random
import time
import urllib.request

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Model & Image Setup
# ---------------------------------------------------------------------------

print("Loading AI model...")
model = YOLO("yolov8n.pt")

IMAGE_URLS = [
    # Safe (ground objects)
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg",
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg",
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/horses.jpg",
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/person.jpg",
    # Yellow alert (birds)
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/eagle.jpg",
    # Red alert (drone proxies)
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/kite.jpg",
]

print("Downloading CCTV sample frames…")
cctv_images = []
for i, url in enumerate(IMAGE_URLS):
    filename = f"cctv_feed_{i}.jpg"
    req = urllib.request.Request(url, headers={"User-Agent": "RMMAF/2.0"})
    try:
        with open(filename, "wb") as f:
            f.write(urllib.request.urlopen(req).read())
        cctv_images.append(filename)
        time.sleep(0.3)
    except Exception:
        print(f"  Skipped image {i} (server blocked).")

print(f"Loaded {len(cctv_images)} frames into CCTV array.")


# ---------------------------------------------------------------------------
# Core Inference Logic
# ---------------------------------------------------------------------------

# COCO class IDs used as drone proxies
THREAT_IDS = {4, 33}   # airplane, kite
BIRD_ID     = 14


def analyze_camera_feed():
    """
    Selects a random CCTV frame, runs YOLOv8 detection, generates a
    simulated thermal overlay, and returns the alert state.
    """
    frame_path = random.choice(cctv_images)
    frame      = cv2.imread(frame_path)

    results          = model(frame)
    annotated_frame  = results[0].plot()

    # Simulated thermal: grayscale → INFERNO colormap
    gray_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thermal_frame = cv2.applyColorMap(gray_frame, cv2.COLORMAP_INFERNO)

    confirmed_threat  = False
    investigating_bird = False
    detected_objects  = []

    for box in results[0].boxes:
        class_id   = int(box.cls[0])
        class_name = model.names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if class_id in THREAT_IDS:
            confirmed_threat = True
            detected_objects.append(class_name)
            cv2.rectangle(thermal_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(thermal_frame, "MECHANICAL HOTSPOT",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        elif class_id == BIRD_ID:
            investigating_bird = True
            detected_objects.append(class_name)
            cv2.rectangle(thermal_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(thermal_frame, "BIOLOGICAL SIGNATURE",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Build alert HTML and countermeasure text
    if confirmed_threat:
        threats      = ", ".join(set(detected_objects)).upper()
        alert_html   = _red_alert(threats)
        countermeasures = "1. Target locked.\n2. Trajectory mapped.\n3. Readying RF jamming signal."
    elif investigating_bird:
        alert_html      = _yellow_alert()
        countermeasures = "EADW Thermal Fusion confirmed normal biological signature. Cancelling alert."
    else:
        alert_html      = _green_alert()
        countermeasures = "Systems nominal. Standard dual-sensor sweep active."

    # Convert BGR → RGB for Gradio
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    thermal_frame   = cv2.cvtColor(thermal_frame,   cv2.COLOR_BGR2RGB)

    return annotated_frame, thermal_frame, alert_html, countermeasures


# ---------------------------------------------------------------------------
# Alert HTML Helpers
# ---------------------------------------------------------------------------

def _red_alert(threats: str) -> str:
    return f"""
    <div style="background:red;color:white;padding:20px;text-align:center;
                border-radius:10px;animation:blinker 1s linear infinite;">
        <h1 style="margin:0">🚨 CONFIRMED THREAT: {threats} 🚨</h1>
        <h3 style="margin:0">Mechanical Aerial Object Detected</h3>
    </div>
    <style>@keyframes blinker{{50%{{opacity:0.5}}}}</style>"""


def _yellow_alert() -> str:
    return """
    <div style="background:#d4a017;color:white;padding:20px;text-align:center;border-radius:10px;">
        <h1 style="margin:0">⚠️ INVESTIGATING AIRSPACE ⚠️</h1>
        <h3 style="margin:0">Bird detected — cross-referencing thermal for biomimicry…</h3>
        <h4 style="margin:5px 0 0 0;color:#e0ffd4">Result: Biological heat signature. No mechanical hotspots. Alert cleared.</h4>
    </div>"""


def _green_alert() -> str:
    return """
    <div style="background:green;color:white;padding:20px;text-align:center;border-radius:10px;">
        <h2 style="margin:0">✅ AIRSPACE SECURE</h2>
    </div>"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Monochrome()) as dashboard:
    gr.Markdown("# 🚁 R-MMAF Tactical Command Center")
    gr.Markdown("Real-Time Multi-Modal Threat Detection System (Visual + Thermal Fusion)")

    with gr.Row():
        camera_output  = gr.Image(label="Live Visual Feed (RGB)")
        thermal_output = gr.Image(label="Live Thermal Feed (IR)")

    with gr.Row():
        with gr.Column(scale=1):
            scan_btn = gr.Button("📸 Scan Next Dual-Sensor Frame", variant="primary", size="lg")
        with gr.Column(scale=2):
            alert_display         = gr.HTML(label="System Status")
            countermeasure_display = gr.Textbox(label="Active Countermeasures", lines=4)

    scan_btn.click(
        fn=analyze_camera_feed,
        inputs=[],
        outputs=[camera_output, thermal_output, alert_display, countermeasure_display],
    )

if __name__ == "__main__":
    dashboard.launch(share=True)
