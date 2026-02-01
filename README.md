# EyeTracking Drawing Analysis Tool
Version: 1.0.0
Date: February 2026

## Overview
**EyeTrackingFrame** is a complete, open-source eye-tracking platform designed to be **affordable (<$50)**, **plug-and-play**, and **research-ready**. 

By combining 3D-printed frames with a robust Python application, this tool transforms standard USB webcams into a high-precision gaze analysis instrument. It is ideal for **interactive art installations**, **academic research**, and **UX testing**.

## Key Capabilities
- **Dual Stream Logic**: Synchronizes an **Eye Camera** (IR) with a **Scene Camera** (World View).
- **Live Analysis**: Real-time **Heatmap Generation** and **Gaze Path** visualization.
- **Data Export**:
  - **3D Mesh (OBJ)**: Export heatmaps as congruently smoothed heightfields.
  - **nTopology (JSON)**: Direct integration with engineering workflows.
  - **Video (MP4)**: Record full sessions with overlaid gaze data.
- **Hardware Agnostic**: Runs on standard PC hardware with Python.

## Installation & Setup
1. **Hardware**:
   - Camera 0: Scene Camera (Webcam aiming at paper/screen)
   - Camera 1: Eye Camera (IR Camera aiming at eye)
   - Ensure proper USB bandwidth (plug into separate hubs if possible).

2. **Running**:
   - Launch `DrawingAnalysis.exe` from the folder.

## Workflow
1. **Camera Setup**: Select correct cameras in "Camera Source" tab. Adjust Exposure/Threshold until pupil is clearly detected (Blue Crosshair).
2. **Calibration**:
   - Click "Start Calibration".
   - Toggle to "Screen Map" (if using digital monitor) or "Scene Map" (if using physical paper).
   - Follow the 9-point target sequence.
3. **Session**:
   - Enter a Session ID.
   - Click "Start Recording".
   - Perform the drawing task.
   - Click "Stop Recording".
4. **Analysis & Export**:
   - Use the "Blur" slider to adjust heatmap smoothness.
   - Use the "Review/Export" tab to save the Heatmap as an OBJ mesh or JSON.

## Troubleshooting
- **No Cameras Found**: Check USB connections and ensure no other app is using them.
- **Unstable Pupil**: Adjust "Threshold" and "Min Area" sliders in the Eye Camera settings.
- **Slider Jitter**: Fixed in v1.0.0 (Feb 2026).

## License
MIT License.
Created by Lyes Hammadouche & Google DeepMind (Antigravity Assistant).
