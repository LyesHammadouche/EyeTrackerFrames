# EyeTracking Drawing Analysis Tool
Version: 1.0.0
Date: February 2026

## Overview
This application is a specialized eye-tracking analysis tool designed for drawing sessions. It captures gaze data using dual USB cameras (Eye + Scene), maps the gaze to a calibrated drawing surface, and generates density heatmaps that can be exported as congruently smoothed 3D meshes or JSON data for nTopology.

## Features
- **Dual Camera Tracking**: Simultaneous processing of Eye (IR) and Scene cameras.
- **Orlosky Pupil Detection**: Robust pupil detection algorithm.
- **9-Point Calibration**: Precise mapping from eye coordinates to drawing plane.
- **Heatmap Visualization**: Live Gaussian-smoothed heatmap overlay.
- **Congruent Smoothing**: "What You See Is What You Export" - mesh topology smoothness matches the visual blur slider exactly.
- **Media Stimulus**: Load Images or Videos to capture gaze responses to digital content.
- **Export Formats**:
  - OBJ (3D Heightfield Mesh)
  - JSON (nTopology Schema)
  - MP4 (Video Recording)

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
Created by [User Name] & Google DeepMind (Antigravity Assistant).
