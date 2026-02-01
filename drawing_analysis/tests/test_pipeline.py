import sys
import os
import cv2
import numpy as np
import time

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

from pupil_adapter import PupilAdapter
from calibration import Calibration
from gaze_mapping import GazeMapper
from heatmap import HeatmapAccumulator
from session import Session
from export_ntopo import export_to_ntopology
from export_mesh import export_heightfield_obj

def test_pipeline():
    print("Initializing components...")
    adapter = PupilAdapter()
    calibration = Calibration()
    mapper = GazeMapper(calibration)
    heatmap = HeatmapAccumulator(297, 210)
    session = Session(output_dir="test_output")

    # Mock Calibration
    print("Setting mock calibration...")
    # Assuming 640x480 image
    calibration.set_corners([
        (100, 100), # TL
        (540, 100), # TR
        (540, 380), # BR
        (100, 380)  # BL
    ])
    
    # Load video
    # Resolve absolute path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    video_path = os.path.join(base_dir, "eyetracker_base/eye_test.mp4")
    
    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return

    print("Processing frames...")
    session.start()
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize to 640x480 to match detector assumptions
        frame = cv2.resize(frame, (640, 480))
        
        # 1. Detect
        pupil = adapter.process_frame(frame)
        
        if pupil:
            # 2. Map
            gaze = mapper.map_gaze(pupil['center'], time.time())
            
            if gaze:
                # 3. Accumulate
                heatmap.add_point(gaze[0], gaze[1])
                session.add_sample(time.time(), gaze[0], gaze[1], pupil)
                
        frame_count += 1
        if frame_count > 100: # Run for 100 frames
            break
            
    session.stop()
    print(f"Processed {frame_count} frames.")
    
    # Export
    print("Exporting data...")
    export_to_ntopology(session.samples, heatmap.get_normalized_grid(), 
                        heatmap.width_mm, heatmap.height_mm, "test_output/test_export.json")
                        
    export_heightfield_obj(heatmap.get_normalized_grid(), 
                           heatmap.width_mm, heatmap.height_mm, output_path="test_output/test_relief.obj")
                           
    print("Test complete.")

if __name__ == "__main__":
    test_pipeline()
