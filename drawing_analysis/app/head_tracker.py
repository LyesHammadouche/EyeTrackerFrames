
import cv2
import mediapipe as mp
import numpy as np
import math
from PySide6.QtCore import QThread, Signal
import time

class HeadTrackerThread(QThread):
    # Signal emits: (has_face, yaw, pitch, x_screen, y_screen, frame_annotated)
    head_signal = Signal(bool, float, float, int, int, object)

    def __init__(self, camera_index=0, screen_w=1920, screen_h=1080):
        super().__init__()
        print(f"DEBUG: HeadTrackerThread __init__ called with index={camera_index}")
        self.camera_index = camera_index
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.running = False
        
        self.filter_length = 5
        self.yaw_buffer = []
        self.pitch_buffer = []

        # Calibration Offsets
        self.offset_yaw = 0
        self.offset_pitch = 0
        
        # Sensitivity / Range
        self.yaw_range = 25 # Degrees left/right to reach edge
        self.pitch_range = 15 # Degrees up/down to reach edge

    def run(self):
        self.running = True
        print(f"DEBUG: HeadTracker launching on camera index {self.camera_index}...")
        
        try:
            # Initialize MediaPipe
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles

            # Try DirectShow first (better for Windows webcams)
            print(f"DEBUG: HeadTracker trying cv2.CAP_DSHOW for index {self.camera_index}")
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                print(f"WARNING: HeadTracker CAP_DSHOW failed. Retrying default backend for index {self.camera_index}...")
                cap = cv2.VideoCapture(self.camera_index)

            # Try to set typical webcam res
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                print(f"CRITICAL ERROR: HeadTracker could not open camera {self.camera_index} with ANY backend.")
                self.running = False
                return

            print(f"DEBUG: HeadTracker Camera {self.camera_index} opened successfully.")

            # Indices (from typical subsets)
            # Nose tip: 1
            # Chin: 152
            # Left Eye Outer: 33
            # Right Eye Outer: 263
            # Left Ear: 234
            # Right Ear: 454
            
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # print("DEBUG: HeadTracker frame read failed (possibly looping or bad signal)")
                    time.sleep(0.1)
                    continue

                # Process
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                has_face = False
                yaw = 0.0
                pitch = 0.0
                sx = 0
                sy = 0

                if results.multi_face_landmarks:
                    has_face = True
                    has_face = True
                    # [Original Logic kept mostly intact, just indented]
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Draw Mesh
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    
                    # 3D Model Points
                    def get_pt(idx):
                        lm = landmarks[idx]
                        return np.array([lm.x * w, lm.y * h, lm.z * w]) # z is relative to width
                    
                    # Key Landmarks
                    pt_left = get_pt(234)  # Left ear area
                    pt_right = get_pt(454) # Right ear area
                    pt_top = get_pt(10)    # Forehead
                    pt_bottom = get_pt(152)# Chin

                    # Axes
                    right_vec = pt_right - pt_left
                    right_vec /= np.linalg.norm(right_vec)
                    
                    up_vec = pt_top - pt_bottom
                    up_vec /= np.linalg.norm(up_vec)
                    
                    forward_vec = np.cross(right_vec, up_vec)
                    forward_vec /= np.linalg.norm(forward_vec)
                    forward_vec = -forward_vec 

                    # Yaw (XZ plane)
                    xz = np.array([forward_vec[0], 0, forward_vec[2]])
                    if np.linalg.norm(xz) > 0:
                        xz /= np.linalg.norm(xz)
                        ref_z = np.array([0, 0, -1])
                        yaw_rad = math.acos(np.clip(np.dot(ref_z, xz), -1.0, 1.0))
                        if forward_vec[0] < 0: 
                            yaw_rad = -yaw_rad 
                    else:
                        yaw_rad = 0
                    
                    # Pitch (YZ plane)
                    yz = np.array([0, forward_vec[1], forward_vec[2]])
                    if np.linalg.norm(yz) > 0:
                        yz /= np.linalg.norm(yz)
                        ref_z = np.array([0, 0, -1])
                        pitch_rad = math.acos(np.clip(np.dot(ref_z, yz), -1.0, 1.0))
                        if forward_vec[1] > 0:
                            pitch_rad = -pitch_rad
                    else:
                        pitch_rad = 0
                    
                    # Convert to degrees
                    yaw_deg = math.degrees(yaw_rad)
                    pitch_deg = math.degrees(pitch_rad)
                    
                    # Apply Offsets
                    yaw_deg += self.offset_yaw
                    pitch_deg += self.offset_pitch

                    # Map to Screen
                    norm_x = (yaw_deg + self.yaw_range) / (2 * self.yaw_range)
                    norm_y = (pitch_deg - (-self.pitch_range)) / (2 * self.pitch_range)
                    
                    sx = int(norm_x * self.screen_w)
                    sy = int(norm_y * self.screen_h) 
                    
                    # Draw Helper
                    center_pt = (pt_left + pt_right) / 2
                    p1 = (int(center_pt[0]), int(center_pt[1]))
                    p2 = (int(center_pt[0] + forward_vec[0]*50), int(center_pt[1] + forward_vec[1]*50))
                    cv2.line(frame, p1, p2, (0, 75, 255), 3)
                    
                    yaw = yaw_deg
                    pitch = pitch_deg
                
                else:
                    has_face = False
                
                # Emit
                self.head_signal.emit(has_face, yaw, pitch, sx, sy, frame)

            cap.release()
            print("DEBUG: HeadTracker Loop ended normally.")

        except Exception as e:
            print(f"CRITICAL ERROR in HeadTrackerThread: {e}")
            import traceback
            traceback.print_exc()
            self.running = False

    def set_calibration(self, current_yaw, current_pitch):
        # Set offset so current yaw/pitch becomes (0,0) (Center of screen)
        # We subtract the current value from the existing offset
        self.offset_yaw -= current_yaw
        self.offset_pitch -= current_pitch

    def stop(self):
        self.running = False
        self.wait()
