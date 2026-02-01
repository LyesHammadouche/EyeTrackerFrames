import sys
import os
import cv2
import numpy as np

# Add the eyetracker_base directory to sys.path to allow importing OrloskyPupilDetector
# Add the eyetracker_base directory to sys.path to allow importing OrloskyPupilDetector
if getattr(sys, 'frozen', False):
    # If frozen (PyInstaller), look in the same directory as the executable
    BASE_DIR = os.path.join(os.path.dirname(sys.executable), 'eyetracker_base')
else:
    # If running from source, use relative path
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../eyetracker_base'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    import OrloskyPupilDetector
except ImportError:
    raise ImportError(f"Could not import OrloskyPupilDetector from {BASE_DIR}")

class PupilAdapter:
    """
    Adapter for the OrloskyPupilDetector to normalize output for the Drawing Analysis app.
    """
    def __init__(self):
        self.last_target_size = (640, 480)

    def process_frame(self, frame):
        """
        Process a single frame and return the pupil ellipse.
        
        Args:
            frame (np.ndarray): BGR image frame.
            
        Returns:
            dict: Normalized pupil data containing:
                - 'center': (x, y)
                - 'axes': (minor, major)
                - 'angle': float (degrees)
                - 'confidence': float (0.0 to 1.0) - currently just 1.0 if found
                - 'raw_rect': The original rotated rect from the detector
            Returns None if no pupil is found.
        """
        if frame is None:
            return None

        # The Orlosky detector returns a rotated_rect: ((center_x, center_y), (width, height), angle)
        # Or ((0,0), (0,0), 0) if nothing found.
        h, w = frame.shape[:2]
        target_w = 320 if w <= 320 else 640
        target_h = 240 if h <= 240 else 480
        self.last_target_size = (target_w, target_h)

        try:
            rotated_rect = OrloskyPupilDetector.process_frame(frame, width=target_w, height=target_h)
        except Exception as e:
            print(f"Error in OrloskyPupilDetector: {e}")
            return None
        except Exception as e:
            print(f"Error in OrloskyPupilDetector: {e}")
            return None

        if rotated_rect is None:
            return None

        center, axes, angle = rotated_rect
        
        # Check if empty result or suspiciously small/flat
        if center == (0, 0) and axes == (0, 0):
            return None
            
        # Basic confidence check
        confidence = 1.0
        if axes[0] < 5 or axes[1] < 5: # Too small
            confidence = 0.0
        elif angle == 0 and center == (0,0): # Likely default
            confidence = 0.0
        
        return {
            'center': center,
            'axes': axes,
            'angle': angle,
            'confidence': confidence,
            'raw_rect': rotated_rect
        }
