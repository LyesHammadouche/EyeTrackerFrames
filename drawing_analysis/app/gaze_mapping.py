import math
import time
import numpy as np

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        Initialize the One Euro Filter.
        min_cutoff: Min cutoff frequency in Hz (lower = more smoothing, more lag)
        beta: Speed coefficient (higher = less lag for fast movement)
        """
        self.t_prev = t0
        self.x_prev = x0
        self.dx_prev = dx0
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        """
        Filter a single value.
        """
        t_e = t - self.t_prev
        
        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e if t_e > 0 else 0
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        
        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat

class GazeMapper:
    def __init__(self, calibration):
        self.calibration = calibration
        self.filter_x = None
        self.filter_y = None
        
        # Filter parameters
        self.min_cutoff = 0.5 # Hz
        self.beta = 0.1       # Speed coefficient
        
        # ROI / Canvas definition
        # (x, y, w, h) in Scene coordinates
        self.roi_scene = (0, 0, 640, 480) 
        # Physical dimensions in mm (e.g. A3 = 420x297)
        self.physical_dim = (420.0, 297.0) 

    def set_roi(self, x, y, w, h, physical_w, physical_h):
        self.roi_scene = (x, y, w, h)
        self.physical_dim = (physical_w, physical_h)

    def map_to_scene(self, pupil_center, timestamp):
        """
        Map pupil (eye cam) -> Scene coordinates (video feed pixels).
        Returns smoothed (x, y) in Scene Space.
        """
        if pupil_center is None:
            return None
            
        # 1. Map to Scene Coordinates (using simple calibration for now, or polynomial)
        # For now, we assume calibration returns SCENE coordinates 0-640, 0-480
        mapped_point = self.calibration.map_point(pupil_center)
        
        if mapped_point is None:
            return None
            
        x, y = mapped_point
        
        # 2. X-Inversion / Mirroring Check (if needed)
        # This should be handled by calibration, but we can force it here if flag is set
        # For now, assuming calibration provides correct raw direction
        
        # 3. Apply Smoothing
        if self.filter_x is None:
            self.filter_x = OneEuroFilter(timestamp, x, min_cutoff=self.min_cutoff, beta=self.beta)
            self.filter_y = OneEuroFilter(timestamp, y, min_cutoff=self.min_cutoff, beta=self.beta)
            return (x, y)
            
        x_smooth = self.filter_x(timestamp, x)
        y_smooth = self.filter_y(timestamp, y)
        
        return (x_smooth, y_smooth)

    def map_scene_to_canvas(self, scene_xy):
        """
        Map Scene (x, y) -> Canvas/Physical (mm_x, mm_y).
        Based on the defined ROI.
        """
        if scene_xy is None: 
            return None
            
        sx, sy = scene_xy
        rx, ry, rw, rh = self.roi_scene
        pw, ph = self.physical_dim
        
        # Normalized position within ROI
        norm_x = (sx - rx) / rw
        norm_y = (sy - ry) / rh
        
        # Map to physical dimensions
        mm_x = norm_x * pw
        mm_y = norm_y * ph
        
        return (mm_x, mm_y)
        
    def map_gaze(self, pupil_center, timestamp):
        """
        Legacy wrapper: Returns Scene Coordinates for now to maintain compatibility 
        until UI is fully updated to distinguish Scene vs Canvas.
        """
        return self.map_to_scene(pupil_center, timestamp)
