import numpy as np
import json
import os
import cv2

class Calibration:
    def __init__(self):
        self.calibration_points_pupil = [] # List of (pupil_x, pupil_y)
        self.calibration_points_world = [] # List of (world_x, world_y) in mm
        
        self.drawing_width_mm = 297.0 # Default A4 width
        self.drawing_height_mm = 210.0 # Default A4 height
        
        self.coeffs_x = None
        self.coeffs_y = None
        self.is_calibrated = False
        self.rmse = 0.0

    def set_dimensions(self, width_mm, height_mm):
        self.drawing_width_mm = width_mm
        self.drawing_height_mm = height_mm

    def clear_points(self):
        self.calibration_points_pupil = []
        self.calibration_points_world = []
        self.is_calibrated = False

    def add_point(self, pupil_xy, world_xy):
        """
        Add a calibration pair: Pupil (x,y) -> World (mm_x, mm_y)
        """
        self.calibration_points_pupil.append(pupil_xy)
        self.calibration_points_world.append(world_xy)

    def compute_model(self):
        """
        Compute 2nd order polynomial regression model.
        x_world = c0 + c1*px + c2*py + c3*px*py + c4*px^2 + c5*py^2
        """
        if len(self.calibration_points_pupil) < 6:
            print("Need at least 6 points for calibration (preferably 9)")
            self.is_calibrated = False
            return

        # Prepare data matrices
        P = np.array(self.calibration_points_pupil) # N x 2
        W = np.array(self.calibration_points_world) # N x 2
        
        px = P[:, 0]
        py = P[:, 1]
        
        # Design matrix for 2nd order polynomial: [1, x, y, xy, x^2, y^2]
        # Shape: N x 6
        ones = np.ones_like(px)
        X = np.column_stack([ones, px, py, px*py, px**2, py**2])
        
        # Solve for coefficients: X * C = W
        # We solve for x and y separately
        wx = W[:, 0]
        wy = W[:, 1]
        
        # Ridge regression (regularized least squares) to prevent overfitting with few points
        # C = (X.T * X + alpha*I)^-1 * X.T * W
        alpha = 0.001
        
        try:
            # Solve for X coefficients
            self.coeffs_x = np.linalg.inv(X.T @ X + alpha * np.eye(6)) @ X.T @ wx
            
            # Solve for Y coefficients
            self.coeffs_y = np.linalg.inv(X.T @ X + alpha * np.eye(6)) @ X.T @ wy
            
            self.is_calibrated = True
            self.rmse = self.compute_rmse()
            print("Calibration successful (Polynomial)!")
            print(f"X Coeffs: {self.coeffs_x}")
            print(f"Y Coeffs: {self.coeffs_y}")
            print(f"RMSE: {self.rmse:.2f} mm")
            
        except np.linalg.LinAlgError as e:
            print(f"Calibration failed: {e}")
            self.is_calibrated = False

    def compute_rmse(self):
        """Calculate Root Mean Square Error of the model on training data."""
        if not self.is_calibrated:
            return 0.0
            
        errors = []
        for i, pupil in enumerate(self.calibration_points_pupil):
            target = self.calibration_points_world[i]
            predicted = self.map_point(pupil)
            if predicted:
                dist = np.sqrt((predicted[0] - target[0])**2 + (predicted[1] - target[1])**2)
                errors.append(dist)
        
        if not errors:
            return 0.0
            
        return np.sqrt(np.mean(np.array(errors)**2))

    def save_calibration(self, filename="calibration.json"):
        if not self.is_calibrated:
            print("Cannot save: Not calibrated")
            return
            
        data = {
            "coeffs_x": self.coeffs_x.tolist(),
            "coeffs_y": self.coeffs_y.tolist(),
            "width_mm": self.drawing_width_mm,
            "height_mm": self.drawing_height_mm,
            "rmse": self.rmse
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Calibration saved to {filename}")
        except Exception as e:
            print(f"Error saving calibration: {e}")

    def load_calibration(self, filename="calibration.json"):
        if not os.path.exists(filename):
            print(f"Calibration file {filename} not found")
            return False
            
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            self.coeffs_x = np.array(data["coeffs_x"])
            self.coeffs_y = np.array(data["coeffs_y"])
            self.drawing_width_mm = data.get("width_mm", 297.0)
            self.drawing_height_mm = data.get("height_mm", 210.0)
            self.rmse = data.get("rmse", 0.0)
            self.is_calibrated = True
            print(f"Calibration loaded from {filename} (RMSE: {self.rmse:.2f} mm)")
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False

    def map_point(self, pupil_xy):
        """
        Map a pupil point (x, y) to drawing space (mm).
        """
        if not self.is_calibrated or self.coeffs_x is None:
            return None

        px, py = pupil_xy
        
        # Construct feature vector: [1, x, y, xy, x^2, y^2]
        features = np.array([1, px, py, px*py, px**2, py**2])
        
        # Predict
        wx = np.dot(features, self.coeffs_x)
        wy = np.dot(features, self.coeffs_y)
        
        return (wx, wy)
