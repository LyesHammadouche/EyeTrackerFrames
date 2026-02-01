import numpy as np
import cv2

class HeatmapAccumulator:
    def __init__(self, width_mm, height_mm, resolution_mm=1.0):
        """
        Initialize heatmap grid.
        resolution_mm: Size of each grid cell in mm.
        """
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.resolution_mm = resolution_mm
        
        self.cols = int(np.ceil(width_mm / resolution_mm))
        self.rows = int(np.ceil(height_mm / resolution_mm))
        
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # For pixel-based mode (scene overlay)
        self.pixel_mode = False
        self.frame_width = 640
        self.frame_height = 480
        
    def set_frame_size(self, width, height):
        """Set frame size for pixel-based heatmap accumulation."""
        self.frame_width = width
        self.frame_height = height
        self.pixel_mode = True
        # Use lower resolution grid for performance (max 128x96 ~12K cells)
        # Gaussian blur in render_overlay smooths it to any output size
        self.cols = min(128, max(32, width // 20))
        self.rows = min(96, max(24, height // 20))
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        
    def add_point(self, x_mm, y_mm, weight=1.0):
        """
        Add a gaze point to the heatmap (mm coordinates).
        """
        if x_mm < 0 or x_mm >= self.width_mm or y_mm < 0 or y_mm >= self.height_mm:
            return # Out of bounds
            
        col = int(x_mm / self.resolution_mm)
        row = int(y_mm / self.resolution_mm)
        
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self.grid[row, col] += weight
    
    def add_point_px(self, x_px, y_px, weight=1.0):
        """
        Add a gaze point using pixel coordinates (for scene overlay).
        """
        if x_px < 0 or x_px >= self.frame_width or y_px < 0 or y_px >= self.frame_height:
            return
        
        col = int(x_px * self.cols / self.frame_width)
        row = int(y_px * self.rows / self.frame_height)
        
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self.grid[row, col] += weight
            
    def get_normalized_grid(self):
        """
        Return grid normalized to [0, 1].
        """
        max_val = np.max(self.grid)
        if max_val > 0:
            return self.grid / max_val
        return self.grid
        
    def get_smoothed_grid(self, blur_size=0):
        """
        Return grid normalized to [0, 1] with Gaussian Blur applied.
        """
        normalized = self.get_normalized_grid()
        
        if blur_size > 0:
            # Ensure odd kernel size
            ksize = int(blur_size)
            if ksize % 2 == 0:
                ksize += 1
            if ksize < 1: 
                ksize = 1
                
            # If kernel fits, apply it. If kernel is larger than image, OpenCV handles or we clamp.
            # Ideally we clamp to min(w, h)
            h, w = normalized.shape
            max_k = min(w, h)
            if max_k % 2 == 0: max_k -= 1
            
            if ksize > max_k:
                ksize = max_k
                
            if ksize > 1:
                normalized = cv2.GaussianBlur(normalized, (ksize, ksize), 0)
                
            # Re-normalize? Gaussian blur usually preserves range for 0-1 float if kernel sums to 1.
            # But peaks might drop. 
            # User wants shape, so re-normalizing might change height mapping. 
            # Better to keep smoothed values relative.
            
        return normalized

    def get_smoothed_grid_sigma(self, sigma=0.0):
        """
        Return grid normalized to [0, 1] with Gaussian Blur applied using Sigma.
        Better for low-resolution grids where kernel size stepping is too coarse.
        """
        normalized = self.get_normalized_grid()
        
        if sigma > 0.1:
            # Let OpenCV calculate kernel size from sigma
            # 0 means compute from sigma
            normalized = cv2.GaussianBlur(normalized, (0, 0), sigmaX=sigma, sigmaY=sigma)
            
        return normalized

    def render_overlay(self, width, height, alpha=0.5, blur_size=25):
        """
        Render heatmap as a colored BGRA overlay image.
        Uses scaling-consistent smoothing.
        
        Args:
            width: Output image width
            height: Output image height  
            alpha: Base alpha for blending (0-1)
            blur_size: Slider value representing visual blur strength (pixels on 1280w screen)
        """
        # 1. Calculate Sigma relative to a Reference Resolution (e.g., 1280px)
        # This breaks the dependency on the actual 'width' argument for the smoothing amount,
        # ensuring the data looks the same "relative to the viewport" regardless of window size,
        # but more importantly, aligning it with the export logic.
        
        # Approximate conversion: Sigma ~= 0.3 * ((ksize-1)*0.5 - 1) + 0.8
        # Let's simple treat the slider (blur_size) as "Target Sigma * 10" or similar?
        # Or just keep it as "Reference Pixels".
        # Sigma = 0.3 * pxradius. blur_size is roughly diameter? 
        # Let's say Sigma = blur_size / 6.0 (Rule of thumb: Radius = 3*Sigma, Diameter=6*Sigma)
        
        REF_WIDTH = 1280.0
        
        # If the grid is 100px wide, and screen is 1280px.
        # A 30px visual feature on screen is 30 * (100/1280) = 2.34px on grid.
        # Sigma for 30px feature -> ~5.0. 
        # Sigma for 2.34px feature -> ~0.4.
        
        scale = self.cols / REF_WIDTH
        
        # Interpret slider 'blur_size' as Visual Diameter in Pixels
        visual_sigma = blur_size / 6.0 
        grid_sigma = visual_sigma * scale
        
        # 2. Smooth the SOURCE Grid first (Congruency)
        # We smooth the data, THEN resize. This ensures the mesh (which is the data) 
        # matches the visual (which is the resized data).
        smoothed_grid = self.get_smoothed_grid_sigma(grid_sigma)
        
        # 3. Resize to output dimensions
        heatmap_resized = cv2.resize(smoothed_grid, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Normalize again after scale/blur helps keep colors popping
        max_val = np.max(heatmap_resized)
        if max_val > 0:
            heatmap_resized = heatmap_resized / max_val
        
        # Convert to 8-bit for colormap
        heatmap_8bit = (heatmap_resized * 255).astype(np.uint8)
        
        # Apply JET colormap (blue -> green -> yellow -> red)
        colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
        
        # Create alpha channel based on intensity (more intense = more visible)
        # Areas with no data should be fully transparent
        alpha_channel = (heatmap_resized * 255 * alpha).astype(np.uint8)
        
        # Merge into BGRA
        bgra = cv2.cvtColor(colored, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha_channel
        
        return bgra

    def reset(self):
        self.grid.fill(0)
    
    def save_png(self, filepath, width, height, alpha=0.8, blur_size=31):
        """
        Save heatmap as transparent PNG at specified resolution.
        
        Args:
            filepath: Output file path (should end with .png)
            width: Output width in pixels
            height: Output height in pixels
            alpha: Opacity of the heatmap (0-1)
            blur_size: Gaussian blur kernel size for smoothing
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            overlay = self.render_overlay(width, height, alpha=alpha, blur_size=blur_size)
            # Save as PNG with alpha channel
            cv2.imwrite(filepath, overlay)
            print(f"Heatmap saved to: {filepath}")
            return True
        except Exception as e:
            print(f"Error saving heatmap: {e}")
            return False
    
    def save_combined(self, filepath, background_image, alpha=0.6, blur_size=31):
        """
        Save heatmap overlaid on background image.
        
        Args:
            filepath: Output file path
            background_image: BGR numpy array (source image)
            alpha: Opacity of the heatmap overlay (0-1)
            blur_size: Gaussian blur kernel size
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            h, w = background_image.shape[:2]
            overlay = self.render_overlay(w, h, alpha=alpha, blur_size=blur_size)
            
            # Alpha blending
            alpha_channel = overlay[:, :, 3:4] / 255.0
            bgr_overlay = overlay[:, :, :3]
            
            # Blend onto background
            combined = (bgr_overlay * alpha_channel + background_image * (1 - alpha_channel)).astype(np.uint8)
            
            cv2.imwrite(filepath, combined)
            print(f"Combined heatmap saved to: {filepath}")
            return True
        except Exception as e:
            print(f"Error saving combined heatmap: {e}")
            return False
