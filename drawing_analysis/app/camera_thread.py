from PySide6.QtCore import QThread, Signal
import cv2
import numpy as np

class CameraThread(QThread):
    frame_ready = Signal(object) # Emits (ret, frame)

    def __init__(self, source, exposure=None, format_mode='MJPG', width=640, height=480, fps=30, rotate_180=False, gamma=None, contrast=None, flip_h=False):
        super().__init__()
        self.source = source
        self.exposure = exposure
        self.format_mode = format_mode
        self.width = width
        self.height = height
        self.fps = fps
        self.rotate_180 = rotate_180
        self.gamma = gamma
        self.contrast = contrast
        self.flip_h = flip_h
        self.cap = None
        self.running = False
        self.requested_exposure = None
        self.requested_gamma = None
        self.requested_contrast = None
        self.requested_rotation = None
        self.requested_flip = None
        self.hardware_exposure_ok = None
        self._gamma_lut = None
        if self.gamma is not None:
            self._update_gamma_lut(self.gamma)

    def set_exposure(self, value):
        self.requested_exposure = value

    def set_gamma(self, value):
        self.requested_gamma = value

    def set_contrast(self, value):
        self.requested_contrast = value

    def set_rotation(self, value):
        self.requested_rotation = value

    def set_flip(self, value):
        self.requested_flip = value

    def _update_gamma_lut(self, gamma):
        if gamma == 0: gamma = 0.01 # Prevent div by zero
        # Gamma typically 0.1 to 5.0. Slider might be 0..100 mapping to 0.1..3.0
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self._gamma_lut = table

    def _apply_exposure(self, value):
        # Try to disable auto-exposure first (value depends on backend/driver)
        # DSHOW commonly uses 0.25 for manual, 0.75 for auto.
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        print(f"Updating Exposure to {value}")
        self.hardware_exposure_ok = self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
        print(f"Exposure Set Result: {self.hardware_exposure_ok}")
        self.exposure = value

    def run(self):
        if isinstance(self.source, str) and not self.source.isdigit():
             self.cap = cv2.VideoCapture(self.source)
        else:
             self.cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW) # Use DSHOW for faster camera access on Windows
        
        if self.cap.isOpened():
            print(f"Opening Camera {self.source} with Mode: {self.format_mode} {self.width}x{self.height}")
            
            # Reduce internal buffering to lower latency.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self.format_mode == 'MJPG':
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            elif self.format_mode == 'YUY2':
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))
                # Prefer raw YUY2 and convert explicitly to avoid striped frames.
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            elif self.format_mode == 'Auto':
                # Don't force FOURCC or Convert RGB, let driver decide
                pass
            
            # Set resolution explicitly
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Enforce requested FPS (camera may clamp to nearest supported)
            if self.fps is not None:
                try:
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                except cv2.error as exc:
                    print(f"Warning: failed to set FPS={self.fps} on camera {self.source}: {exc}")
            
            if self.exposure is not None and self.hardware_exposure_ok is False:
                self._apply_exposure(self.exposure)
            
            # Print actual settings for debugging
            actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
            print(f"Camera {self.source} opened: {actual_w}x{actual_h} @ {actual_fps}fps FOURCC={fourcc_str} Exposure={self.exposure}")
        
        if not self.cap.isOpened():
            self.running = False
            return

        self.running = True
        consecutive_failures = 0
        max_failures = 30
        while self.running:
            # Handle dynamic updates safely in the capture thread
            if self.requested_exposure is not None:
                self._apply_exposure(self.requested_exposure)
                self.requested_exposure = None
            
            if self.requested_gamma is not None:
                self.gamma = self.requested_gamma
                self._update_gamma_lut(self.gamma)
                # Also try hardware
                self.cap.set(cv2.CAP_PROP_GAMMA, self.gamma * 100) # backend dependent
                self.requested_gamma = None
                
            if self.requested_contrast is not None:
                self.contrast = self.requested_contrast
                # Try hardware
                self.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast)
                self.requested_contrast = None

            if self.requested_rotation is not None:
                self.rotate_180 = self.requested_rotation
                self.requested_rotation = None

            if self.requested_flip is not None:
                self.flip_h = self.requested_flip
                self.requested_flip = None

            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Loop if file
                if isinstance(self.source, str) and not self.source.isdigit():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"Camera {self.source} read failed {consecutive_failures} times, stopping.")
                        break
                    self.msleep(5)
                    continue
            else:
                consecutive_failures = 0
            
            # Convert raw YUY2 to BGR if needed.
            if self.format_mode == 'YUY2' and frame is not None:
                if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 2):
                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
                    except cv2.error as exc:
                        print(f"YUY2 convert failed: {exc}")

            if self.exposure is not None:
                # Map -13 (Dark) to 0 (Bright) -> Offset
                # Standard slider range was -13 to 0.
                # If value is -5, we might want to darken?
                # Actually, negative exposure usually means "shorter shutter", so darker image.
                # 0 means "longer shutter", brighter image.
                
                # If hardware fails, let's just add the value * 10 to brightness? 
                # -13 * 10 = -130 brightness. 0 = 0.
                beta = self.exposure * 10 
                # Alpha (contrast) could be 1.0
                
                # Only apply if it's likely the hardware didn't do enough (optional)
                # Or just apply it on top.
                if beta != 0:
                     frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)

            # --- Gamma Correction ---
            if self._gamma_lut is not None and frame is not None:
                frame = cv2.LUT(frame, self._gamma_lut)
            
            # --- Rotation ---
            if self.rotate_180 and frame is not None:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            # --- Flip ---
            if self.flip_h and frame is not None:
                frame = cv2.flip(frame, 1) # 1 = horizontal flip

            self.frame_ready.emit((ret, frame))

            # Small yield to keep UI responsive without throttling high FPS
            self.msleep(1)
        if self.cap is not None:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait()
