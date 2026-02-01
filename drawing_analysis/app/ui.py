import sys
import warnings

# Suppress annoying Protobuf/Mediapipe warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
import cv2
import numpy as np
import time
import threading
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QComboBox, QGroupBox, QFormLayout, QSlider,
                               QCheckBox, QMessageBox, QDockWidget, QScrollArea, QSizePolicy, QFileDialog, QButtonGroup, QFrame)
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtCore import QEvent, QTimer, Qt, QThread, Signal, Slot, QRect, QSize, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QKeySequence, QShortcut, QFont, QFontInfo, QIcon

import os
import json
from camera_thread import CameraThread
from pupil_adapter import PupilAdapter
from heatmap import HeatmapAccumulator # Changed from Heatmap to HeatmapAccumulator to match existing usage
from gaze_mapping import GazeMapper
from session import Session
from export_ntopo import export_to_ntopology
from export_mesh import export_heightfield_obj # Changed from export_to_obj_mesh to export_heightfield_obj to match existing usage
from calibration import Calibration
from globe_fitting import GlobeFitter
from video_recorder import VideoRecorder
from styles import MODERN_THEME_QSS
from head_tracker import HeadTrackerThread
class AspectRatioLabel(QLabel):
    def __init__(self, text="", parent=None, ratio=4/3):
        super().__init__(text, parent)
        self.ratio = ratio
        self.ratio = ratio
        self.setMinimumSize(1, 1)
        
        # Policy: Preferred/Preferred with HeightForWidth
        # allows expansion in both directions while respecting the ratio.
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        
        self._pixmap = None

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        super().setPixmap(pixmap)
        self.update()

    def paintEvent(self, event):
        if not self._pixmap or self._pixmap.isNull():
            super().paintEvent(event)
            return

        # Systemic Fix: Draw manually to guarantee Aspect Ratio is ALWAYS respected.
        # This prevents "Squishing" regardless of what the Layout forces upon us.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Calculate target rect maintaining aspect ratio
        target_rect = QRect(0, 0, self.width(), self.height())
        scaled_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Center the image
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        
        painter.drawPixmap(x, y, scaled_pixmap)

    def setAspectRatio(self, w, h):
        if h > 0:
            self.ratio = w / h
            self.updateGeometry() # Notify layout system of new ratio preference
            self.update() # Trigger repaint

    def hasHeightForWidth(self):
        # Feature: "No Black Bars in Dock"
        # Tells the layout engine: "Please resize my height to match my width * ratio"
        return True

    def heightForWidth(self, width):
        return int(width / self.ratio)

    def sizeHint(self):
        # Default starting size (User requested ~400px height)
        # Assuming typical docked width ~300-400px
        w = 400
        h = int(w / self.ratio)
        return QSize(w, h)

class ReDockableWidget(QDockWidget):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.forced_ratio = None # w / h

    def set_aspect_ratio(self, ratio):
        self.forced_ratio = ratio

    def closeEvent(self, event):
        if self.isFloating():
            self.setFloating(False)
            event.ignore()
        else:
            super().closeEvent(event)

    def closeEvent(self, event):
        if self.isFloating():
            self.setFloating(False)
            event.ignore()
        else:
            super().closeEvent(event)

    def resizeEvent(self, event):
        # Feature: "Elegant Floating View" (Zero Black Bars)
        # If floating, snap the window size to the camera ratio.
        if self.isFloating() and self.forced_ratio:
            current_size = event.size()
            old_size = event.oldSize()
            
            w = current_size.width()
            h = current_size.height()
            
            # If this is the first resize (old_size invalid), default to width driving
            if not old_size.isValid():
                target_h = int(w / self.forced_ratio)
                if abs(h - target_h) > 2:
                    self.resize(w, target_h)
            else:
                # Determine which dimension changed MORE
                dw = abs(w - old_size.width())
                dh = abs(h - old_size.height())
                
                if dw > dh:
                    # Width Driven -> Snap Height
                    target_h = int(w / self.forced_ratio)
                    if abs(h - target_h) > 2:
                        self.resize(w, target_h)
                else:
                    # Height Driven -> Snap Width
                    target_w = int(h * self.forced_ratio)
                    if abs(w - target_w) > 2:
                        self.resize(target_w, h)
        
        super().resizeEvent(event)

class PupilWorker(QThread):
    pupil_ready = Signal(object)

    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self._lock = threading.Lock()
        self._frame = None
        self._running = False

    def set_frame(self, frame):
        if frame is None:
            return
        with self._lock:
            self._frame = frame

    def run(self):
        self._running = True
        while self._running:
            frame = None
            with self._lock:
                if self._frame is not None:
                    frame = self._frame
                    self._frame = None
            if frame is None:
                self.msleep(1)
                continue
            try:
                pupil_data = self.adapter.process_frame(frame)
            except Exception as exc:
                print(f"PupilWorker error: {exc}")
                pupil_data = None
            self.pupil_ready.emit(pupil_data)

    def stop(self):
        self._running = False
        self.wait()

class CalibrationOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating) # Don't steal focus!
        # Actually for calibration we want to click? No, we use 'C' key global shortcut or focus.
        # For gaze bubble, we definitely want click through.
        
        self.active_target = None # (x, y)
        self.gaze_point = None # (x, y)
        self.msg = ""
        
        # Load Custom Crosshair
        # Construct absolute path to ensure loading works regardless of CWD
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file) # .../drawing_analysis/app
        # traverse up: app -> drawing_analysis -> EyeTrackingFrame
        project_root = os.path.dirname(os.path.dirname(current_dir)) 
        
        # PRIORITIZE HARDCODED PATH based on user confirmation
        hardcoded_path = r"C:\Users\xdlye\Desktop\WIP\Ai Progs\EyeTrackingFrame\assets\interface_elements\calibrationCrossHair.png"
        
        # Try multiple potential paths to be robust
        potential_paths = [
            hardcoded_path, # Try this first!
            os.path.join(project_root, "assets", "interface_elements", "calibrationCrossHair.png"),
            os.path.join(project_root, "interface elements", "calibrationCrossHair.png"), # Fallback
            os.path.join(current_dir, "calibrationCrossHair.png") 
        ]
        
        self.crosshair_pixmap = None
        
        for p in potential_paths:
            if os.path.exists(p):
                self.crosshair_pixmap = QPixmap(p)
                if not self.crosshair_pixmap.isNull():
                    break
        
        if self.crosshair_pixmap is None:
             print(f"Warning: Could not load calibrationCrossHair.png from any expected location.")
        
    def set_target(self, x, y):
        self.active_target = (x, y)
        self.update()
        
    def set_gaze(self, x, y):
        self.gaze_point = (x, y)
        self.update()
        
    def set_message(self, msg):
        self.msg = msg
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw Target
        if self.active_target:
            x, y = self.active_target
            
            # Draw Custom Crosshair
            if self.crosshair_pixmap:
                # Responsive Scaling: 10% of min screen dimension, clamped between 40px and 150px
                base_dim = min(self.width(), self.height())
                target_size = int(max(40, min(150, base_dim * 0.15)))
                
                # Scale Image
                scaled_crosshair = self.crosshair_pixmap.scaled(
                    target_size, target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                
                # Center on Target
                cx = int(x - scaled_crosshair.width() / 2)
                cy = int(y - scaled_crosshair.height() / 2)
                
                painter.drawPixmap(cx, cy, scaled_crosshair)
            else:
                # Fallback drawing
                painter.setPen(QPen(QColor(255, 255, 0), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(int(x)-15, int(y)-15, 30, 30)
                
                painter.setBrush(QColor(0, 255, 0))
                painter.drawEllipse(int(x)-5, int(y)-5, 10, 10)
            
        # Draw Gaze Bubble
        if self.gaze_point:
            gx, gy = self.gaze_point
            # Semi-transparent red bubble
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 0, 0, 100))
            painter.drawEllipse(int(gx)-20, int(gy)-20, 40, 40)
            
            # Center dot
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(int(gx)-2, int(gy)-2, 4, 4)
            
        # Draw Message
        if self.msg:
            painter.setPen(QColor(0, 255, 255))
            font = painter.font()
            font.setPointSize(24)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignTop | Qt.AlignHCenter, self.msg)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drawing Analysis Eye Tracker")
        self.resize(1600, 900) # Increased to accommodate 600px docks
        
        # Apply Dark Theme
        # Apply Modern Theme
        self.setStyleSheet(MODERN_THEME_QSS)
        # Enable advanced docking (Nested + Tabbed)
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks)
        
        # Screens
        self.screens_info = [] # List of tuples: (QScreen, name, geometry, scale_factor)
        self.populate_screens()



        # --- Components ---
        self.pupil_adapter = PupilAdapter()
        self.pupil_worker = None
        self.calibration = Calibration()
        self.gaze_mapper = GazeMapper(self.calibration)
        self.heatmap = HeatmapAccumulator(297, 210, resolution_mm=2.0) # A4 default
        self.session = Session()
        self.video_recorder = VideoRecorder()
        
        self.video_recorder = VideoRecorder()
        
        # Custom gaze cursor image
        self._load_gaze_cursor_image()
        
        # Load Custom Crosshair for Scene Overlay (OpenCV)
        # Use hardcoded path prioritizing user confirmation
        hardcoded_path = r"C:\Users\xdlye\Desktop\WIP\Ai Progs\EyeTrackingFrame\assets\interface_elements\calibrationCrossHair.png"
        self.crosshair_img_cv = None
        
        if os.path.exists(hardcoded_path):
            # Load with Alpha Channel (IMREAD_UNCHANGED)
            self.crosshair_img_cv = cv2.imread(hardcoded_path, cv2.IMREAD_UNCHANGED)
            if self.crosshair_img_cv is None:
                print(f"Error: Failed to load crosshair CV2 from {hardcoded_path}")
            else:
                pass # Loaded successfully
        else:
            print(f"Warning: Crosshair image not found at {hardcoded_path}")
            
        # Load Eye-Logo (Placeholder)
        logo_path = r"C:\Users\xdlye\Desktop\WIP\Ai Progs\EyeTrackingFrame\assets\interface_elements\Eye-logo.png"
        self.logo_pixmap = None
        if os.path.exists(logo_path):
            self.logo_pixmap = QPixmap(logo_path)
        
        # Performance throttling
        self.heatmap_last_update = 0
        self.heatmap_overlay_cache = None
        self.heatmap_proxy_width = 320 # Default to 0.5K (Low Res Proxy)

        self.heatmap_blur_radius = 31  # Default Blur
        self.heatmap_opacity = 0.6     # Default Opacity

        # --- State ---
        self.thread_eye = None
        self.thread_scene = None
        self.head_tracker = None
        self.head_gaze_data = None # (yaw, pitch, x, y)
        self.head_recentered = False
        self.head_active = False # Face detected?
        self.head_calibration = Calibration() # Separate calibration model for head
        
        self.latest_eye_frame = None
        self.latest_scene_frame = None
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_calibrating = False
        self.calibration_step = 0
        self.calibration_targets = [] # List of (world_x, world_y)
        self.current_pupil = None
        self.current_gaze = None
        self.center_calibrated = False
        self.center_pupil = None
        self.cal_mode = None
        self.center_cal_step = 0
        self.gaze_path = [] # List of (x, y) tuples
        self.show_gaze_path = False
        self.record_video = False
        
        # Mapping State
        self.current_map_mode = "scene" # "scene" or "screen"
        self.calibration_overlay = CalibrationOverlay()
        
        # Gaze State
        self.scene_gaze = None  # (x, y) in Scene (0-640, 0-480)
        self.canvas_gaze = None # (x, y) in Physical mm
        self.gaze_sensitivity = 10.0 # Multiplier for pupil movement
        self.mirror_x = False   # Mirror X-axis for gaze mapping
        self.globe_radius = 40.0 # Eye Globe Radius in pixels (tuneable)
        self.globe_center = (160, 120) # Manual Center (Default to half of 320x240)
        self.cal_zero_vector = None # (dx, dy) Vector from globe center to pupil when looking straight ahead
        
        # Calibration Buffer
        # Calibration Buffer
        self.cal_buffer = [] 
        self.cal_collecting = False
        self.cal_samples_needed = 10 # Number of frames to average
        
        # Auto-Fit State
        self.globe_fitter = GlobeFitter()
        self.is_fitting_globe = False
        self.fit_start_time = 0
        self.fit_duration = 5.0 # Seconds to look around
        
        # Heatmap State
        self.heatmap_alpha = 0.6
        self.heatmap_blur_radius = 31
        self.heatmap_overlay_cache = None
        
        # Media Stimulus State
        self.media_mode = False # False=Camera, True=Image/Video
        self.media_image = None # numpy array for static image
        self.media_video_path = None
        self.media_video_cap = None  # cv2.VideoCapture object
        self.media_original_size = (640, 480) # (w, h) of loaded media
        self.media_playing = False
        self.media_paused = False
        self.media_frame_count = 0 
        self.media_current_frame_idx = 0
        self.media_video_fps = 30.0  # For seek bar time display
        self.last_valid_media_frame = None  # Store frame for display when paused
        self.scene_was_running = False  # Track if scene camera was running before media load
        
        # Multi-Image Support
        self.media_images = []  # List of loaded image paths
        self.media_image_index = 0  # Current image index
        # Head Tracking State (Initialized above)

        # --- UI Layout ---
        # --- UI Layout (Dock Architecture) ---
        
        # 1. Central Widget (Main View)
        self.central_container = QWidget()
        self.central_container.setObjectName("CentralWidget")
        self.setCentralWidget(self.central_container)
        
        main_layout = QVBoxLayout(self.central_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main Video View
        self.video_label = QLabel("No Message Loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000; color: #555;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(100, 100) # Allow shrinking so Docks can be wide
        # Enable mouse events
        self.video_label.mousePressEvent = self.video_mouse_press
        main_layout.addWidget(self.video_label)

        # 2. Left Dock: Tuning (Camera & Image Adjustments)
        self.dock_tuning, tuning_content = self.create_dock("", "DockTuning", Qt.LeftDockWidgetArea)
        self.dock_tuning.setMinimumWidth(220) # USER_REQUEST: Restore freedom, min 220px
        self.dock_tuning.topLevelChanged.connect(lambda floating: self.on_dock_floating(self.dock_tuning, floating))
        tuning_layout = QVBoxLayout(tuning_content)
        tuning_layout.setAlignment(Qt.AlignTop)
        
        # [HEADER LOGO & CREDITS]
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        
        # Logo
        self.logo_label_left = QLabel()
        self.logo_label_left.setAlignment(Qt.AlignCenter)
        try:
            if getattr(sys, 'frozen', False):
                base_app_path = os.path.dirname(sys.executable)
            else:
                base_app_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
            round_logo_path = os.path.join(base_app_path, "assets", "interface_elements", "Eye-fb-round.png")
            if os.path.exists(round_logo_path):
                pix_round = QPixmap(round_logo_path)
                self.logo_label_left.setPixmap(pix_round.scaledToHeight(50, Qt.SmoothTransformation))
        except Exception as e:
            print(f"Error loading Left Logo: {e}")
            
        header_layout.addWidget(self.logo_label_left)
        
        # Vertical Separator Line
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #555;")
        header_layout.addWidget(sep)
        
        # Credits Text
        credits_text = QLabel(
            "Vibe Code by Lyes Hammadouche<br>"
            "<span style='color:#888'>based on original code by Orlosky (MIT License)</span><br>"
            "<a href='https://github.com/JEOresearch/EyeTracker' style='color:#aaa'>Source: JEOresearch/EyeTracker</a>"
        )
        credits_text.setOpenExternalLinks(True)
        credits_text.setStyleSheet("font-size: 10px;")
        header_layout.addWidget(credits_text)
        header_layout.addStretch()
        
        tuning_layout.addLayout(header_layout)
        
        # 3. Left Dock: Eye View (Floatable)
        # 3. Left Dock: Eye View (Floatable, No Scroll check)
        self.dock_eye = ReDockableWidget("Eye View", self)
        self.dock_eye.setObjectName("DockEye")
        self.dock_eye.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.TopDockWidgetArea)
        eye_content = QWidget()
        eye_content.setObjectName("DockContent")
        eye_content.setStyleSheet("background-color: black;") # Eliminate Grey Void
        self.dock_eye.setWidget(eye_content)
        self.dock_eye.setWidget(eye_content)
        self.dock_eye.setWidget(eye_content)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_eye)
        # Auto-size on float
        self.dock_eye.topLevelChanged.connect(lambda floating: self.on_dock_floating(self.dock_eye, floating))
        # Default to 4:3, but will be updated by video frame size
        self.dock_eye.set_aspect_ratio(4.0/3.0) 
        
        eye_layout = QVBoxLayout(eye_content)
        eye_layout.setContentsMargins(0,0,0,0)
        # REMOVED: eye_layout.setAlignment(Qt.AlignCenter) -- Causes widget to collapse!
        
        self.eye_video_label = AspectRatioLabel("Eye View")
        self.eye_video_label.setAlignment(Qt.AlignCenter)
        self.eye_video_label.setStyleSheet("background-color: black; color: white;")
        # Fix: Ensure Minimum Size (User request: ~400px high initially, but min can be smaller)
        # Let's set a reasonable min size to prevent total collapse
        self.eye_video_label.setMinimumSize(220, 165) 
        # Policy is set in __init__ now (Preferred/Fixed+HFW)
        
        # INTERACTIVE GLOBE ADJUSTMENTS
        # Use Event Filter for robustness (replacing monkey-patching)
        self.eye_video_label.installEventFilter(self)
        self.eye_video_label.setMouseTracking(True) # Ensure move events
        self.dragging_globe = False
        self.last_globe_drag_pos = None
        
        eye_layout.addWidget(self.eye_video_label)

        # 3b. Left Dock: Head View (Floatable, No Scroll check)
        self.dock_head = ReDockableWidget("Head View", self)
        self.dock_head.setObjectName("DockHead")
        self.dock_head.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.TopDockWidgetArea)
        head_content = QWidget()
        head_content.setObjectName("DockContent")
        self.dock_head.setWidget(head_content)
        self.dock_head.setWidget(head_content)
        self.dock_head.setWidget(head_content)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_head)
        self.dock_head.topLevelChanged.connect(lambda floating: self.on_dock_floating(self.dock_head, floating))
        self.dock_head.set_aspect_ratio(4.0/3.0) # Enforce 4:3
        
        head_layout = QVBoxLayout(head_content)
        head_layout.setContentsMargins(0,0,0,0)
        # REMOVED: head_layout.setAlignment(Qt.AlignCenter) -- Causes widget to collapse!
        
        self.head_video_label = AspectRatioLabel("Head View")
        self.head_video_label.setAlignment(Qt.AlignCenter)
        self.head_video_label.setStyleSheet("background-color: black; color: white;")
        # Fix: Remove setScaledContents(True) as we now handle painting manually
        # Fix: Ensure Minimum Size so dock doesn't collapse
        self.head_video_label.setMinimumSize(100, 75) 
        # self.head_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Policy handled in __init__
        head_layout.addWidget(self.head_video_label)
        
        # MINIMIZE Eye View and Head View by default (no black screens at startup)
        self.dock_eye.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.dock_head.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        # Hide them initially - they will show when signal is detected
        self.dock_eye.hide()
        self.dock_head.hide()
        
        # 4. Right Dock: Workflow (Calibration, Session, Export)
        self.dock_workflow, workflow_content = self.create_dock("Workflow", "DockWorkflow", Qt.RightDockWidgetArea)
        self.dock_workflow.setMinimumWidth(220) # USER_REQUEST: Restore freedom
        self.dock_workflow.topLevelChanged.connect(lambda floating: self.on_dock_floating(self.dock_workflow, floating))
        workflow_layout = QVBoxLayout(workflow_content)
        workflow_layout.setAlignment(Qt.AlignTop)

        # [HEADER] Workflow Help Text
        help_text = QLabel(
            "<b>Workflow:</b><br>"
            "1. <b>Setup Cameras:</b> Select Eye and Scene cameras.<br>"
            "2. <b>Tune Image:</b> Adjust Exposure until pupil is clear.<br>"
            "3. <b>Adjust Globe:</b> Drag to center/Scroll to match eye.<br>"
            "4. <b>Calibrate:</b> Look at targets & press 'C' or use Buttons.<br>"
            "5. <b>Record:</b> Start session -> Export Data."
        )
        help_text.setWordWrap(True)
        # help_text.setStyleSheet("color: #aaa; font-size: 11px; padding: 5px; border-bottom: 1px solid #444; margin-bottom: 5px;")
        # Use a boxed style for importance
        help_text.setStyleSheet("""
            QLabel {
                background-color: #252525; 
                color: #ccc; 
                font-size: 11px; 
                padding: 8px; 
                border-radius: 4px; 
                border: 1px solid #3d3d3d;
                margin-bottom: 10px;
            }
        """)
        workflow_layout.addWidget(help_text)
        
        # --- Populate Docks ---
        
        # 1. Camera Setup
        # --- GROUP 1: DEVICE SOURCES ---
        source_layout = self.add_section(tuning_layout, "Device Sources", QFormLayout)


        self.scan_cam_btn = QPushButton("Refresh") # Shortened from "Refresh Devices"
        self.scan_cam_btn.clicked.connect(self.scan_and_populate_cameras)

        self.eye_cam_combo = QComboBox()
        self.scene_cam_combo = QComboBox()
        self.head_cam_combo = QComboBox() # New Head Source
        
        # Camera Mode Selector (for Eye Cam)
        self.cam_mode_combo = QComboBox()
        self.cam_mode_combo.addItem("Auto", "Auto")
        self.cam_mode_combo.addItem("High FPS", "low")
        self.cam_mode_combo.addItem("Std 640p", "std")
        self.cam_mode_combo.addItem("HD 720p", "high")
        self.cam_mode_combo.setCurrentIndex(0) 

        self.scene_enable_checkbox = QCheckBox("Scene Cam") # Shortened from "Enable Scene"
        self.scene_enable_checkbox.setChecked(True)
        self.scene_enable_checkbox.setToolTip("Activates the secondary 'Scene' camera to show your field of view.")

        source_layout.addRow(self.scan_cam_btn)
        source_layout.addRow("Eye:", self.eye_cam_combo)
        source_layout.addRow("Mode:", self.cam_mode_combo)
        source_layout.addRow("Scene:", self.scene_cam_combo)
        source_layout.addRow("", self.scene_enable_checkbox)
        
        # --- GROUP 2: IMAGE TUNING ---
        tuning_layout_inner = self.add_section(tuning_layout, "Image Tuning", QFormLayout)

        self.exposure_lbl = QLabel("Exposure: -5")
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(-13, 0)
        self.exposure_slider.setValue(0)
        self.exposure_slider.valueChanged.connect(self.update_exposure_label)
        self.exposure_slider.setToolTip("Adjust camera exposure. Lower values darken the image, helpful for isolating the pupil.")

        self.gamma_lbl = QLabel("Gamma: 1.0")
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(1, 50)
        self.gamma_slider.setValue(10)
        self.gamma_slider.valueChanged.connect(self.update_gamma_label)
        self.gamma_slider.setToolTip("Adjust gamma correction. Higher values can broaden the dark areas to help pupil detection.")

        # self.contrast_slider removed per user request

        self.rotate_checkbox = QCheckBox("Rotate 180")
        self.rotate_checkbox.setChecked(True)
        self.rotate_checkbox.stateChanged.connect(self.update_rotation)

        self.flip_h_checkbox = QCheckBox("Flip Horizontal")
        self.flip_h_checkbox.setChecked(True)
        self.flip_h_checkbox.stateChanged.connect(self.update_flip)

        tuning_layout_inner.addRow(self.exposure_lbl, self.exposure_slider)
        tuning_layout_inner.addRow(self.gamma_lbl, self.gamma_slider)
        # tuning_layout_inner.addRow(self.contrast_lbl, self.contrast_slider)
        tuning_layout_inner.addRow(self.exposure_lbl, self.exposure_slider)
        tuning_layout_inner.addRow(self.gamma_lbl, self.gamma_slider)
        # tuning_layout_inner.addRow(self.contrast_lbl, self.contrast_slider)
        tuning_layout_inner.addRow(self.rotate_checkbox, self.flip_h_checkbox)

        # --- GROUP 3: TRACKING SETTINGS ---
        feature_layout = self.add_section(tuning_layout, "Tracking Settings", QFormLayout)

        # Eye Start
        self.start_cam_btn = QPushButton("Start Cameras")
        # Eye Start
        self.start_cam_btn = QPushButton("Start Cameras")
        self.start_cam_btn.clicked.connect(self.toggle_camera)
        self.start_cam_btn.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 5px;")

        # Head Tracking
        self.head_enable_check = QCheckBox("Head Track") # Shortened
        self.head_enable_check.stateChanged.connect(self.toggle_head_tracker)
        self.head_enable_check.setToolTip("Enables head pose tracking using a front-facing webcam.")
        
        # Head Camera Selection
        self.head_cam_combo = QComboBox()
        self.head_cam_combo.setToolTip("Select front-facing camera for head tracking")
        # Will be populated by populate_cameras()
        
        # Head Mode - Button Group
        self.head_mode_group = QButtonGroup(self)
        self.btn_eye_only = QPushButton("Eye") # Shortened
        self.btn_eye_only.setCheckable(True)
        self.btn_eye_only.setChecked(True)
        self.btn_eye_only.setToolTip("Track using only eye movements (Pupil).")
        
        self.btn_head_only = QPushButton("Head") # Shortened
        self.btn_head_only.setCheckable(True)
        self.btn_head_only.setToolTip("Track using only head orientation.")
        
        self.btn_hybrid = QPushButton("Hybrid")
        self.btn_hybrid.setCheckable(True)
        self.btn_hybrid.setToolTip("Combine Eye and Head tracking for broader range.")
        
        self.head_mode_group.addButton(self.btn_eye_only)
        self.head_mode_group.addButton(self.btn_head_only)
        self.head_mode_group.addButton(self.btn_hybrid)
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.btn_eye_only)
        mode_layout.addWidget(self.btn_head_only)
        mode_layout.addWidget(self.btn_hybrid)
        
        self.cal_head_btn = QPushButton("Center Head Cursor")
        self.cal_head_btn.clicked.connect(self.calibrate_head_center)
        self.cal_head_btn.setEnabled(False)
        self.cal_head_btn.setToolTip("Reset the head tracker center to your current head position.")
        
        self.cal_head_9_btn = QPushButton("Calibrate Head (9 Pts)")
        self.cal_head_9_btn.clicked.connect(lambda: self.start_calibration_routine(mode="head"))
        self.cal_head_9_btn.setEnabled(False)
        self.cal_head_9_btn.setStyleSheet(self.cal_head_btn.styleSheet()) # Same style
        self.cal_head_9_btn.setToolTip("Calibrate head tracking with a 9-point grid.")
        
        self.head_status_lbl = QLabel("Head: Off")

        feature_layout.addRow(self.start_cam_btn)
        feature_layout.addRow(self.head_enable_check)
        feature_layout.addRow("Head Cam:", self.head_cam_combo)
        feature_layout.addRow("Mode:", mode_layout)
        feature_layout.addRow(self.cal_head_btn, self.head_status_lbl)
        feature_layout.addRow(self.cal_head_9_btn)
        feature_layout.addRow(self.cal_head_btn, self.head_status_lbl)
        feature_layout.addRow(self.cal_head_9_btn)

        # --- GROUP 4: GLOBE PARAMETERS (Advanced) ---
        model_layout = self.add_section(tuning_layout, "Globe Parameters", QFormLayout)
        
        self.globe_slider = QSlider(Qt.Horizontal)
        self.globe_slider.setRange(10, 400) # Limited to 400 (larger values = bad calibration)
        self.globe_slider.setValue(40)
        self.globe_slider.valueChanged.connect(self.update_globe_radius)
        self.globe_slider.setToolTip("Radius of the virtual eye model in pixels.")
        self.globe_lbl = QLabel("Radius: 40px")
        
        self.globe_x_slider = QSlider(Qt.Horizontal)
        self.globe_x_slider.setRange(-500, 1000) # Increased range
        self.globe_x_slider.setValue(160)
        self.globe_x_slider.valueChanged.connect(self.update_globe_center)
        self.globe_x_slider.setToolTip("Horizontal (X) position of the eye model center.")
        self.globe_x_lbl = QLabel("Center X: 160")
        
        self.globe_y_slider = QSlider(Qt.Horizontal)
        self.globe_y_slider.setRange(-500, 1000) # Increased range
        self.globe_y_slider.setValue(120)
        self.globe_y_slider.valueChanged.connect(self.update_globe_center)
        self.globe_y_slider.setToolTip("Vertical (Y) position of the eye model center.")
        self.globe_y_lbl = QLabel("Center Y: 120")
        
        model_layout.addRow(self.globe_lbl, self.globe_slider)
        model_layout.addRow(self.globe_x_lbl, self.globe_x_slider)
        model_layout.addRow(self.globe_y_lbl, self.globe_y_slider)
        model_layout.addRow(self.globe_x_lbl, self.globe_x_slider)
        model_layout.addRow(self.globe_y_lbl, self.globe_y_slider)

        # Eye View and Head View Labels are already created and added to docks above (lines ~280 & ~309)
        # We do NOT re-instantiate them here to preserve EventFilters and Layouts.
        
        # REMOVED Pupil Status Label per User Request
        
        # Head View Preview is also already handled.

        # 2. Calibration
        cal_layout = self.add_section(workflow_layout, "Calibration", QVBoxLayout)
        self.cal_status_lbl = QLabel("Status: Uncalibrated")
        
        # Row 1: Globe Tools
        cal_row1 = QHBoxLayout()
        self.auto_fit_btn = QPushButton("Globe Auto-Fit")
        self.auto_fit_btn.clicked.connect(self.start_auto_fit_globe)
        self.auto_fit_btn.setStyleSheet("color: #66ff66; font-weight: bold;") 
        self.auto_fit_btn.setToolTip("Automatically estimates Globe Radius and Center by asking you to look around.")
        
        self.reset_globe_btn = QPushButton("Reset")
        self.reset_globe_btn.clicked.connect(self.reset_globe_data)
        self.reset_globe_btn.setStyleSheet("color: #ff6666; font-weight: bold;")
        self.reset_globe_btn.setToolTip("Resets the globe model to default values.")
        
        cal_row1.addWidget(self.auto_fit_btn)
        cal_row1.addWidget(self.reset_globe_btn)
        
        cal_layout.addWidget(self.cal_status_lbl)
        cal_layout.addLayout(cal_row1)
        
        # Row 2: Calibration Action
        cal_row2 = QHBoxLayout()
        self.cal_btn_grid = QPushButton("Calibrate 9 Points (Ctrl+C)")
        self.cal_btn_grid.clicked.connect(lambda: self.start_calibration_routine(mode="grid"))
        self.cal_btn_grid.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cal_btn_grid.setMinimumHeight(40) # Taller
        self.cal_btn_grid.setToolTip("Starts a 9-point calibration sequence to map your gaze to the screen/scene. (Shortcut: Ctrl+C)")
        
        self.cal_capture_btn = QPushButton("Capture (C)")
        self.cal_capture_btn.setEnabled(False) # Only enable during calibration
        self.cal_capture_btn.clicked.connect(self.capture_calibration_point)
        # self.cal_capture_btn.setStyleSheet("background-color: #FFA500; font-weight: bold;")
        self.cal_capture_btn.setMinimumHeight(40)
        
        # Ratio 2:1
        cal_row2.addWidget(self.cal_btn_grid, 2)
        cal_row2.addWidget(self.cal_capture_btn, 1)

        cal_layout.addWidget(self.cal_status_lbl)
        cal_layout.addLayout(cal_row1)
        cal_layout.addLayout(cal_row2)
        
        # 3. Session & Media
        sess_layout = self.add_section(workflow_layout, "Session & Media", QVBoxLayout)
        
        # Media Stimulus
        media_layout = QHBoxLayout()
        self.load_media_btn = QPushButton("Load Pic(s)/Video")
        self.load_media_btn.clicked.connect(self.load_media_file)
        self.load_media_btn.setToolTip("Load an image or video file to use as a visual stimulus (e.g., for analyzing reactions to specific content).")
        
        self.reset_cam_btn = QPushButton("Reset Camera")
        self.reset_cam_btn.clicked.connect(self.reset_to_camera_source)
        self.reset_cam_btn.setEnabled(False)
        self.reset_cam_btn.setToolTip("Unload the media file and return to the live Scene Camera feed.")
        
        media_layout.addWidget(self.load_media_btn)
        media_layout.addWidget(self.reset_cam_btn)
        sess_layout.addLayout(media_layout)

        # Media Controls (Play/Pause/Stop)
        self.media_ctrl_layout = QHBoxLayout()
        self.btn_media_play = QPushButton("▶") # Icons
        self.btn_media_stop = QPushButton("■")
        self.btn_media_replay = QPushButton("↺")
        
        self.btn_media_play.clicked.connect(self.toggle_media_playback)
        self.btn_media_stop.clicked.connect(self.stop_media_playback)
        self.btn_media_replay.clicked.connect(self.replay_media)
        
        # Icons could be added here
        self.media_ctrl_layout.addWidget(self.btn_media_play) # Play/Pause
        self.media_ctrl_layout.addWidget(self.btn_media_stop)
        self.media_ctrl_layout.addWidget(self.btn_media_replay)
        
        # Hide initially, show only when video loaded?
        # Or just disable. Let's disable.
        self.btn_media_play.setEnabled(False)
        self.btn_media_stop.setEnabled(False)
        self.btn_media_replay.setEnabled(False)
        
        sess_layout.addLayout(self.media_ctrl_layout)
        
        # Video Seek Bar (Timeline)
        self.seek_layout = QHBoxLayout()
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 100)
        self.seek_slider.setValue(0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.sliderPressed.connect(self.on_seek_pressed)
        self.seek_slider.sliderReleased.connect(self.on_seek_released)
        self.seek_slider.valueChanged.connect(self.on_seek_changed)
        self.seek_time_lbl = QLabel("0:00 / 0:00")
        self.seek_time_lbl.setFixedWidth(100)
        self.seek_layout.addWidget(self.seek_slider)
        self.seek_layout.addWidget(self.seek_time_lbl)
        self.seek_slider.hide()
        self.seek_time_lbl.hide()
        sess_layout.addLayout(self.seek_layout)
        self._seeking = False  # Flag to prevent feedback loop during seek
        
        # Multi-Image Navigation
        self.img_nav_layout = QHBoxLayout()
        self.btn_img_prev = QPushButton("◀ Prev")
        self.btn_img_next = QPushButton("Next ▶")
        self.lbl_img_counter = QLabel("1 / 1")
        self.lbl_img_counter.setAlignment(Qt.AlignCenter)
        self.btn_img_prev.clicked.connect(self.prev_image)
        self.btn_img_next.clicked.connect(self.next_image)
        self.img_nav_layout.addWidget(self.btn_img_prev)
        self.img_nav_layout.addWidget(self.lbl_img_counter)
        self.img_nav_layout.addWidget(self.btn_img_next)
        # Hide initially - only show when multiple images loaded
        self.btn_img_prev.hide()
        self.btn_img_next.hide()
        self.lbl_img_counter.hide()
        sess_layout.addLayout(self.img_nav_layout)

        # Calibration Media Button
        self.cal_btn_media = QPushButton("Calibrate 9 Points (Media) (Ctrl+C)")
        self.cal_btn_media.clicked.connect(lambda: self.start_calibration_routine(mode="grid"))
        self.cal_btn_media.setStyleSheet("height: 40px; font-weight: bold;")
        self.cal_btn_media.setToolTip("Start a 9-point calibration on top of the loaded media (Image/Video). Helpful if you want to calibrate specifically for the content area.")
        # Widget added later for strict ordering
        
        # Session Control
        self.start_sess_btn = QPushButton("▶ Start Recording Session")
        self.start_sess_btn.clicked.connect(self.toggle_session)
        self.start_sess_btn.setStyleSheet("height: 40px; font-size: 14px;")
        
        self.sess_info_lbl = QLabel("Time: 0.0s | Samples: 0")
        self.sess_info_lbl.setAlignment(Qt.AlignCenter)
        self.sess_info_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        
        # Options - Recording
        self.check_video = QPushButton("Record Video")
        self.check_video.setCheckable(True)
        self.check_video.clicked.connect(self.toggle_video_option)
        
        self.check_path = QPushButton("Gaze Path")
        self.check_path.setCheckable(True)
        self.check_path.clicked.connect(self.toggle_path_option)
        
        # Repurpose Sensitivity -> Smoothing Control (Moved from ROI)
        self.sens_lbl = QLabel("Smoothing: 1.0 Hz")
        self.sens_slider = QSlider(Qt.Horizontal)
        self.sens_slider.setRange(1, 50) # 0.1 Hz to 5.0 Hz
        self.sens_slider.setValue(10) # 1.0 Hz default
        self.sens_slider.valueChanged.connect(self.update_sensitivity)
        
        # Layout for smoothing
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(self.sens_lbl)
        smooth_layout.addWidget(self.sens_slider)
        
        # Gaze Cursor visibility (OFF by default to avoid self-consciousness)
        self.check_gaze_cursor = QPushButton("Gaze Cursor")
        self.check_gaze_cursor.setCheckable(True)
        self.check_gaze_cursor.setChecked(False)  # Hidden by default during recording
        self.check_gaze_cursor.clicked.connect(self.toggle_gaze_cursor_option)
        self.show_gaze_cursor = False  # State for gaze cursor visibility
        
        # Heatmap controls
        self.check_heatmap = QPushButton("Heatmap")
        self.check_heatmap.setCheckable(True)
        self.check_heatmap.clicked.connect(self.toggle_heatmap_option)
        self.check_heatmap.setToolTip("Toggle the real-time Heatmap overlay. Red areas indicate high gaze duration.")
        self.show_heatmap = False  # State for heatmap visualization
        self.record_heatmap = True  # Always record heatmap data (separate from display)
        
        # Heatmap Config Layout (Vertical Stack under button)
        hm_config_layout = QVBoxLayout()
        hm_config_layout.setSpacing(2)
        
        # Row 1: Resolution Buttons
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Res:"))
        
        self.heatmap_res_group = QButtonGroup(self)
        # self.heatmap_res_group.setExclusive(True) # Default is True
        
        resolutions = [("0.5K", 0), ("1K", 1), ("2K", 2), ("4K", 3)]
        for label, idx in resolutions:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setStyleSheet("padding: 2px 5px;")
            btn.setToolTip(f"Set Heatmap Resolution to {label}. Higher resolution = finer details but more CPU usage.")
            if idx == 0: btn.setChecked(True) # Default 0.5K
            self.heatmap_res_group.addButton(btn, idx)
            row1.addWidget(btn)
            
        # Connect Group ID to update function (passes int index)
        self.heatmap_res_group.idClicked.connect(self.update_heatmap_res)
        
        hm_config_layout.addLayout(row1)
        
        # Row 2: Blur
        self.lbl_blur_val = QLabel("Blur: 31")
        self.lbl_blur_val.setFixedWidth(70) 
        self.slider_heatmap_blur = QSlider(Qt.Horizontal)
        self.slider_heatmap_blur.setRange(5, 500) 
        self.slider_heatmap_blur.setValue(31)
        self.slider_heatmap_blur.valueChanged.connect(self._on_blur_changed_v2)
        self.slider_heatmap_blur.setToolTip("Adjust the spread/smoothness of heatmap points.")
        
        row2 = QHBoxLayout()
        row2.addWidget(self.lbl_blur_val)
        row2.addWidget(self.slider_heatmap_blur)
        hm_config_layout.addLayout(row2)
        
        # Row 3: Opacity
        self.lbl_opacity_val = QLabel("Opacity: 60%")
        self.lbl_opacity_val.setFixedWidth(80)
        self.slider_heatmap_opacity = QSlider(Qt.Horizontal)
        self.slider_heatmap_opacity.setRange(0, 100)
        self.slider_heatmap_opacity.setValue(60)
        self.slider_heatmap_opacity.valueChanged.connect(self._on_opacity_changed_v2)
        self.slider_heatmap_opacity.setToolTip("Adjust transparency of the Heatmap overlay.")
        
        row3 = QHBoxLayout()
        row3.addWidget(self.lbl_opacity_val)
        row3.addWidget(self.slider_heatmap_opacity)
        hm_config_layout.addLayout(row3)
        
        # sess_layout.addLayout(hm_config_layout) -- Removed early add to fix order
        
        self.btn_clear_heatmap = QPushButton("Clear Heatmap Data")
        self.btn_clear_heatmap.clicked.connect(self.clear_heatmap_data)
        self.btn_clear_heatmap.setStyleSheet("color: #ff6666;")
        self.btn_clear_heatmap.setToolTip("Permanently delete all recorded Heatmap history from memory.")
        
        # Organize Session Layout Order
        # Organize Session Layout Order
        sess_layout.addWidget(self.start_sess_btn)
        sess_layout.addWidget(self.sess_info_lbl) # Info under Start Button
        
        sess_layout.addWidget(self.cal_btn_media) # Calibrate Button Here
        
        sess_layout.addWidget(self.check_video)
        sess_layout.addWidget(self.check_gaze_cursor)
        sess_layout.addWidget(self.check_path)
        sess_layout.addLayout(smooth_layout)
        
        # Heatmap Section
        sess_layout.addWidget(self.check_heatmap)
        # Put Controls immediately under "Show Heatmap" (User Request)
        sess_layout.addLayout(hm_config_layout) 
        sess_layout.addWidget(self.btn_clear_heatmap)
        # sess_group.setLayout(sess_layout) --- Removed
        # workflow_layout.addWidget(sess_group) --- Removed

        # 4. Export & ROI
        # 4. Export & Settings
        exp_group = QGroupBox("Export & Settings")
        exp_layout = QVBoxLayout()
        
        # ROI Inputs
        roi_form = QFormLayout()
        # Removed phys_w_spin creation
        
        # Detect Screens
        self.screens_info = [] # List of tuples: (QScreen, name, geometry, scale_factor)
        self.populate_screens()

        # Mapping Mode
        self.map_mode_combo = QComboBox()
        # Add Screens
        for i, (screen, name, rect, scale) in enumerate(self.screens_info):
            label = f"{name} ({rect.width()}x{rect.height()} @ {scale:.1f}x)"
            self.map_mode_combo.addItem(label, f"screen_{i}")
        
        self.map_mode_combo.addItem("Scene Camera (640x480)", "scene") 
        self.map_mode_combo.setCurrentIndex(0) # Default to first screen (Primary usually)
        self.map_mode_combo.currentIndexChanged.connect(self.update_mapping_mode)
        
        # Manual numeric inputs for custom size
        self.roi_w_mm = 420.0
        self.roi_h_mm = 297.0
        
        self.mirror_check = QCheckBox("Mirror Gaze X")
        self.mirror_check.stateChanged.connect(self.toggle_mirror)
        
        # Input Removed: Paper Size (phys_w_spin)
        # Input Removed: Smoothing (Moved to Session)

        roi_form.addRow("Mapping Space:", self.map_mode_combo)
        roi_form.addRow(self.mirror_check)
        
        exp_layout.addLayout(roi_form)
        
        self.exp_json_btn = QPushButton("Export Recording Session (JSON)")
        self.exp_json_btn.clicked.connect(self.export_json)
        self.exp_json_btn.setToolTip("Exports the recorded session data to a JSON file compatible with nTopology.")
        
        self.exp_obj_btn = QPushButton("Export Mesh (OBJ)")
        self.exp_obj_btn.clicked.connect(self.export_obj)
        self.exp_obj_btn.setToolTip("Exports the current heatmap as a 3D Mesh (OBJ file).")
        
        self.exp_heatmap_btn = QPushButton("Export Heatmap")
        self.exp_heatmap_btn.clicked.connect(self.export_heatmap_png)
        self.exp_heatmap_btn.setToolTip("Saves the current heatmap visualization as a PNG image.")
        # self.exp_heatmap_btn.setStyleSheet("background-color: #B71C1C; color: white;") # Removed
        exp_layout.addWidget(self.exp_json_btn)
        exp_layout.addWidget(self.exp_obj_btn)
        exp_layout.addWidget(self.exp_heatmap_btn)
        exp_group.setLayout(exp_layout)
        workflow_layout.addWidget(exp_group)
        
        # --- SPACER to push content up ---
        workflow_layout.addStretch()
        
        # Help Text (Restored, without "Section Button" wrapper)
        # Help Text moved to top (lines 567+)
        
        # Signature Moved to Top Left (Done)

        # Initial populate using QMediaDevices if possible, else 0-3
        # Initial populate using QMediaDevices if possible, else 0-3
        self.scan_and_populate_cameras()

        # Shortcuts (System-wide in App)
        self.cal_shortcut = QShortcut(QKeySequence("C"), self)
        self.cal_shortcut.setContext(Qt.ApplicationShortcut)
        self.cal_shortcut.activated.connect(self.handle_c_shortcut)
        
        self.cal_start_shortcut = QShortcut(QKeySequence("Ctrl+C"), self)
        self.cal_start_shortcut.setContext(Qt.ApplicationShortcut)
        self.cal_start_shortcut.activated.connect(self.handle_ctrl_c_shortcut)

        # Set Initial Dock Widths to 400px (User Request - 1/3 reduction)
        # Use singleShot to enforce this AFTER the window is shown/layout is computed
        QTimer.singleShot(100, lambda: self.resizeDocks([self.dock_tuning, self.dock_workflow], [400, 400], Qt.Horizontal))

        # --- Apply Icons (User Request) ---
        self.apply_icon(self.start_cam_btn, "Start cameras.svg")
        self.apply_icon(self.cal_btn_grid, "Calibrate 9 Points.svg")
        self.apply_icon(self.auto_fit_btn, "Globe auto fit .svg")
        self.apply_icon(self.reset_globe_btn, "Reset.svg")
        
        self.apply_icon(self.start_sess_btn, "Start Recording Session.svg")
        self.apply_icon(self.load_media_btn, "load media.svg")
        self.apply_icon(self.reset_cam_btn, "Reset Camera.svg")
        
        self.apply_icon(self.check_video, "Record Video.svg")
        self.apply_icon(self.check_gaze_cursor, "Gaze Cursor.svg")
        self.apply_icon(self.check_path, "GazePath.svg")
        self.apply_icon(self.check_heatmap, "Heatmap.svg")
        
        self.apply_icon(self.btn_clear_heatmap, "Clear Heatmap Data.svg")
        self.apply_icon(self.exp_json_btn, "Export Recording Session.svg")
        self.apply_icon(self.exp_obj_btn, "Export Mesh.svg")
        self.apply_icon(self.exp_heatmap_btn, "Export Heatmap.svg")

    def populate_screens(self):
        screens = QApplication.screens()
        for i, screen in enumerate(screens):
            name = screen.name()
            geom = screen.geometry()
            scale = screen.devicePixelRatio()
            phys_w = int(geom.width() * scale)
            phys_h = int(geom.height() * scale)
            self.screens_info.append((screen, name, geom, scale))
            print(f"Detected Screen {i}: {name} Geom:{geom.width()}x{geom.height()} Scale:{scale} Phys:{phys_w}x{phys_h}")

    def get_media_display_rect(self):
        """Calculates the ROI of the media image within the video_label (screen coords)."""
        if not self.media_mode:
            return self.video_label.rect() # Fallback
            
        # Get Label Geometry
        lbl_w = self.video_label.width()
        lbl_h = self.video_label.height()
        
        # Get Media Geometry
        mw, mh = self.media_original_size
        if mw == 0 or mh == 0: return self.video_label.rect()
        
        # Calculate Aspect Fit
        scale = min(lbl_w / mw, lbl_h / mh)
        new_w = int(mw * scale)
        new_h = int(mh * scale)
        
        # Centered offsets
        off_x = (lbl_w - new_w) // 2
        off_y = (lbl_h - new_h) // 2
        
        # Return rect relative to Label (or Screen?)
        # Calibration Overlay is full-screen usually or over the video_label?
        # If map mode is "screen", overlay covers the whole screen.
        # So we need Global Coordinates if overlay is separate window.
        # BUT video_label is in the Main Window.
        # Let's return local label coordinates first.
        return QRect(off_x, off_y, new_w, new_h)

    def create_dock(self, title, object_name, area, scrollable=True):
        dock = ReDockableWidget(title, self)
        dock.setObjectName(object_name)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        if scrollable:
            # Scrollable Container
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            
            content = QWidget()
            content.setObjectName("DockContent") # For styling (Dark Background)
            scroll.setWidget(content)
            
            dock.setWidget(scroll)
        else:
            # Direct Widget (for video feeds to scale properly)
            content = QWidget()
            content.setObjectName("DockContent")
            dock.setWidget(content)
            
        self.addDockWidget(area, dock)
        return dock, content

    def apply_icon(self, widget, icon_name):
        """
        Helper to apply SVG icon.
        - Checkboxes: Standard setIcon (Icon Left, Text Left).
        - Buttons: Custom Background Image (Icon Fixed Left, Text Centered).
        """
        base_app_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        icon_path = os.path.join(base_app_path, "assets", "interface_elements", "SVG icons", icon_name)
        
        if not os.path.exists(icon_path):
            print(f"Warning: Icon not found: {icon_name}")
            return

        # Checkbox: Standard Icon
        if isinstance(widget, QCheckBox):
            widget.setIcon(QIcon(icon_path))
            widget.setIconSize(QSize(20, 20))
            return
            
        # PushButton: Advanced Styling (Icon Left, Text Center)
        if isinstance(widget, QPushButton):
            # 1. Create Cache Dir
            cache_dir = os.path.join(base_app_path, "temp_icons_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # 2. Render Icon to Pixmap (Scaled Down - User Request "20% smaller")
            # Original was 24, 20% smaller is ~19-20. Let's go with 19px for safety.
            base_icon = QIcon(icon_path)
            pix = base_icon.pixmap(19, 19)
            
            # 3. Save to PNG
            # Sanitize name: Replace spaces with underscores to avoid CSS URL issues
            safe_name = "".join([c if c.isalnum() or c in ('.', '_') else '_' for c in icon_name])
            while "__" in safe_name: safe_name = safe_name.replace("__", "_")
            cache_path = os.path.join(cache_dir, f"{safe_name}.png")
            pix.save(cache_path)
            
            # 4. Apply Stylesheet
            # Escape path? Forward slashes are safer for CSS URLs
            css_path = cache_path.replace("\\", "/")
            
            current_style = widget.styleSheet() or "" # Handle None
            if current_style and not current_style.strip().endswith(";"):
                current_style += ";"
            
            # Background-image, left positioned, no repeat. 
            # We explicitly set background-origin/clip to border-box or padding-box? 
            # Default is usually okay.
            # We use `background-image` so we don't overwrite `background-color`.
            # IMPORTANT: Do NOT use `QPushButton { ... }` wrapper if we are appending to existing raw properties (like 'color: red;').
            # Just append the properties directly.
            
            new_style = f"""
                background-image: url("{css_path}");
                background-position: left 15px center;
                background-repeat: no-repeat;
                text-align: center;
                padding-left: 0px;
            """
            
            widget.setStyleSheet(current_style + new_style)
            
            # Check if button is narrow? If text overlaps icon, we need logic.
            # User wants "Text Centered". 
            # If text is long and overlaps icon, that's bad.
            # But forcing padding-left shifts the center.
            # Let's assume buttons are wide enough (User said "Regular" not "Compact").
            
            widget.setStyleSheet(current_style + new_style)

    def add_section(self, parent_layout, title, layout_class=QVBoxLayout):
        """
        Creates a visual section with a green title and separator line.
        Returns the content layout to add widgets to.
        """
        # Container
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 15, 0, 5) # Spacing top/bottom
        container_layout.setSpacing(5)
        
        # Header (Label + Line)
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(5, 0, 5, 0)
        header_layout.setSpacing(10)
        
        # Title
        lbl = QLabel(title)
        lbl.setStyleSheet("color: #00e676; font-weight: bold; font-size: 13px; font-family: 'Segoe UI', sans-serif;")
        header_layout.addWidget(lbl)
        
        # Line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Plain)
        line.setStyleSheet("background-color: #00e676; border: none; min-height: 1px; max-height: 1px;")
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header_layout.addWidget(line)
        
        container_layout.addWidget(header_widget)
        
        # Content Layout
        content_layout = layout_class()
        # content_layout.setContentsMargins(5, 5, 5, 5) # Optional indentation
        container_layout.addLayout(content_layout)
        
        # --- Initialize Placeholder Logo on Video Label ---
        if hasattr(self, 'logo_pixmap') and self.logo_pixmap:
             # Create a black pixmap (1280x720 default)
             placeholder = QPixmap(1280, 720)
             placeholder.fill(Qt.black)
             
             painter = QPainter(placeholder)
             painter.setOpacity(0.2)
             
             # Scale logo to 50% height
             target_h = int(720 * 0.5)
             scaled_logo = self.logo_pixmap.scaledToHeight(target_h, Qt.SmoothTransformation)
             
             lx = (1280 - scaled_logo.width()) // 2
             ly = (720 - scaled_logo.height()) // 2
             
             painter.drawPixmap(lx, ly, scaled_logo)
             painter.end()
             
             self.video_label.setPixmap(placeholder.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            # Show Logo Placeholder instead of Text
            if hasattr(self, 'logo_pixmap') and self.logo_pixmap:
                 placeholder = QPixmap(1280, 720)
                 placeholder.fill(Qt.black)
                 painter = QPainter(placeholder)
                 painter.setOpacity(0.2)
                 target_h = int(720 * 0.5)
                 scaled_logo = self.logo_pixmap.scaledToHeight(target_h, Qt.SmoothTransformation)
                 lx = (1280 - scaled_logo.width()) // 2
                 ly = (720 - scaled_logo.height()) // 2
                 painter.drawPixmap(lx, ly, scaled_logo)
                 painter.end()
                 self.video_label.setPixmap(placeholder.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.video_label.clear()
             
        # 3. Create Dock Layout Structure
        parent_layout.addWidget(container)
        return content_layout

    def show_help_dialog(self):
        """Show calibration help dialog."""
        msg = (
            "Recommended Workflow\n"
            "1) Auto-Fit Globe: Run this 1 or 2 times until the globe is perfectly centered on the eye.\n"
            "   Move your eyes randomly. Watch the colored bars - aim for GREEN!\n"
            "2) Load Media: Load your stimulus (Image or Video).\n"
            "3) Calibrate 9 Points (Media): Essential for accuracy on specific media resolutions.\n"
            "   Look at each target and press 'C' to capture.\n\n"
            "Troubleshooting Pupil Detection\n"
            "- If the pupil is not detected (Red status), try adjusting 'Exposure' and 'Gamma'.\n"
            "- Ensure the IR camera focus is sharp and the eye is well-lit.\n"
            "- Use High FPS (e.g., 120fps) for best tracking stability.\n\n"
            "Advanced Tips\n"
            "- Use 'Reset Globe Data' if you start over or change lighting significantly.\n"
            "- Accuracy is best when the eye camera doesn't move after calibration."
        )
        QMessageBox.information(self, "Eye Tracking Workflow & Help", msg)


    def toggle_head_tracker(self, state):
        print(f"DEBUG: toggle_head_tracker called with state={state} (Type: {type(state)})")
        
        try:
            # Simplify check: Usually 2 is Checked, 0 is Unchecked. 
            # Treating as truthy/falsy might be safer or direct int comparison if needed.
            # Qt.Checked is 2.
            
            if state: # Truthy (2 is True)
                print("DEBUG: State is CHECKED (Truthy)")
                # IMMEDIATE FEEDBACK
                self.head_status_lbl.setText("Head: Init...")
                self.head_status_lbl.repaint() # Force update
                
                # Start
                print(f"DEBUG: Current self.head_tracker is: {self.head_tracker}")
                
                # If it exists but is not running, we might need to recreate or restart?
                if self.head_tracker is not None:
                    if self.head_tracker.isRunning():
                        print("DEBUG: HeadTracker is already running. Ignoring start request.")
                        self.head_status_lbl.setText("Head: Running")
                        return
                    else:
                        print("DEBUG: HeadTracker exists but not running. Recreating...")
                        self.head_tracker = None

                if self.head_tracker is None:
                    cam_idx = 0
                    if hasattr(self, 'head_cam_combo'):
                        data = self.head_cam_combo.currentData()
                        if isinstance(data, int):
                             cam_idx = data
                    
                    print(f"Starting Head Tracker on Camera Index: {cam_idx}")
                    self.head_tracker = HeadTrackerThread(camera_index=cam_idx) 
                    self.head_tracker.head_signal.connect(self.on_head_data)
                    self.head_tracker.start()
                    
                self.cal_head_btn.setEnabled(True)
                self.cal_head_9_btn.setEnabled(True)
                self.head_status_lbl.setText("Head: Starting...")
            else:
                # Stop
                if self.head_tracker:
                    self.head_tracker.stop()
                    self.head_tracker = None
                    self.cal_head_btn.setEnabled(False)
                    self.cal_head_9_btn.setEnabled(False)
                    self.head_status_lbl.setText("Head: Off")
                    self.head_active = False

        except Exception as e:
            print(f"CRITICAL ERROR in toggle_head_tracker: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Head Tracker Error", f"Failed to start Head Tracker:\n{str(e)}")
            self.head_status_lbl.setText("Head: Error")
            # Uncheck to reflect state
            self.head_enable_check.blockSignals(True)
            self.head_enable_check.setChecked(False)
            self.head_enable_check.blockSignals(False)

    def on_head_data(self, has_face, yaw, pitch, x, y, frame):
        self.head_active = has_face
        if has_face:
            self.head_gaze_data = (yaw, pitch, x,y)
            self.head_status_lbl.setText(f"Head: Active ({int(yaw)}, {int(pitch)})")
            self.head_status_lbl.setStyleSheet("color: #00e676; font-weight: bold;")
            
            # AUTO-SHOW: Display Head View when face is detected
            if hasattr(self, 'dock_head') and not self.dock_head.isVisible():
                self.dock_head.show()
        else:
            self.head_status_lbl.setText("Head: Lost")
            self.head_status_lbl.setStyleSheet("color: orange; font-weight: bold;")
            
            # AUTO-HIDE: Hide Head View when no face (avoid showing black square)
            if hasattr(self, 'dock_head') and self.dock_head.isVisible():
                self.dock_head.hide()
        
        # Display Head Frame
        if frame is not None and hasattr(self, 'head_video_label'):
            rgb_head = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_head.shape
            qt_head = QImage(rgb_head.data, w, h, ch * w, QImage.Format_RGB888)
            self.head_video_label.setPixmap(QPixmap.fromImage(qt_head).scaled(self.head_video_label.size(), Qt.KeepAspectRatio))
    
    def calibrate_head_center(self):
        if self.head_tracker and self.head_gaze_data:
            yaw, pitch, _, _ = self.head_gaze_data
            self.head_tracker.set_calibration(yaw, pitch)

    def load_media_file(self):
        # Support multiple file selection for images
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Media File(s)", "", "Images/Videos (*.png *.jpg *.jpeg *.mp4 *.avi *.mov)")
        if not file_paths:
            return
        
        # Track if scene camera was running for restart later
        self.scene_was_running = self.thread_scene is not None and self.thread_scene.isRunning()

        if self.thread_scene:
            self.thread_scene.stop()
            self.thread_scene.wait()
            self.thread_scene = None

        if self.media_video_cap:
            self.media_video_cap.release()
            self.media_video_cap = None
        
        # Clear previous multi-image state
        self.media_images = []
        self.media_image_index = 0
        self.media_image = None
        
        # Categorize files
        image_files = []
        video_file = None
        
        for fp in file_paths:
            ext = os.path.splitext(fp)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg']:
                image_files.append(fp)
            elif ext in ['.mp4', '.avi', '.mov']:
                video_file = fp
                break  # Only one video at a time
        
        # Priority: If a video is selected, load that. Otherwise load images.
        if video_file:
            # Video mode
            self.media_video_cap = cv2.VideoCapture(video_file)
            if not self.media_video_cap.isOpened():
                QMessageBox.warning(self, "Error", "Could not load video.")
                return
            w = int(self.media_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.media_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.media_original_size = (w, h)
            self.media_image = None
            self.media_video_path = video_file
            self.media_frame_count = int(self.media_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.media_video_fps = self.media_video_cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            # BUG FIX #1: Enable video controls when video is loaded
            self.btn_media_play.setEnabled(True)
            self.btn_media_stop.setEnabled(True)
            self.btn_media_replay.setEnabled(True)
            self.btn_media_play.setText("Play")
            self.media_playing = False
            self.media_paused = False
            
            # Setup seek bar
            self.seek_slider.setRange(0, max(1, self.media_frame_count - 1))
            self.seek_slider.setValue(0)
            self.seek_slider.setEnabled(True)
            self.seek_slider.show()
            self.seek_time_lbl.show()
            self._update_seek_label(0)
            
            # BUG FIX #3: Show first frame immediately
            ret, frame = self.media_video_cap.read()
            if ret:
                self.last_valid_media_frame = frame.copy()
                self.media_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
            
            # Hide multi-image navigation for video
            self.btn_img_prev.hide()
            self.btn_img_next.hide()
            self.lbl_img_counter.hide()
            
            self.load_media_btn.setText(f"Loaded: {os.path.basename(video_file)}")
            
        elif image_files:
            # Image mode (single or multiple)
            self.media_images = image_files
            self.media_image_index = 0
            self._load_current_image()
            
            # Disable video controls for images
            self.btn_media_play.setEnabled(False)
            self.btn_media_stop.setEnabled(False)
            self.btn_media_replay.setEnabled(False)
            
            # Hide seek bar for images
            self.seek_slider.hide()
            self.seek_time_lbl.hide()
            
            # Show/hide multi-image navigation based on count
            if len(self.media_images) > 1:
                self.btn_img_prev.show()
                self.btn_img_next.show()
                self.lbl_img_counter.show()
                self._update_image_counter()
            else:
                self.btn_img_prev.hide()
                self.btn_img_next.hide()
                self.lbl_img_counter.hide()
            
            self.load_media_btn.setText(f"Loaded: {len(self.media_images)} image(s)")
        else:
            QMessageBox.warning(self, "Error", "No valid media files selected.")
            return

        self.media_mode = True
        self.reset_cam_btn.setEnabled(True)
        self.video_label.setText("")  # Clear text
        
        # Ensure UI timer is running to render the media
        if not self.timer.isActive():
            self.timer.start(30)
            
        # USER_REQUEST: Reset Heatmap & Start Calibration on new media load
        self.heatmap.reset()
        self.heatmap_overlay_cache = None # Force complete clear
        print("Media loaded -> Auto-starting 9-point calibration")
        # Delay slightly to ensure UI update? No, direct call is fine.
        QTimer.singleShot(100, lambda: self.start_calibration_routine(mode="grid"))
    
    def _load_current_image(self):
        """Load the current image from media_images list."""
        if not self.media_images or self.media_image_index >= len(self.media_images):
            return
        
        path = self.media_images[self.media_image_index]
        self.media_image = cv2.imread(path)
        if self.media_image is None:
            QMessageBox.warning(self, "Error", f"Could not load image: {os.path.basename(path)}")
            return
        h, w = self.media_image.shape[:2]
        self.media_original_size = (w, h)
        self.media_video_cap = None
    
    def _update_image_counter(self):
        """Update the image counter label."""
        total = len(self.media_images)
        current = self.media_image_index + 1
        self.lbl_img_counter.setText(f"{current} / {total}")
        # Update prev/next button states
        self.btn_img_prev.setEnabled(self.media_image_index > 0)
        self.btn_img_next.setEnabled(self.media_image_index < total - 1)
    
    def prev_image(self):
        """Navigate to previous image."""
        if self.media_image_index > 0:
            self.media_image_index -= 1
            self._load_current_image()
            self._update_image_counter()
    
    def next_image(self):
        """Navigate to next image."""
        if self.media_image_index < len(self.media_images) - 1:
            self.media_image_index += 1
            self._load_current_image()
            self._update_image_counter()
    
    def _update_seek_label(self, frame_idx):
        """Update seek bar time label."""
        if self.media_video_fps <= 0:
            self.media_video_fps = 30.0
        current_sec = frame_idx / self.media_video_fps
        total_sec = self.media_frame_count / self.media_video_fps
        self.seek_time_lbl.setText(f"{int(current_sec//60)}:{int(current_sec%60):02d} / {int(total_sec//60)}:{int(total_sec%60):02d}")
    
    def on_seek_pressed(self):
        """Called when user starts dragging the seek slider."""
        self._seeking = True
    
    def on_seek_released(self):
        """Called when user releases the seek slider."""
        if self.media_video_cap:
            frame_idx = self.seek_slider.value()
            self.media_video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            # Read the frame at new position
            ret, frame = self.media_video_cap.read()
            if ret:
                self.last_valid_media_frame = frame.copy()
                # If not playing, reset position for next play
                if not self.media_playing:
                    self.media_video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._seeking = False
    
    def on_seek_changed(self, value):
        """Called when seek slider value changes."""
        self._update_seek_label(value)


    def reset_to_camera_source(self):
        self.media_mode = False
        self.media_playing = False
        self.media_paused = False
        
        if self.media_video_cap:
            self.media_video_cap.release()
            self.media_video_cap = None
        self.media_image = None
        self.media_images = []
        self.media_image_index = 0
        self.last_valid_media_frame = None
        
        self.reset_cam_btn.setEnabled(False)
        self.load_media_btn.setText("Load Media")
        
        # Disable Media Controls
        self.btn_media_play.setEnabled(False)
        self.btn_media_stop.setEnabled(False)
        self.btn_media_replay.setEnabled(False)
        
        # Hide seek bar
        self.seek_slider.hide()
        self.seek_time_lbl.hide()
        self.seek_slider.setEnabled(False)
        
        # Hide multi-image navigation
        self.btn_img_prev.hide()
        self.btn_img_next.hide()
        self.lbl_img_counter.hide()
        
        # BUG FIX #2: Restart scene camera if it was running before media load
        if self.scene_was_running and self.thread_eye is not None and self.thread_eye.isRunning():
            # Get scene camera index
            scene_data = self.scene_cam_combo.currentData()
            scene_source = 1
            if scene_data is not None:
                scene_source = int(scene_data)
            
            # Prevent opening same camera as eye
            eye_data = self.eye_cam_combo.currentData()
            if eye_data != "file" and scene_source != int(eye_data):
                self.thread_scene = CameraThread(scene_source, width=640, height=480, fps=30, format_mode='MJPG')
                self.thread_scene.frame_ready.connect(self.on_scene_frame)
                self.thread_scene.start()
                print(f"Restarted scene camera on index {scene_source}")
        
        self.scene_was_running = False


    def toggle_media_playback(self):
        if not self.media_video_cap: return
        
        if self.media_paused or not self.media_playing:
            # Play
            self.media_playing = True
            self.media_paused = False
            self.btn_media_play.setText("Pause")
        else:
            # Pause
            self.media_paused = True
            self.btn_media_play.setText("Play")

    def stop_media_playback(self):
        if not self.media_video_cap: return
        self.media_playing = False
        self.media_paused = False
        self.media_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.btn_media_play.setText("Play")
        # In stop state, we might show black or first frame?
        # Let's show first frame (Replay effectively resets to 0 and pauses?)
        # Or just clear? "Stop" usually means reset position.
        
        # Let's read first frame to have something to show
        ret, frame = self.media_video_cap.read()
        if ret:
            self.last_valid_media_frame = frame
            # Reset pos for next play
            self.media_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
    def replay_media(self):
        if not self.media_video_cap: return
        self.media_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.media_playing = True
        self.media_paused = False
        self.btn_media_play.setText("Pause")

    def populate_cameras_initial(self):
        self.scan_and_populate_cameras()
        
    def update_exposure_label(self, value):
        self.exposure_lbl.setText(f"Exposure: {value}")
        # Live update if thread is running
        if self.thread_eye is not None and self.thread_eye.isRunning():
            self.thread_eye.set_exposure(value)

    def update_gamma_label(self, value):
        gamma = value / 10.0
        self.gamma_lbl.setText(f"Gamma: {gamma:.1f}")
        if self.thread_eye is not None and self.thread_eye.isRunning():
            self.thread_eye.set_gamma(gamma)

    def update_contrast_label(self, value):
        contrast = value / 10.0
        self.contrast_lbl.setText(f"Contrast: {contrast:.1f}")
        if self.thread_eye is not None and self.thread_eye.isRunning():
            self.thread_eye.set_contrast(contrast)

    def update_rotation(self, state):
        rotate = state == Qt.Checked
        if self.thread_eye is not None and self.thread_eye.isRunning():
            self.thread_eye.set_rotation(rotate)

    def update_flip(self, state):
        flip = state == Qt.Checked
        if self.thread_eye is not None and self.thread_eye.isRunning():
            self.thread_eye.set_flip(flip)

    def toggle_mirror(self, state):
        self.mirror_x = (state == Qt.Checked)

    def update_sensitivity(self, val):
        # Map 1-50 to 0.1 - 5.0 Hz (Min Cutoff)
        # Lower Hz = More Smoothing (Lag), Higher Hz = More Responsive (Jitter)
        hz = val / 10.0
        if hz < 0.1: hz = 0.1
        
        self.gaze_mapper.min_cutoff = hz
        # Update existing filters if active
        if self.gaze_mapper.filter_x: self.gaze_mapper.filter_x.min_cutoff = hz
        if self.gaze_mapper.filter_y: self.gaze_mapper.filter_y.min_cutoff = hz
            
        self.sens_lbl.setText(f"Smoothing: {hz:.1f} Hz")

    def update_mapping_mode(self, index):
        data = self.map_mode_combo.itemData(index)
        self.current_map_mode = data
        print(f"Mapping Mode Switched to: {data}")
        
        if data.startswith("screen_"):
            idx = int(data.split("_")[1])
            if idx < len(self.screens_info):
                screen, name, rect, scale = self.screens_info[idx]
                
                # Move overlay to that screen
                self.calibration_overlay.setParent(None) # Detach?
                self.calibration_overlay.close() # Reset
                
                # Create/Move
                # To move, we set geometry to the screen geometry (which includes top/left offset)
                self.calibration_overlay.setGeometry(rect) 
                
                # Move overlay to that screen
            if hasattr(self, 'calibration_overlay'):
                if self.calibration_overlay.isVisible():
                    self.calibration_overlay.windowHandle().setScreen(screen)
                    self.calibration_overlay.setGeometry(screen.geometry())
                else:
                    # If not visible, just setting geometry might be enough for next show,
                    # but setScreen requires a handle. If handle is None, we skip setScreen.
                    handle = self.calibration_overlay.windowHandle()
                    if handle:
                        handle.setScreen(screen)
                    # self.calibration_overlay.setGeometry(screen.geometry()) # Maybe risky if not shown
 # or show() with geometry
                
                # Update current screen dimensions for ROI logic
                self.screen_width = rect.width()
                self.screen_height = rect.height()
                self.screen_scale = scale
                
        elif data == "scene":
            self.calibration_overlay.hide()
            self.screen_width = 640
            self.screen_height = 480
            self.screen_scale = 1.0

    def update_globe_radius(self, val):
        self.globe_radius = float(val)
        self.globe_lbl.setText(f"Globe Radius: {int(self.globe_radius)}px")

    def update_globe_center(self, val):
        gx = self.globe_x_slider.value()
        gy = self.globe_y_slider.value()
        self.globe_center = (gx, gy)
        self.globe_x_lbl.setText(f"Globe X: {gx}")
        self.globe_y_lbl.setText(f"Globe Y: {gy}")
        self.globe_y_lbl.setText(f"Globe Y: {gy}")

    @Slot(object)
    def on_pupil_data(self, pupil_data):
        self.current_pupil = pupil_data
        
        if self.current_pupil and self.current_pupil.get('confidence', 0) > 0.1:
            # self.pupil_status_lbl.setText("Pupil: FOUND")
            pass
        else:
            # self.pupil_status_lbl.setText("Pupil: LOST")
            pass

    def _normalize_eye_frame(self, frame):
        if frame is None:
            return None
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 2:
            return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
        return frame
    
    def _load_gaze_cursor_image(self):
        """Load custom gaze cursor images."""
        import os
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Load Round (Camera/Scene Mode)
        path_round = os.path.join(base_path, "assets", "interface_elements", "Eye-fb-round.png")
        if not os.path.exists(path_round):
            # Fallback for dev environment or flat structure
            path_round = os.path.join(base_path, "interface elements", "Eye-fb-round.png")
            
        if os.path.exists(path_round):
             self.gaze_cursor_round = cv2.imread(path_round, cv2.IMREAD_UNCHANGED)
        else:
             self.gaze_cursor_round = None

        # Load Long (Media Mode)
        path_long = os.path.join(base_path, "assets", "interface_elements", "Eye-fb-oval.png")
        if os.path.exists(path_long):
             self.gaze_cursor_long = cv2.imread(path_long, cv2.IMREAD_UNCHANGED)
        else:
             self.gaze_cursor_long = None
             
        # Cache dicts
        self._cursor_cache_round = {}
        self._cursor_cache_long = {}
    
    def _overlay_image_alpha(self, background, overlay, x, y):
        """
        Overlay an BGRA image onto a BGR background at position (x, y).
        Handles alpha blending and boundary clipping.
        """
        if overlay is None or overlay.shape[2] != 4:
            return
        
        h, w = overlay.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # Calculate centered position
        x1 = int(x - w // 2)
        y1 = int(y - h // 2)
        x2 = x1 + w
        y2 = y1 + h
        
        # Clip to background bounds
        if x1 < 0:
            overlay = overlay[:, -x1:]
            x1 = 0
        if y1 < 0:
            overlay = overlay[-y1:, :]
            y1 = 0
        if x2 > bg_w:
            overlay = overlay[:, :bg_w - x2]
            x2 = bg_w
        if y2 > bg_h:
            overlay = overlay[:bg_h - y2, :]
            y2 = bg_h
        
        if overlay.shape[0] <= 0 or overlay.shape[1] <= 0:
            return
        
        # Extract alpha channel and normalize
        alpha = overlay[:, :, 3:4] / 255.0
        bgr = overlay[:, :, :3]
        
        # Blend
        roi = background[y1:y2, x1:x2]
        if roi.shape[:2] == alpha.shape[:2]:
            background[y1:y2, x1:x2] = (bgr * alpha + roi * (1 - alpha)).astype(np.uint8)


    def scan_and_populate_cameras(self):
        self.scan_cam_btn.setText("Scanning...")
        QApplication.processEvents()
        
        self.eye_cam_combo.clear()
        self.scene_cam_combo.clear()
        if hasattr(self, 'head_cam_combo'):
            self.head_cam_combo.clear()
        
        devices = QMediaDevices.videoInputs()
        
        if not devices:
             # Fallback
             for i in range(4):
                 label = f"Camera {i}"
                 self.eye_cam_combo.addItem(label, i)
                 self.scene_cam_combo.addItem(label, i)
                 if hasattr(self, 'head_cam_combo'):
                     self.head_cam_combo.addItem(label, i)
        else:
            default_eye_index = 0
            default_scene_index = 0
            default_head_index = 0
            
            # Inventory
            for i, camera in enumerate(devices):
                desc = camera.description()
                label = f"{desc} (Index {i})"
                self.eye_cam_combo.addItem(label, i)
                self.scene_cam_combo.addItem(label, i)
                if hasattr(self, 'head_cam_combo'):
                    self.head_cam_combo.addItem(label, i)
                
                print(f"Discovered Camera {i}: {desc}")
                
                # Smart Defaults
                if "USB Camera" in desc: 
                    default_eye_index = i
                
                if "HD Camera" in desc or "Insta360" in desc: 
                    default_scene_index = i
                
                if "Integrated" in desc or "Webcam" in desc or "FaceTime" in desc:
                     default_head_index = i

            # Set Defaults with Conflict Avoidance
            self.eye_cam_combo.setCurrentIndex(default_eye_index)
            
            # Scene Logic
            if default_scene_index == default_eye_index and len(devices) > 1:
                 for i in range(len(devices)):
                     if i != default_eye_index:
                         default_scene_index = i
                         break
            self.scene_cam_combo.setCurrentIndex(default_scene_index)
            
            # Head Logic
            if hasattr(self, 'head_cam_combo'):
                used_indices = [default_eye_index, default_scene_index]
                if default_head_index in used_indices and len(devices) > 2:
                    for i in range(len(devices)):
                        if i not in used_indices:
                            default_head_index = i
                            break
                self.head_cam_combo.setCurrentIndex(default_head_index)

        self.eye_cam_combo.addItem("Video File", "file")
        self.scan_cam_btn.setText("Refresh Device List")

    def toggle_camera(self):
        if self.thread_eye is None and self.thread_scene is None:
            # --- Start Eye Camera ---
            eye_data = self.eye_cam_combo.currentData()
            eye_source = 0
            
            if eye_data == "file":
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
                eye_source = os.path.join(base_dir, "eyetracker_base/eye_test.mp4")
            else:
                eye_source = int(eye_data)

            # Determine Mode
            mode_data = self.cam_mode_combo.currentData()
            format_mode = 'MJPG'
            w, h = 640, 480
            fps = 30
            
            if mode_data == "Auto":
                format_mode = 'Auto'
                w, h = 640, 480
                fps = 30
            elif mode_data == "low":
                format_mode = 'YUY2'
                w, h = 320, 240
                fps = 120
            elif mode_data == "std":
                format_mode = 'MJPG'
                w, h = 640, 480
                fps = 30
            elif mode_data == "high":
                format_mode = 'MJPG'
                w, h = 1280, 720
                fps = 30
                
            exposure_val = self.exposure_slider.value()
            gamma_val = self.gamma_slider.value() / 10.0
            contrast_val = 1.0 # self.contrast_slider.value() / 10.0 (Removed)
            rotate_val = self.rotate_checkbox.isChecked()
            flip_val = self.flip_h_checkbox.isChecked()
            if self.flip_h_checkbox.isChecked(): # Ensure default is respected if not manually toggled? 
                # actually isChecked() reads the UI state, which we set to default True above.
                pass
            
            self.thread_eye = CameraThread(eye_source, exposure=exposure_val, format_mode=format_mode, width=w, height=h, fps=fps, rotate_180=rotate_val, gamma=gamma_val, contrast=contrast_val, flip_h=flip_val)
            self.thread_eye.frame_ready.connect(self.on_eye_frame)
            self.thread_eye.start()

            self.pupil_worker = PupilWorker(self.pupil_adapter)
            self.pupil_worker.pupil_ready.connect(self.on_pupil_data)
            self.pupil_worker.start()

            self.pupil_worker.start()
            
            # GIVE TIME FOR EYE CAMERA TO INITIALIZE (Prevent DSHOW Conflict)
            QApplication.processEvents()
            time.sleep(0.5)

            # --- Start Scene Camera ---
            # Safe parsing
            if self.scene_enable_checkbox.isChecked():
                scene_data = self.scene_cam_combo.currentData()
                scene_source = 1
                if scene_data is not None:
                     scene_source = int(scene_data)
                     
                # Prevent opening the same camera twice
                if isinstance(eye_source, int) and isinstance(scene_source, int):
                    if eye_source == scene_source:
                        print("ERROR: Eye and Scene Source are the same!")
                        self.start_cam_btn.setText("Error: Duplicate Camera Selection")
                        # Clean up eye thread we just started
                        if self.thread_eye:
                            self.thread_eye.stop()
                            self.thread_eye = None
                        return

                # Request Standard Resolution (640x480) per user request
                # Aspect ratio issues are handled by Qt.KeepAspectRatio in display
                self.thread_scene = CameraThread(scene_source, width=640, height=480, fps=30, format_mode='MJPG')
                self.thread_scene.frame_ready.connect(self.on_scene_frame)
                self.thread_scene.start()
            
            self.start_cam_btn.setText("Stop Cameras")
            self.cal_btn_grid.setEnabled(True)  # Enable calibration when cameras running
            self.timer.start(30) # UI update loop
        else:
            # Disable calibration button when cameras stop
            self.cal_btn_grid.setEnabled(False)
            
            # Reset calibration model logic
            if self.cal_mode == "grid":
                self.calibration.clear_points()
                self.center_calibrated = False 
            elif self.cal_mode == "center":
                self.center_calibrated = False
            elif self.cal_mode == "head":
                self.head_calibration.clear_points()
            if self.thread_eye:
                self.thread_eye.stop()
                self.thread_eye = None
            if self.pupil_worker:
                self.pupil_worker.stop()
                self.pupil_worker = None
            if self.thread_scene:
                self.thread_scene.stop()
                self.thread_scene = None
            
            self.start_cam_btn.setText("Start Cameras")
            self.video_label.clear()
            self.eye_video_label.clear()
            self.eye_video_label.setText("Eye View")

    def on_eye_frame(self, data):
        ret, frame = data
        if ret:
            self.latest_eye_frame = self._normalize_eye_frame(frame)
            h, w = frame.shape[:2]
            current_ratio = w / h
            
            # Optimize: Only update aspect ratio if it changes significantly
            if not hasattr(self, 'last_eye_ratio') or self.last_eye_ratio is None or abs(self.last_eye_ratio - current_ratio) > 0.01:
                self.last_eye_ratio = current_ratio
                # Update Label (Handles Docked sizing via heightForWidth)
                self.eye_video_label.setAspectRatio(w, h)
                # Update Dock (Handles Floating sizing via resizeEvent)
                if hasattr(self, 'dock_eye'):
                    self.dock_eye.set_aspect_ratio(current_ratio)
                    # If floating, trigger a resize to snap immediately
                    if self.dock_eye.isFloating():
                        dw = self.dock_eye.width()
                        dh = int(dw / current_ratio)
                        self.dock_eye.resize(dw, dh)
                        
            if self.pupil_worker is not None and self.pupil_worker.isRunning():
                self.pupil_worker.set_frame(self.latest_eye_frame)

    def on_scene_frame(self, data):
        ret, frame = data
        if ret:
            self.latest_scene_frame = frame

    def update_frame(self):
        # --- Process Eye Frame ---
        eye_frame = self.latest_eye_frame
        if eye_frame is not None:
            # AUTO-SHOW: Display Eye View when camera signal is present
            if hasattr(self, 'dock_eye') and not self.dock_eye.isVisible():
                self.dock_eye.show()
            
            # Resize for consistency with detector if needed
            target_w, target_h = self.pupil_adapter.last_target_size
            if eye_frame.shape[1] != target_w or eye_frame.shape[0] != target_h:
                eye_frame_proc = cv2.resize(eye_frame, (target_w, target_h))
            else:
                eye_frame_proc = eye_frame
            
            # Draw overlay on eye frame for preview
            # We work on a copy to not mess up the original frame if we needed it for something else
            eye_display = eye_frame_proc.copy()
            self.draw_eye_overlay(eye_display)
            
            # Display Eye Frame
            rgb_eye = cv2.cvtColor(eye_display, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_eye.shape
            qt_eye = QImage(rgb_eye.data, w, h, ch * w, QImage.Format_RGB888)
            self.eye_video_label.setPixmap(QPixmap.fromImage(qt_eye).scaled(self.eye_video_label.size(), Qt.KeepAspectRatio))
        else:
            # AUTO-HIDE: Hide Eye View when no camera signal (avoid black screen)
            if hasattr(self, 'dock_eye') and self.dock_eye.isVisible():
                self.dock_eye.hide()

        # --- Process Scene Frame ---
        scene_frame = None
        
        if self.media_mode:
            if self.media_image is not None:
                # Static Image
                scene_frame = self.media_image.copy()
            elif self.media_video_cap is not None:
                # Video Logic
                if self.media_playing and not self.media_paused:
                    ret, frame = self.media_video_cap.read()
                    if not ret:
                        # End of video -> Pause at end
                        self.media_playing = False
                        self.media_paused = True
                        self.btn_media_play.setText("▶")
                        if self.last_valid_media_frame is not None:
                            scene_frame = self.last_valid_media_frame
                    else:
                        scene_frame = frame
                        self.last_valid_media_frame = frame.copy()
                        
                        # Sync seek bar with current position (if not manually seeking)
                        if not self._seeking:
                            current_pos = int(self.media_video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                            self.seek_slider.blockSignals(True)
                            self.seek_slider.setValue(current_pos)
                            self.seek_slider.blockSignals(False)
                            self._update_seek_label(current_pos)
                else:
                    # Paused or not started: Show last valid frame (or first frame)
                    if self.last_valid_media_frame is not None:
                        scene_frame = self.last_valid_media_frame
        else:
            scene_frame = self.latest_scene_frame
        
        # If no scene camera, create a black frame
        if scene_frame is None:
            scene_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            scene_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Text is handled by _draw_qt_text_overlays or the logo placeholder now
        else:
            if not self.media_mode:
                # Do NOT resize scene frame. Keep native resolution (e.g. 1920x1080)
                # scene_frame = cv2.resize(scene_frame, (640, 480))
                pass

        # 2. Map Gaze (using pupil data from eye frame)
        timestamp = time.time()
        
        
        # Blink/Noise Filter: If pupil is lost or low confidence, we stop updating gaze.
        # Ideally we would fade out the last point, but for now just don't update.
        pupil_ok = False
        if self.current_pupil and eye_frame is not None:
            if self.current_pupil.get('confidence', 1.0) > 0.1:
                pupil_ok = True
                
        if pupil_ok:
            # --- Calibration Buffer Collection ---
            if self.is_calibrating and self.cal_collecting:
                if self.cal_mode == "head":
                    # Collect Head Data (Yaw, Pitch)
                    if self.head_active and self.head_gaze_data:
                        # head_gaze_data is (yaw, pitch, x, y)
                        # We want (yaw, pitch)
                        self.cal_buffer.append((self.head_gaze_data[0], self.head_gaze_data[1]))
                    else:
                         print("Head Calibration: Head not active, skipping sample.")
                else:
                    # Collect Pupil Data
                    self.cal_buffer.append(self.current_pupil['center'])
                    
                if len(self.cal_buffer) >= self.cal_samples_needed:
                    self.finalize_calibration_point()

            # --- Auto-Fit Globe Collection ---
            if self.is_fitting_globe:
                self.globe_fitter.add_pupil_data(self.current_pupil)
                elapsed = time.time() - self.fit_start_time
                
                # Get quality metrics for live feedback
                metrics = self.globe_fitter.get_quality_metrics()
                coverage = metrics['angular_coverage']
                ray_count = metrics['ray_count']
                is_ready = metrics['is_ready']
                
                # Live feedback in status label
                self.cal_status_lbl.setText(f"Fitting Globe... {elapsed:.1f}s | Rays: {ray_count} | Coverage: {coverage:.0f}%")
                
                # Auto-stop conditions:
                # 1. Quality is good (>= 60% coverage AND >= 50 rays)
                # 2. OR timeout reached (7 seconds max)
                if is_ready or elapsed >= 7.0:
                    self.finalize_auto_fit()

            # --- Simple Center Calibration Mode ---
            if self.center_calibrated and self.center_pupil:
                px, py = self.current_pupil['center']
                
                # Calculate vector from current pupil to globe center
                # We want the vector relative to the "Zero Vector" (calibrated straight-ahead)
                gx, gy = self.globe_center
                
                # Current vector from globe to pupil
                vx = px - gx
                vy = py - gy
                
                # Subtract the calibration zero offset (if any)
                if self.cal_zero_vector:
                    zvx, zvy = self.cal_zero_vector
                    dx = vx - zvx
                    dy = vy - zvy
                else:
                    dx = vx
                    dy = vy
                
                # Apply Mirroring to X (if checked)
                if self.mirror_x:
                    dx = -dx
                
                # Apply Sensitivity/Scaling
                # We map this delta to screen pixels. 
                # Sensitivity 10.0 means 1 pixel of pupil movement = 10 pixels of screen movement
                gx_delta = dx * self.gaze_sensitivity
                gy_delta = dy * self.gaze_sensitivity # usually Y is consistent
                
                # Center of Scene is (320, 240) OR Screen Center OR Media Size
                if self.current_map_mode.startswith("screen"):
                     base_w, base_h = self.screen_width, self.screen_height
                elif self.media_mode and self.media_original_size:
                     # Use media resolution for proper gaze scaling
                     base_w, base_h = self.media_original_size
                else:
                     base_w, base_h = 640, 480
                
                center_x = base_w / 2
                center_y = base_h / 2
                
                gx = center_x + gx_delta
                gy = center_y + gy_delta
                
                # Clamp
                gx = float(np.clip(gx, 0, base_w))
                gy = float(np.clip(gy, 0, base_h))
                
                # --- HYBRID / HEAD TRACKING LOGIC ---
                # Determine "Base Gaze" from Eye
                eye_gaze_x, eye_gaze_y = gx, gy
                
                # Check Head Mode
                mode = "Eye Only"
                if self.btn_head_only.isChecked(): mode = "Head Only"
                elif self.btn_hybrid.isChecked(): mode = "Hybrid"
                
                final_gaze = None
                
                if mode == "Eye Only":
                    final_gaze = (eye_gaze_x, eye_gaze_y)
                    
                elif mode == "Head Only":
                    if self.head_active and self.head_gaze_data:
                        _, _, hx_screen, hy_screen = self.head_gaze_data
                        
                        # Use Calibrated Model if available
                        if self.head_calibration.is_calibrated:
                            res = self.head_calibration.map_point((self.head_gaze_data[0], self.head_gaze_data[1]))
                            if res:
                                final_gaze = res
                        else:
                            # Fallback to screen mapping (approximated)
                            scale_x = base_w / getattr(self, 'screen_width', 1920)
                            scale_y = base_h / getattr(self, 'screen_height', 1080)
                            final_gaze = (hx_screen * scale_x, hy_screen * scale_y)
                
                elif mode == "Hybrid":
                    # KEY SYSTEMIC LOGIC: COMPLEMENTARITY
                    # 1. Base is Head (Gross Movement)
                    # 2. Add Eye Offset (Fine Movement)
                    
                    head_pt = None
                    if self.head_active and self.head_gaze_data:
                        # Use Calibrated Model if available
                        if self.head_calibration.is_calibrated:
                            res = self.head_calibration.map_point((self.head_gaze_data[0], self.head_gaze_data[1]))
                            if res:
                                head_pt = res
                        else:
                            _, _, hx_s, hy_s = self.head_gaze_data
                            scale_x = base_w / getattr(self, 'screen_width', 1920)
                            scale_y = base_h / getattr(self, 'screen_height', 1080)
                            head_pt = (hx_s * scale_x, hy_s * scale_y)
                    
                    if head_pt and pupil_ok:
                        # BOTH AVAILABLE: Combine them
                        # But wait, 'eye_gaze_x' is absolute gaze on screen.
                        # If we want Eye to be relative to Head, we need to know "Eye Delta".
                        # For simplicity in this implementation:
                        # The "Eye Gaze" we calculated (gx, gy) is heavily dependent on "Globe Center".
                        # In Hybrid mode, the "Globe Center" effectively moves with the Head.
                        
                        # So: Final = Head_Pos + (Eye_Gaze - Screen_Center)
                        # Or closer to Orlosky: Head provides the coarse location, Eye provides offset.
                        
                        screen_cx, screen_cy = base_w / 2, base_h / 2
                        eye_delta_x = eye_gaze_x - screen_cx
                        eye_delta_y = eye_gaze_y - screen_cy
                        
                        final_gaze = (head_pt[0] + eye_delta_x, head_pt[1] + eye_delta_y)
                        
                    elif head_pt and not pupil_ok:
                        # EYE LOST -> FALLBACK TO HEAD
                        final_gaze = head_pt
                        
                    elif not head_pt and pupil_ok:
                        # HEAD LOST -> FALLBACK TO EYE
                        final_gaze = (eye_gaze_x, eye_gaze_y)
                
                if final_gaze:
                    fgx, fgy = final_gaze
                    # Clamp
                    fgx = float(np.clip(fgx, 0, base_w))
                    fgy = float(np.clip(fgy, 0, base_h))
                    
                    if self.current_map_mode.startswith("screen"):
                        self.calibration_overlay.set_gaze(fgx, fgy)
                        self.scene_gaze = (fgx * 640/base_w, fgy * 480/base_h) 
                    else:
                        self.scene_gaze = (fgx, fgy)
                else:
                    self.scene_gaze = None
                
            # --- Polynomial Calibration Mode (Legacy/Advanced) ---
            elif self.calibration.is_calibrated:
                center = self.current_pupil['center']
                # If calibrated in screen mode, raw_gaze is screen coords
                raw_gaze = self.gaze_mapper.map_to_scene(center, timestamp)
                
                if raw_gaze:
                    # Apply Static Drift (from Quick Center)
                    gx_final = raw_gaze[0] + getattr(self, 'gaze_drift', (0,0))[0]
                    gy_final = raw_gaze[1] + getattr(self, 'gaze_drift', (0,0))[1]
                    raw_gaze = (gx_final, gy_final)

                    # === HYBRID MODE: HEAD COMPENSATION ===
                    # If Hybrid Mode is ON, we add Head Delta to Eye Gaze.
                    # This compensates for head rotation (if look at same spot, head turns right -> pupil turns left.
                    # Head Tracker detects Right turn. Adding Right Turn to Left Pupil cancels out drift.)
                    if self.btn_hybrid.isChecked() and self.head_tracker and self.head_gaze_data:
                        try:
                            # head_gaze_data is (yaw, pitch, sx, sy)
                            _, _, h_sx, h_sy = self.head_gaze_data
                            
                            scr_w = getattr(self, 'screen_width', 1920)
                            scr_h = getattr(self, 'screen_height', 1080)
                            
                            # Calculate Deviation from Center
                            # (Assuming Head Zero set center at screen center)
                            h_dx = h_sx - (scr_w / 2)
                            h_dy = h_sy - (scr_h / 2)
                            
                            # Apply Compensation (Simple Linear Addition)
                            # Only apply if using Screen Mapping (where coords match)
                            if self.current_map_mode.startswith("screen"):
                                raw_gaze = (raw_gaze[0] + h_dx, raw_gaze[1] + h_dy)
                        except Exception as e:
                            print(f"Hybrid Comp Error: {e}")

                    if self.current_map_mode.startswith("screen"):
                        self.calibration_overlay.set_gaze(raw_gaze[0], raw_gaze[1])
                        # Map back for scene/canvas
                        sx, sy = raw_gaze
                        # Map from screen to scene frame coords
                        scene_w = scene_frame.shape[1]
                        scene_h = scene_frame.shape[0]
                        self.scene_gaze = (sx * scene_w/self.screen_width, sy * scene_h/self.screen_height)
                    elif self.media_mode and self.media_original_size:
                        # MEDIA MODE: If calibration was done on media, raw_gaze is already in media coords
                        # No scaling needed - calibration targets were placed at media resolution
                        self.scene_gaze = raw_gaze
                    else:
                        self.scene_gaze = raw_gaze

        # 3. Map to Canvas (Constraint ROI) & Record Session
        if self.scene_gaze:
            # === HEATMAP ACCUMULATION ===
            # Always record heatmap data (separate from display toggle)
            if self.record_heatmap:
                gx, gy = self.scene_gaze
                # Set frame size for pixel-based accumulation
                if scene_frame is not None:
                    sh, sw = scene_frame.shape[:2]
                    if self.heatmap.frame_width != sw or self.heatmap.frame_height != sh:
                        self.heatmap.set_frame_size(sw, sh)
                    self.heatmap.add_point_px(gx, gy)
            
            # Update GazeMapper's ROI based on current settings
            # Update GazeMapper's ROI based on current settings
            # Default to A3 (Standard) since UI control was removed
            w_mm, h_mm = 420.0, 297.0
            
            if self.current_map_mode.startswith("screen"):
                roi_w, roi_h = self.screen_width, self.screen_height
            elif self.media_mode and self.media_original_size:
                # Use media dimensions for correct ROI mapping
                roi_w, roi_h = self.media_original_size
            else:
                 # Use actual scene frame size
                 roi_w = scene_frame.shape[1]
                 roi_h = scene_frame.shape[0]
            
            self.gaze_mapper.set_roi(0, 0, roi_w, roi_h, w_mm, h_mm)
            
            # Map Scene -> Canvas
            self.canvas_gaze = self.gaze_mapper.map_scene_to_canvas(self.scene_gaze)
            
            # Update Path
            self.gaze_path.append(self.scene_gaze)
            if len(self.gaze_path) > 40: self.gaze_path.pop(0)
            
            # Record Session (using Canvas coordinates for physical accuracy)
            if self.session.is_running and self.canvas_gaze:
                cx_mm, cy_mm = self.canvas_gaze
                self.session.add_sample(timestamp, cx_mm, cy_mm, self.current_pupil)
                self.heatmap.add_point(self.scene_gaze[0], self.scene_gaze[1]) 
            
            # Update Info
            self.sess_info_lbl.setText(f"Time: {self.session.get_duration():.1f}s\nSamples: {len(self.session.samples)}")

        # 4. Draw Overlay on Scene
        scene_display = scene_frame.copy()
        self.draw_scene_overlay(scene_display)

        # 5. Record Video Frame
        if self.session.is_running and self.video_recorder.is_recording:
             self.video_recorder.write_frame(scene_display)

        # 6. Display Scene
        rgb_scene = cv2.cvtColor(scene_display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_scene.shape
        qt_scene = QImage(rgb_scene.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_scene)
        
        # Draw High-Quality Text Overlays (Qt)
        self._draw_qt_text_overlays(pixmap)
        
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))

    def draw_eye_overlay(self, frame):
        # Draw pupil on eye frame
        if self.current_pupil:
            # Draw standard pupil
            cv2.ellipse(frame, self.current_pupil['raw_rect'], (0, 255, 0), 2)
            cx, cy = self.current_pupil['center']
            
            # --- Eye Globe Feedback (Orlosky Style) ---
            # Use MANUAL globe center (Model Fitting)
            gx, gy = self.globe_center
            r = int(self.globe_radius)
            
            # Draw blue "Globe" reference (Fixed Model)
            cv2.circle(frame, (gx, gy), r, (255, 0, 0), 2)
            cv2.circle(frame, (gx, gy), 4, (255, 255, 0), -1) # Yellow center dot
            
            # Draw line properly connecting globe center to current pupil (Gaze Vector)
            # Yellow line RESTORED per user request
            cv2.line(frame, (gx, gy), (int(cx), int(cy)), (0, 255, 255), 2)

    def draw_scene_overlay(self, frame):
        h, w = frame.shape[:2]
        
        # --- BORESIGHT CROSSHAIR (Hybrid/Head Mode Only) ---
        # Draws a green crosshair to assist aligning the Scene Camera (Head) with the Screen Center.
        if self.is_calibrating and not self.btn_eye_only.isChecked():
            cx, cy = w // 2, h // 2
            cx, cy = w // 2, h // 2
            # Green Crosshair (Open Center)
            gap = 10
            len_ = 35
            # Left
            cv2.line(frame, (cx - len_, cy), (cx - gap, cy), (0, 255, 0), 2)
            # Right
            cv2.line(frame, (cx + gap, cy), (cx + len_, cy), (0, 255, 0), 2)
            # Top
            cv2.line(frame, (cx, cy - len_), (cx, cy - gap), (0, 255, 0), 2)
            # Bottom
            cv2.line(frame, (cx, cy + gap), (cx, cy + len_), (0, 255, 0), 2)
            
            # Outer circle
            cv2.circle(frame, (cx, cy), 18, (0, 255, 0), 1)
            
            # Instruction Text for alignment
            # Split into two lines for clarity
            cv2.putText(frame, "1. Overlay Crosshair on Target", (cx - 130, cy + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "2. LOOK AT VIDEO FEEDBACK TARGET", (cx - 150, cy + 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # === HEATMAP OVERLAY ===
        # Blend heatmap visualization onto the scene frame if enabled
        if self.show_heatmap and np.max(self.heatmap.grid) > 0:
            h, w = frame.shape[:2]
            # Render the colored heatmap overlay with alpha channel
            # OPTIMIZATION: Dynamic PROXY & BLUR from UI
            current_time = time.time()
            if current_time - self.heatmap_last_update > 0.5:
                # Proxy settings from UI
                PROXY_W = self.heatmap_proxy_width
                scale = w / PROXY_W
                small_w = PROXY_W
                small_h = int(h / scale)
                
                if small_w > 0 and small_h > 0:
                     # Dynamic Blur Scaling: Treat slider as "Base Blur at 320px"
                     # Scale blur proportionally to current resolution to maintain visual consistency
                     base_w = 320.0
                     blur_scale = PROXY_W / base_w
                     scaled_blur = int(self.heatmap_blur_radius * blur_scale)
                     
                     small_overlay = self.heatmap.render_overlay(small_w, small_h, alpha=0.6, blur_size=scaled_blur)
                     # Upscale to full size
                     self.heatmap_overlay_cache = cv2.resize(small_overlay, (w, h), interpolation=cv2.INTER_LINEAR)
                self.heatmap_last_update = current_time
            
            overlay = getattr(self, 'heatmap_overlay_cache', None)
            
            # Initial load fallback
            if overlay is None or overlay.shape[:2] != (h, w):
                 PROXY_W = self.heatmap_proxy_width
                 scale = w / PROXY_W
                 small_w = PROXY_W
                 small_h = int(h / scale)
                 if small_w > 0 and small_h > 0:
                     small_overlay = self.heatmap.render_overlay(small_w, small_h, alpha=0.6, blur_size=self.heatmap_blur_radius)
                     overlay = cv2.resize(small_overlay, (w, h), interpolation=cv2.INTER_LINEAR)
                     self.heatmap_overlay_cache = overlay
            
            if overlay is not None:
                # Alpha blending: output = foreground * alpha + background * (1 - alpha)
                # overlay is BGRA, frame is BGR
                alpha = overlay[:, :, 3:4] / 255.0
                bgr_overlay = overlay[:, :, :3]
                
                # Blend where alpha > 0
                frame[:] = (bgr_overlay * alpha + frame * (1 - alpha)).astype(np.uint8)
        # Legacy CV2 Drawing Removed - Handled by _draw_qt_text_overlays
        if self.is_calibrating:
            # Instructions handled in _draw_qt_text_overlays
            
            if self.cal_mode == "center":
                if self.center_cal_step == 2:
                    h, w = frame.shape[:2]
                    px = w // 2
                    py = h // 2
                    # Scale UI elements based on resolution
                    scale = max(1, w / 640)
                    circle_r = int(15 * scale)
                    dot_r = int(5 * scale)
                    line_len = int(20 * scale)
                    thickness = max(2, int(2 * scale))
                    
                    # Text handled in Qt Overlay
                    cv2.circle(frame, (px, py), circle_r, (0, 0, 255), thickness)
                    cv2.circle(frame, (px, py), dot_r, (0, 255, 0), -1)
                    cv2.line(frame, (px-line_len, py), (px+line_len, py), (0, 0, 255), thickness)
                    cv2.line(frame, (px, py-len_len), (px, py+len_len), (0, 0, 255), thickness)
            else:
                if self.calibration_step < len(self.calibration_targets):
                    nx, ny = self.calibration_targets[self.calibration_step]
                    # De-normalize for visualization on the frame
                    h, w = frame.shape[:2]
                    px = int(nx * w)
                    py = int(ny * h)
                    # Scale UI elements based on resolution
                    scale = max(1, w / 640)
                    circle_r = int(15 * scale)
                    dot_r = int(5 * scale)
                    line_len = int(20 * scale)
                    thickness = max(2, int(2 * scale))
                    
                    if self.crosshair_img_cv is not None:
                         # Responsive Scale: 15% of min dimension, clamped 40-150px
                         base_dim = min(w, h)
                         target_size = int(max(40, min(150, base_dim * 0.15)))
                         
                         # Resize image
                         # crosshair_img_cv is BGRA
                         orig_h, orig_w = self.crosshair_img_cv.shape[:2]
                         scale_factor = target_size / max(orig_h, orig_w)
                         new_w = int(orig_w * scale_factor)
                         new_h = int(orig_h * scale_factor)
                         
                         resized_crosshair = cv2.resize(self.crosshair_img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
                         
                         # --- Inline Overlay with Alpha Blending ---
                         ch_h, ch_w = resized_crosshair.shape[:2]
                         x1 = px - ch_w // 2
                         y1 = py - ch_h // 2
                         x2 = x1 + ch_w
                         y2 = y1 + ch_h
                         
                         # Clamp to frame boundaries
                         if x1 >= 0 and y1 >= 0 and x2 < w and y2 < h:
                             overlay_img = resized_crosshair
                             alpha_mask = overlay_img[:, :, 3] / 255.0
                             alpha_inv = 1.0 - alpha_mask
                             
                             # Efficient Alpha Blending
                             roi = frame[y1:y2, x1:x2]
                             
                             for c in range(3): # BGR channels
                                 roi[:, :, c] = (alpha_mask * overlay_img[:, :, c] +
                                                 alpha_inv * roi[:, :, c])
                             
                             frame[y1:y2, x1:x2] = roi.astype(np.uint8)
                    else:
                        # Fallback drawing
                        cv2.circle(frame, (px, py), circle_r, (0, 0, 255), thickness)
                        cv2.circle(frame, (px, py), dot_r, (0, 255, 0), -1)
                        cv2.line(frame, (px-line_len, py), (px+line_len, py), (0, 0, 255), thickness)
                        cv2.line(frame, (px, py-len_len), (px, py+len_len), (0, 0, 255), thickness)

        # Draw Gaze Path
        if self.show_gaze_path and len(self.gaze_path) > 1:
            pts = []
            # Custom Gaze Path (Comet Style: Red=New -> Blue=Old)
            points = self.gaze_path
            
            # Limit path length for gradient effect
            max_points = 60 # ~2 seconds history
            if len(points) > max_points:
                points = points[-max_points:]
            
            # Draw segment by segment
            for i in range(1, len(points)):
                pt1 = points[i-1]
                pt2 = points[i]
                
                # Progress t: 0=Oldest (Blue), 1=Newest (Red)
                t = i / len(points)
                
                # BGR Color: Blue (255,0,0) -> Red (0,0,255)
                b = int(255 * (1-t))
                r = int(255 * t)
                color = (b, 0, r)
                
                # Taper thickness
                thickness = max(1, int(4 * t))
                
                pt1_int = (int(pt1[0]), int(pt1[1]))
                pt2_int = (int(pt2[0]), int(pt2[1]))
                
                cv2.line(frame, pt1_int, pt2_int, color, thickness)

        # Draw Gaze Cursor (Custom Image or Fallback Red Dot)
        # Only draw if show_gaze_cursor is enabled (hidden by default to avoid self-consciousness)
        if self.scene_gaze and self.show_gaze_cursor:
            gx, gy = self.scene_gaze
            h, w = frame.shape[:2]
            
            # Select cursor image based on mode
            cursor_img = None
            cache = None
            
            if self.media_mode:
                if hasattr(self, 'gaze_cursor_long'): cursor_img = self.gaze_cursor_long
                if hasattr(self, '_cursor_cache_long'): cache = self._cursor_cache_long
            else:
                if hasattr(self, 'gaze_cursor_round'): cursor_img = self.gaze_cursor_round
                if hasattr(self, '_cursor_cache_round'): cache = self._cursor_cache_round
            
            # Use custom cursor image if available
            if cursor_img is not None:
                # Scale cursor: target 20px at 640px width, scale proportionally
                base_size = 20
                scale_factor = max(0.5, w / 640)
                target_size = int(base_size * scale_factor)
                
                # Check cache for scaled version
                if target_size not in cache:
                    # Scale the cursor image
                    orig_h, orig_w = cursor_img.shape[:2]
                    aspect = orig_w / orig_h
                    new_h = target_size
                    new_w = int(target_size * aspect)
                    scaled = cv2.resize(cursor_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    cache[target_size] = scaled
                
                cursor_scaled = cache[target_size]
                self._overlay_image_alpha(frame, cursor_scaled, int(gx), int(gy))
            else:
                # Fallback: Red dot cursor
                scale_factor = max(1, w / 640)
                dot_radius = int(5 * scale_factor)
                ring_radius = int(7 * scale_factor)
                ring_thickness = max(1, int(scale_factor))
                cv2.circle(frame, (int(gx), int(gy)), dot_radius, (0, 0, 255), -1) 
                cv2.circle(frame, (int(gx), int(gy)), ring_radius, (255, 255, 255), ring_thickness)
            
            # Text Coordinates (also scaled)
            # Text Coordinates (also scaled)
            # Handled in _draw_qt_text_overlays

        # --- Head Tracking Visualization ---
        if self.head_enable_check.isChecked():
            # Determine mode from buttons
            mode = "Eye Only"
            if self.btn_head_only.isChecked(): mode = "Head Only"
            elif self.btn_hybrid.isChecked(): mode = "Hybrid"
            
            if mode != "Eye Only":
                if self.head_active and self.head_gaze_data:
                    _, _, hx_screen, hy_screen = self.head_gaze_data
                    
                    # SYSTEMIC VISUALIZATION (Orlosky Style)
                    # 1. Map Head Screen Coords back to Scene Frame (640x480)
                    # Assumption: hx_screen is in the domain of screen_width (default 1920)
                    # We map it to local frame coordinates for display
                    
                    sx = 640
                    sy = 480
                    # Robust scaling (avoid div by zero if screen_width not set)
                    sw = getattr(self, 'screen_width', 1920)
                    sh = getattr(self, 'screen_height', 1080)
                    
                    head_x = int(hx_screen * sx / sw)
                    head_y = int(hy_screen * sy / sh)
                    
                    # Center of the "Head Field" (Forward)
                    center_x, center_y = sx // 2, sy // 2
                    
                    # DRAW HEAD VECTOR (Blue)
                    # Line from Center (Forward) -> Head Gaze Point
                    cv2.arrowedLine(frame, (center_x, center_y), (head_x, head_y), (230, 216, 173), 3, tipLength=0.05) # Light Blue
                    
                    # Draw Head Cursor (Blue Crosshair)
                    cv2.drawMarker(frame, (head_x, head_y), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(frame, "Head", (head_x+10, head_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 216, 173), 1)

                    # DRAW HYBRID OFFSET (Green)
                    # If we have eye gaze, draw the vector from Head Point -> Eye Point
                    if self.scene_gaze:
                        gx, gy = self.scene_gaze
                        # Scale gaze if needed (scene_gaze is usually 640x480 already)
                        if self.current_map_mode == "scene":
                             frame_gx, frame_gy = int(gx), int(gy)
                        else:
                             # If mapped to screen, scale back to frame for visualization?
                             # For simplicity, let's assume scene_gaze is sufficient proxy or use projected gaze
                             # If scene_gaze is available, it's usually in Scene Monitor space
                             frame_gx, frame_gy = int(gx), int(gy)

                        # Draw "Rubber Band" or Vector from Head to Eye - REMOVED per user request
                        # cv2.line(frame, (head_x, head_y), (frame_gx, frame_gy), (0, 255, 0), 2)
                        # cv2.circle(frame, (frame_gx, frame_gy), 4, (0, 255, 0), -1)

                elif not self.head_active:
                    # LOST TRACKING - Recenter Trigger (Crosshair/Square)
                    cx, cy = 320, 240
                    # Draw Hollow Square
                    cv2.rectangle(frame, (cx-60, cy-60), (cx+60, cy+60), (0, 0, 255), 4)
                    # Draw Crosshair
                    cv2.line(frame, (cx-40, cy), (cx+40, cy), (0, 0, 255), 3)
                    cv2.line(frame, (cx, cy-40), (cx, cy+40), (0, 0, 255), 3)
                    
                    cv2.putText(frame, "LOST VISUAL - RECENTER", (cx-130, cy+90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def start_calibration_center(self):
        self.is_calibrating = True
        self.cal_capture_btn.setEnabled(True)
        self.center_calibrated = False
        self.center_pupil = None
        self.cal_mode = "center"
        self.center_cal_step = 1
        self.gaze_path = [] # Clear path on recalibration
        
        # RESET Grid calibration to avoid interference
        self.calibration.clear_points()
        
        self.cal_status_lbl.setText("Step 1: Reposition eye to fill frame -> Press 'C'")
        self.setFocus() # Force focus so key presses work
        print("Calibration Started (Center mode). Grid calibration reset.")


    def auto_tune_head_tracker(self):
        """Calculates and sets optimal Head Tracker Yaw/Pitch ranges based on calibration interaction."""
        if not self.head_cal_points or not self.head_tracker:
            return

        print(f"Auto-Tuning Head Tracker with {len(self.head_cal_points)} points...")
        try:
            # Extract lists
            yaws = [p[0] for p in self.head_cal_points]
            pitches = [p[1] for p in self.head_cal_points]
            wxs = [p[2] for p in self.head_cal_points]
            wys = [p[3] for p in self.head_cal_points]
            
            # Ranges
            yaw_min, yaw_max = min(yaws), max(yaws)
            pitch_min, pitch_max = min(pitches), max(pitches)
            
            scr_w = getattr(self, 'screen_width', 1920)
            scr_h = getattr(self, 'screen_height', 1080)
            
            wx_range = max(wxs) - min(wxs)
            wy_range = max(wys) - min(wys)
            dy_yaw = yaw_max - yaw_min
            dy_pitch = pitch_max - pitch_min
            
            if wx_range > 100 and dy_yaw > 1:
                # Formula: Range = (ScreenW * DeltaYaw) / (2 * DeltaScreenX)
                new_yaw_range = (scr_w * dy_yaw) / (2 * wx_range)
                new_yaw_range = max(10, min(90, new_yaw_range))
                self.head_tracker.yaw_range = new_yaw_range
                print(f"  -> Updated Yaw Range: {new_yaw_range:.1f}")
                
            if wy_range > 100 and dy_pitch > 1:
                new_pitch_range = (scr_h * dy_pitch) / (2 * wy_range)
                new_pitch_range = max(5, min(60, new_pitch_range))
                self.head_tracker.pitch_range = new_pitch_range
                print(f"  -> Updated Pitch Range: {new_pitch_range:.1f}")
                
        except Exception as e:
            print(f"Auto-Tune Error: {e}")

    def start_calibration_routine(self, mode="grid"):
        # FORCE FOCUS to MainWindow to ensure 'C' key works
        self.raise_()
        self.activateWindow()
        self.setFocus()
        
        self.is_calibrating = True
        self.cal_capture_btn.setEnabled(True)
        self.calibration_step = 0
        self.cal_mode = mode
        self.gaze_path = [] # Clear path on recalibration
        self.head_cal_points = [] # Store (yaw, pitch, wx, wy) for head range tuning
        
        # RESET Center calibration to avoid interference
        self.center_calibrated = False
        self.center_pupil = None
        self.cal_zero_vector = None
        
        if mode == "grid":
            self.calibration.clear_points()
            print("Calibration Started (Grid mode). Center calibration reset.")
        elif mode == "head":
            self.head_calibration.clear_points()
            print("Calibration Started (HEAD mode). Head model reset.")
            # Ensure head tracker is active?
            if not self.head_active:
                QMessageBox.warning(self, "Head Tracker", "Head Tracker is not active. Please enable it first.")
        
        self.cal_status_lbl.setText("Look at Target 1/9 -> Press 'C'")
        self.setFocus() # Force focus so key presses work
        
        if mode == "quick_drift":
             self.calibration_targets = [(0.5, 0.5)]
             self.cal_status_lbl.setText("Quick Center: Align Head & Eye -> Press 'C'")
             # Show overlay immediately
             if self.current_map_mode.startswith("screen"):
                 self.calibration_overlay.showFullScreen()
                 self.update_overlay_target()
             return

        # Old pixel based logic removed.
        self.calibration_targets = []
        
        # If in screen mode, remap targets to screen coordinates
        # If in screen mode, remap targets to screen coordinates
        if self.current_map_mode.startswith("screen"):
            if self.media_mode:
                 # MEDIA CALIBRATION: Constrain to Image ROI
                 roi = self.get_media_display_rect()
                 # Convert local label coords to global screen coords for the Overlay
                 tl = self.video_label.mapToGlobal(roi.topLeft())
                 br = self.video_label.mapToGlobal(roi.bottomRight())
                 
                 # The Overlay is typically full screen on the monitor.
                 # We need to express targets in NORMALIZED coords relative to the Overlay Size?
                 # Currently update_overlay_target takes (0-1) and multiplies by overlay size.
                 # So we need to normalize global RECT relative to Overlay Rect?
                 # Assuming Overlay covers the screen:
                 
                 # Let's get the screen geometry the overlay is on
                 screen_rect = self.calibration_overlay.geometry() # or screen geometry?
                 # If overlay is full screen:
                 sw = self.screen_width
                 sh = self.screen_height
                 
                 # Calculate normalized margins based on ROI
                 min_x = (tl.x() - screen_rect.x()) / sw
                 min_y = (tl.y() - screen_rect.y()) / sh
                 max_x = (br.x() - screen_rect.x()) / sw
                 max_y = (br.y() - screen_rect.y()) / sh
                 
                 mid_x = (min_x + max_x) / 2
                 mid_y = (min_y + max_y) / 2
                 
                 xs = [min_x, mid_x, max_x]
                 ys = [min_y, mid_y, max_y]
                 
                 self.calibration_targets = []
                 for y in ys:
                     for x in xs:
                         self.calibration_targets.append((x, y))
            else:
                # STANDARD SCREEN CALIBRATION
                # For screen, we also use normalized, but we might want margins.
                # Let's map margins to normalized coords approx 0.05 to 0.95
                self.calibration_targets = []
                ys = [0.1, 0.5, 0.9]
                xs = [0.1, 0.5, 0.9]
                for y in ys:
                    for x in xs:
                        self.calibration_targets.append((x, y))
            
            self.calibration_overlay.show()
            self.update_overlay_target()
        else:
            # Normalized Scene Coords (roughly 0.1 to 0.9 to avoid edge clipping)
            self.calibration_targets = []
            ys = [0.1, 0.5, 0.9]
            xs = [0.1, 0.5, 0.9]
            for y in ys:
                for x in xs:
                    self.calibration_targets.append((x, y))
            
            self.calibration_overlay.hide() # Why hide? Because we draw on scene image directly if not overlay?
            # Wait, if not screen mode, we draw on the OpenCV frame using draw_scene_overlay.
            # We should probably UNIFY this. But for now, let's keep logic intact.
            # If start_calibration_grid is called, typically we are invalidating previous calibration.
            pass
            self.update_overlay_target()

        
    def start_calibration_collection(self):
        """Starts collecting samples for the current point."""
        if not self.is_calibrating:
            return
        
        self.cal_buffer = []
        self.cal_collecting = True
        self.cal_capture_btn.setEnabled(False) # Disable while collecting
        if self.cal_mode == "center":
            self.cal_status_lbl.setText("Capturing center...")
            if self.current_map_mode.startswith("screen"):
                self.calibration_overlay.set_message("Capturing Center...")
        else:
            self.cal_status_lbl.setText(f"Collecting... 0/{self.cal_samples_needed}")
            if self.current_map_mode.startswith("screen"):
                self.calibration_overlay.set_message(f"Collecting... 0/{self.cal_samples_needed}")

    def finalize_calibration_point(self):
        """Averages the buffer and adds the point."""
        self.cal_collecting = False
        self.cal_capture_btn.setEnabled(True)
        
        if not self.cal_buffer:
            print("Buffer empty, failed to capture.")
            self.cal_status_lbl.setText("Capture Failed (No Data)")
            return

        avg_x = sum(p[0] for p in self.cal_buffer) / len(self.cal_buffer)
        avg_y = sum(p[1] for p in self.cal_buffer) / len(self.cal_buffer)
        pupil_xy = (avg_x, avg_y)

        if self.cal_mode == "center":
            self.center_pupil = pupil_xy
            
            # Calculate Zero Vector (Globe Center -> Straight Ahead Pupil)
            # This is the "offset" we subtract during tracking
            gx, gy = self.globe_center
            self.cal_zero_vector = (avg_x - gx, avg_y - gy)
            
            self.center_calibrated = True
            
            # Advance step
            if self.center_cal_step == 1:
                self.center_cal_step = 2
                self.cal_status_lbl.setText("Step 2: Look at scene center -> Press 'C'")
            else:
                self.is_calibrating = False
                self.cal_collecting = False
                self.cal_capture_btn.setEnabled(False)
                self.cal_status_lbl.setText("Center Calibration Complete")
                print(f"Center Calibration Complete. Zero Vector: {self.cal_zero_vector}")
            return

        if self.calibration_step < len(self.calibration_targets):
            norm_xy = self.calibration_targets[self.calibration_step]
            
            # De-normalize based on Map Mode
            world_xy = (0,0)
            if self.current_map_mode.startswith("screen"):
                 # Screen Pixels
                 world_xy = (norm_xy[0] * getattr(self, 'screen_width', 1920), norm_xy[1] * getattr(self, 'screen_height', 1080))
            else:
                 # Scene Pixels (use current scene frame or stored size)
                 base_w, base_h = self.media_original_size if (self.media_mode and self.media_original_size) else (640, 480)
                 world_xy = (norm_xy[0] * base_w, norm_xy[1] * base_h)
            
            if self.cal_mode == "head":
                 self.head_calibration.add_point(pupil_xy, world_xy) # pupil_xy is (yaw, pitch)
                 print(f"Added Head Calibration Point {self.calibration_step}: YawPitch={pupil_xy} -> World={world_xy}")
            else:
                     
                     # Check if Quick Drift
                     if self.cal_mode == "quick_drift" and self.current_map_mode.startswith("screen"):
                         # Current Predicted Gaze (with Head Comp if active)
                         # We need the CURRENT Gaze from update_frame?
                         # Or we calculate it here based on pupil.
                         # Better: use the LAST calculated gaze from update_frame loop?
                         # But update_frame runs constantly. 
                         # Let's map current pupil.
                         curr_gaze = self.calibration.map_pupil(pupil_xy)
                         if curr_gaze:
                             # Add Head Comp if Hybrid
                             if not self.btn_eye_only.isChecked() and self.head_gaze_data:
                                 _, _, h_sx, h_sy = self.head_gaze_data
                                 scr_w = getattr(self, 'screen_width', 1920)
                                 scr_h = getattr(self, 'screen_height', 1080)
                                 h_dx = h_sx - (scr_w / 2)
                                 h_dy = h_sy - (scr_h / 2)
                                 curr_gaze = (curr_gaze[0] + h_dx, curr_gaze[1] + h_dy)
                             
                             # Drift = Target - Predicted
                             # Target is Center (Screen W/2, H/2)
                             scr_w = getattr(self, 'screen_width', 1920)
                             scr_h = getattr(self, 'screen_height', 1080)
                             tx, ty = scr_w/2, scr_h/2
                             
                             dx = tx - curr_gaze[0]
                             dy = ty - curr_gaze[1]
                             
                             # Set Drift (Accumulate or Replace? Replace is safer for 'Recenter')
                             self.gaze_drift = (dx, dy)
                             print(f"Quick Center Drift Applied: {dx}, {dy}")
                             
                             # End
                             self.calibration_step = 999 # Finish
                     
                     
                     else:
                        self.calibration.add_point(pupil_xy, world_xy)
                        print(f"Added Calibration Point {self.calibration_step}: Pupil={pupil_xy} -> World={world_xy}")
                 
                     # HYBRID/HEAD MODE: Capture Head Zero at Center Point AND Buffer Data
                     # If we are calibrating Eye (Grid) but in Hybrid Mode, we need to Zero the head at Center.
                     if not self.btn_eye_only.isChecked():
                         if self.head_tracker and self.head_gaze_data:
                             h_yaw, h_pitch, _, _ = self.head_gaze_data
                             
                             # Buffer for Auto-Tune ONLY if Grid Setup (not quick drift)
                             if self.cal_mode == "grid":
                                 self.head_cal_points.append((h_yaw, h_pitch, world_xy[0], world_xy[1]))
                             
                             # Zero if Center (Always Zero on Center target)
                             if norm_xy == (0.5, 0.5):
                                 self.head_tracker.set_calibration(h_yaw, h_pitch)
                                 print(f"Captured Head Zero at Center: Yaw={h_yaw}, Pitch={h_pitch}")
            
            self.calibration_step += 1
            if self.calibration_step < len(self.calibration_targets):
                self.cal_status_lbl.setText(f"Look at Target {self.calibration_step+1}/{len(self.calibration_targets)} -> Press 'C'")
                if self.current_map_mode.startswith("screen"):
                    self.update_overlay_target()
                else:
                    self.update_overlay_target() # Force repaint
            else:
                # Done
                if self.cal_mode == "head":
                    self.head_calibration.compute_model()
                    self.cal_status_lbl.setText(f"Head Calibration Complete. RMSE: {self.head_calibration.rmse:.2f}px")
                else:
                    self.calibration.compute_model()
                    
                    # HYBRID: Auto-Tune Head Tracker
                    if not self.btn_eye_only.isChecked():
                        self.auto_tune_head_tracker()
                        
                    self.cal_status_lbl.setText(f"Calibration Complete. RMSE: {self.calibration.rmse:.2f}px")
                
                self.is_calibrating = False
                self.cal_capture_btn.setEnabled(False)
                
                # USER_REQUEST: Auto-enable gaze cursor after calibration
                if not self.show_gaze_cursor:
                    self.check_gaze_cursor.setChecked(True)
                    self.toggle_gaze_cursor_option()
                
                if self.current_map_mode.startswith("screen"):
                   self.calibration_overlay.hide()
    
    def update_overlay_target(self):
        if self.calibration_step < len(self.calibration_targets):
             # These are now NORMALIZED (0.0-1.0)
             nx, ny = self.calibration_targets[self.calibration_step]
             
             # Map to current overlay size
             w = self.calibration_overlay.width()
             h = self.calibration_overlay.height()
             tx = int(nx * w)
             ty = int(ny * h)
             
             self.calibration_overlay.set_target(tx, ty)
             if self.cal_mode == "quick_drift":
                  self.calibration_overlay.set_message("Quick Center: Align Head & Eye -> Press 'C'")
             else:
                  self.calibration_overlay.set_message(f"Look at Target {self.calibration_step+1}/{len(self.calibration_targets)} -> Press 'C'")

    def capture_calibration_point(self):
        if not self.is_calibrating:
            return
        if self.cal_mode == "center":
            if self.center_cal_step == 1:
                self.center_cal_step = 2
                self.cal_status_lbl.setText("Step 2: Align scene center to screen center -> Press 'C'")
                return
            if self.center_cal_step == 2:
                self.start_calibration_collection()
                return
        else:
            # Grid mode
            self.start_calibration_collection()

    def handle_c_shortcut(self):
        """Handle 'C' key press: Capture if calibrating, Quick Center if not."""
        if self.is_calibrating:
            self.capture_calibration_point()
        else:
            # Quick Center (Drift Correction)
            self.start_calibration_routine(mode="quick_drift")
            
    def handle_ctrl_c_shortcut(self):
        """Handle 'Ctrl+C': Start Full 9-Point Calibration."""
        if not self.is_calibrating:
            self.start_calibration_routine(mode="grid")

    def video_mouse_press(self, event):
        """Handle mouse clicks on the video label."""
        # If in calibration mode, clicking acts like pressing 'C'
        if self.is_calibrating:
             print("Video clicked during calibration -> Capturing point")
             self.capture_calibration_point()
        
        # Ensure focus returns to main window on click
        self.setFocus()
        
    def keyPressEvent(self, event):
        # print(f"Key Pressed: {event.key()}")
        if False: # event.key() == Qt.Key_C: # Redundant (Handled by Shortcut)
             if self.is_calibrating:
                  self.capture_calibration_point()
             else:
                  if event.modifiers() & Qt.ControlModifier:
                       self.start_calibration_routine(mode="grid") # 9-Point
                  else:
                       # Quick Center (Drift Correction)
                       self.start_calibration_routine(mode="quick_drift")
                 
        elif event.key() == Qt.Key_Space:
            # Space bar toggles recording session (operator convenience)
            self.toggle_session()
        elif event.key() == Qt.Key_Escape:
            # Cancel any active calibration
            if self.is_calibrating:
                self.is_calibrating = False
                self.cal_collecting = False
                self.cal_buffer = []
                self.calibration_overlay.hide()
                self.cal_status_lbl.setText("Calibration Cancelled")
                self.cal_capture_btn.setEnabled(False)
                self.center_cal_step = 0
                print("Calibration Cancelled by Escape key")
            
            if self.is_fitting_globe:
                self.is_fitting_globe = False
                self.cal_status_lbl.setText("Auto-Fit Cancelled")
                print("Auto-Fit Cancelled by Escape key")
        elif event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
                self.menuBar().show()
            else:
                self.showFullScreen()
                self.menuBar().hide()
        super().keyPressEvent(event)


    def eventFilter(self, source, event):
        if source == self.eye_video_label:
            if event.type() == QEvent.MouseButtonPress:
                self.eye_view_mouse_press(event)
                return True
            elif event.type() == QEvent.MouseMove:
                self.eye_view_mouse_move(event)
                # Don't return True here so others can see it? Or just return True if processed.
                # If dragging, return True.
                if self.dragging_globe:
                    return True
            elif event.type() == QEvent.MouseButtonRelease:
                self.eye_view_mouse_release(event)
                return True
            elif event.type() == QEvent.Wheel:
                self.eye_view_wheel_event(event)
                return True
        return super().eventFilter(source, event)

    def get_image_coords(self, widget_pos):
        """Maps Widget Coordinates to Image Coordinates (640x480) handling KeepAspectRatio."""
        label_w = self.eye_video_label.width()
        label_h = self.eye_video_label.height()
        img_w, img_h = 640.0, 480.0
        
        if label_w == 0 or label_h == 0:
            return 320, 240
            
        ratio_w = label_w / img_w
        ratio_h = label_h / img_h
        scale = min(ratio_w, ratio_h)
        
        view_w = img_w * scale
        view_h = img_h * scale
        
        off_x = (label_w - view_w) / 2
        off_y = (label_h - view_h) / 2
        
        img_x = (widget_pos.x() - off_x) / scale
        img_y = (widget_pos.y() - off_y) / scale
        
        return int(img_x), int(img_y)

    def eye_view_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging_globe = True
            gx, gy = self.get_image_coords(event.pos())
            
            # Absolute Teleport as requested: "Wherever I click -> Center"
            # Update Sliders (triggers update_globe_center)
            self.globe_x_slider.setValue(gx)
            self.globe_y_slider.setValue(gy)
            print(f"Globe Teleport Press: {gx}, {gy}")

    def eye_view_mouse_move(self, event):
        if self.dragging_globe:
            gx, gy = self.get_image_coords(event.pos())
            
            # Absolute Teleport
            self.globe_x_slider.setValue(gx)
            self.globe_y_slider.setValue(gy)
            # print(f"Globe Drag: {gx}, {gy}")

    def eye_view_mouse_release(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging_globe = False
            # print("Globe Drag Ended")

    def eye_view_wheel_event(self, event):
        # Fine Adjustment
        delta = event.angleDelta().y()
        step = 3 # Increased speed (was 1, user asked for faster)
        
        current_r = self.globe_slider.value()
        if delta > 0:
            current_r += step
        else:
            current_r -= step
            
        # Update Slider (triggers update_globe_radius)
        self.globe_slider.setValue(current_r)
        print(f"Adjusting Globe Radius: {current_r}")

    def resizeEvent(self, event):
        # Resize overlay/canvas if needed
        if hasattr(self, 'calibration_overlay'):
             self.calibration_overlay.resize(self.size())
             if self.is_calibrating:
                 self.update_overlay_target()
        super().resizeEvent(event)

    def on_dock_floating(self, dock, floating):
        if floating:
            # Set Title for Window (User doesn't want "Python")
            if dock.objectName() == "DockTuning":
                dock.setWindowTitle("Tuning & Source")
            elif dock.objectName() == "DockWorkflow":
                dock.setWindowTitle("Workflow")
            elif dock.objectName() == "DockEye":
                dock.setWindowTitle("Eye View")
            elif dock.objectName() == "DockHead":
                 dock.setWindowTitle("Head View")
                 
            # Auto-Size to content but enforce a taller default for comfort
            # User request: "at least twice the size" vertically
            
            # Get current size hint
            sh = dock.sizeHint()
            target_h = max(800, sh.height() * 2) # Double it, or at least 800px
            
            # Clamp to screen height (avoid going off screen)
            screen_geom = QApplication.primaryScreen().availableGeometry()
            if target_h > screen_geom.height() * 0.9:
                target_h = int(screen_geom.height() * 0.9)
            
            dock.resize(sh.width(), target_h)
        else:
            # Clear title when docked to keep header clean (if logic allows)
            # dock.setWindowTitle("") 
            pass
            
    # ... (Mouse handling needs to be robust, will add a simple version in a moment)



    def start_auto_fit_globe(self):
        """Starts the auto-fitting process using pupil path intersection."""
        self.is_fitting_globe = True
        self.fit_start_time = time.time()
        self.globe_fitter = GlobeFitter() # Reset fitter
        # Disable buttons
        self.auto_fit_btn.setEnabled(False)
        self.cal_status_lbl.setText("Fitting Globe... Move eyes!")
        
    def finalize_auto_fit(self):
        self.is_fitting_globe = False
        self.auto_fit_btn.setEnabled(True)
        
        # Compute intersection
        center = self.globe_fitter.compute_average_intersection(640, 480)
        if center:
            cx, cy = center
            self.globe_center = (cx, cy)
            
            # Estimate radius (max distance from center)
            # We can iterate through stored rays to find max dist, or just use a default/heuristic
            # For now, let's keep the existing radius or update if we had data
            # Orlosky updates 'max_observed_distance'
            
            # Simple heuristic:
            max_dist = 0
            for ellipse in self.globe_fitter.ray_lines:
                pupil_center = ellipse[0]
                dist = np.sqrt((pupil_center[0] - cx)**2 + (pupil_center[1] - cy)**2)
                if dist > max_dist:
                    max_dist = dist
            
            # Update radius slider if reasonable
            if 30 < max_dist < 2000:
                self.globe_radius = float(max_dist)
                self.globe_slider.setValue(int(max_dist))
                
            # Update Center Sliders
            self.globe_x_slider.setValue(int(cx))
            self.globe_y_slider.setValue(int(cy))
            
            # Quality Check: Warn if radius is unreasonably large
            if max_dist > 400:
                self.cal_status_lbl.setText(f"⚠️ Globe Fit: Radius={int(max_dist)}px TOO LARGE! Re-do calibration.")
                QMessageBox.warning(self, "Globe Calibration Warning", 
                    f"Globe radius ({int(max_dist)}px) is very large compared to camera resolution.\n\n"
                    "This may indicate:\n"
                    "• Not enough eye movement during calibration\n"
                    "• Camera too close to eye\n"
                    "• Lighting issues affecting pupil detection\n\n"
                    "Recommendation: Click 'Reset Globe Data' and try again, making wider eye movements.")
            else:
                self.cal_status_lbl.setText(f"Globe Fit: ({int(cx)}, {int(cy)}) R={int(self.globe_radius)}")
        else:
            self.cal_status_lbl.setText("Globe Fit Failed: No convergence")
    
    def reset_globe_data(self):
        """Clear accumulated globe fitting data for fresh calibration start."""
        self.globe_fitter = GlobeFitter()
        self.cal_status_lbl.setText("Globe data reset. Ready for fresh calibration.")
        print("Globe fitting data cleared - starting fresh")


    def load_calibration(self):
        if self.calibration.load_calibration():
            self.cal_status_lbl.setText(f"Loaded! RMSE: {self.calibration.rmse:.2f}mm")
        else:
            self.cal_status_lbl.setText("Load Failed")
    
    def _calculate_angle_coverage(self, angles):
        """
        Calculate angular diversity coverage (0-100%).
        Measures how well-distributed the eye positions are across different angles.
        """
        if len(angles) < 2:
            return 0.0
        
        # Bin angles into 12 sectors (30° each)
        bins = [0] * 12
        for angle in angles:
            # Normalize angle to 0-360
            normalized = angle % 360
            bin_idx = int(normalized / 30) % 12
            bins[bin_idx] = 1
        
        # Coverage = percentage of bins filled
        coverage = (sum(bins) / 12.0) * 100.0
        return coverage
    
    def _get_quality_color(self, value, low_thresh, high_thresh):
        """
        Map quality value to color gradient: red (bad) → orange → yellow → green (good).
        
        Args:
            value: Current quality value
            low_thresh: Threshold for low quality (red)
            high_thresh: Threshold for high quality (green)
        
        Returns:
            (B, G, R) tuple for OpenCV
        """
        if value < low_thresh:
            # Red
            return (0, 0, 255)
        elif value >= high_thresh:
            # Green
            return (0, 255, 0)
        else:
            # Interpolate between red → orange → yellow → green
            progress = (value - low_thresh) / (high_thresh - low_thresh)
            
            if progress < 0.33:
                # Red → Orange (255,0,0) → (255,165,0)
                t = progress / 0.33
                g = int(165 * t)
                return (0, g, 255)
            elif progress < 0.66:
                # Orange → Yellow (255,165,0) → (255,255,0)
                t = (progress - 0.33) / 0.33
                g = int(165 + 90 * t)
                return (0, g, 255)
            else:
                # Yellow → Green (255,255,0) → (0,255,0)
                t = (progress - 0.66) / 0.34
                r = int(255 * (1 - t))
                return (0, 255, r)


    def toggle_session(self):
        if not self.session.is_running:
            self.session.start()
            self.heatmap.reset()
            self.gaze_path = [] # Reset path on new session
            
            if self.record_video:
                 self.video_recorder.start(640, 480, self.session.session_id)
                 
            self.start_sess_btn.setText("Stop Session")
        else:
            self.session.stop()
            
            # Stop video and get temp path
            saved_video_path = None
            if self.video_recorder.is_recording:
                saved_video_path = self.video_recorder.stop()
                
            self.start_sess_btn.setText("▶ Start Recording Session")
            
            # Handle Video Save As
            if saved_video_path and os.path.exists(saved_video_path):
                import shutil
                import datetime
                
                # Default logic
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                default_name = f"video_{timestamp}.mp4"
                
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Video Recording", default_name, 
                    "MP4 Video (*.mp4)"
                )
                
                if file_path:
                    try:
                        # Move temp file to user location
                        # Ensure target doesn't exist to avoid shutil errors on some systems or duplicate overwrite issues
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        shutil.move(saved_video_path, file_path)
                        print(f"Video saved to {file_path}")
                    except Exception as e:
                        print(f"Error saving video: {e}")
                else:
                    # User Cancelled -> Delete temp file
                    try:
                        os.remove(saved_video_path)
                        print("Video save cancelled. Temp file deleted.")
                    except Exception as e:
                        print(f"Error deleting temp video: {e}")

    def toggle_video_option(self):
        self.record_video = self.check_video.isChecked()
        # Just toggle state, no text update needed (color handles it)
        pass
        
    def toggle_path_option(self):
        self.show_gaze_path = self.check_path.isChecked()
        if not self.show_gaze_path:
            self.gaze_path = []
    
    def toggle_heatmap_option(self):
        self.show_heatmap = self.check_heatmap.isChecked()
        # NOTE: Data is NOT reset when hiding - use Clear button explicitly

    def update_heatmap_res(self, index):
        """Update Proxy Resolution for Heatmap Preview."""
        resolutions = [320, 640, 1920, 3840]
        if 0 <= index < len(resolutions):
            self.heatmap_proxy_width = resolutions[index]
            # Reset cache to force redraw
            self.heatmap_overlay_cache = None
            print(f"Heatmap Proxy Resolution set to: {self.heatmap_proxy_width}px")

    def _on_blur_changed_v2(self, value):
        """Update Gaussian Blur Radius."""
        # Ensure odd number
        if value % 2 == 0: value += 1
        self.heatmap_blur_radius = value
        self.lbl_blur_val.setText(f"Blur: {value}")
        # Reset cache
        self.heatmap_overlay_cache = None

    def _on_opacity_changed_v2(self, value):
        """Update Heatmap Opacity (0-100%)."""
        self.heatmap_alpha = value / 100.0
        self.lbl_opacity_val.setText(f"Opacity: {value}%")
        # Reset cache
        self.heatmap_overlay_cache = None
    
    def toggle_gaze_cursor_option(self):
        self.show_gaze_cursor = self.check_gaze_cursor.isChecked()
    
    def clear_heatmap_data(self):
        """Clear all accumulated heatmap data."""
        self.heatmap.reset()
        self.heatmap_overlay_cache = None
        print("Heatmap data cleared")

    def export_json(self):
        if not self.session.samples:
            print("No data to export")
            return
            
        # Ask user where to save
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        default_name = f"drawing_session_{timestamp}.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export nTopology JSON", default_name, 
            "JSON Files (*.json)"
        )
        
        if not file_path:
            return

        export_to_ntopology(self.session.samples, self.heatmap.get_normalized_grid(), 
                            self.heatmap.width_mm, self.heatmap.height_mm, file_path)
        print(f"JSON exported to {file_path}")

    def export_obj(self):
        if not self.session.samples:
            print("No data to export")
            return
            
        # Ask user where to save
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        default_name = f"heightmap_mesh_{timestamp}.obj"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Mesh OBJ", default_name, 
            "OBJ Files (*.obj)"
        )
        
        if not file_path:
            return

        # User Request: Match Export Smoothing to Visual Blur Slider CONGRUENTLY
        # We now use the exact same logic as HeatmapAccumulator.render_overlay
        
        blur_val = self.slider_heatmap_blur.value()
        
        # Consistent Reference Width (1280px)
        # This matches the logic in HeatmapAccumulator.render_overlay
        REF_WIDTH = 1280.0
        
        scale = self.heatmap.cols / REF_WIDTH
        visual_sigma = blur_val / 6.0 
        grid_sigma = visual_sigma * scale
            
        print(f"Export Mesh: Applying congruent smoothing. Slider={blur_val}, GridCols={self.heatmap.cols}, Sigma={grid_sigma:.4f}")
        
        # Get smoothed grid using SIGMA
        smoothed_grid = self.heatmap.get_smoothed_grid_sigma(grid_sigma)

        export_heightfield_obj(smoothed_grid, 
                               self.heatmap.width_mm, self.heatmap.height_mm, output_path=file_path)
        print(f"Mesh exported to {file_path}")

    def export_heatmap_png(self):
        """Export heatmap as PNG files - both transparent overlay and combined with source."""
        import datetime
        
        # Check if we have heatmap data
        if np.max(self.heatmap.grid) == 0:
            QMessageBox.warning(self, "No Data", "No heatmap data to export. Enable heatmap and look around first.")
            return
        
        # Determine resolution from source
        if self.media_mode and self.media_original_size:
            width, height = self.media_original_size
        elif self.latest_scene_frame is not None:
            height, width = self.latest_scene_frame.shape[:2]
        else:
            width, height = 640, 480
        
        # Generate timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ask user where to save
        default_name = f"heatmap_{timestamp}.png"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Heatmap PNG", default_name, 
            "PNG Files (*.png)"
        )
        
        if not filepath:
            return
        
        # Remove extension to create base path
        base_path = filepath.rsplit('.', 1)[0]
        
        # 1. Save transparent heatmap only (for printing on transparent paper)
        # Scale blur to match proxy preview settings (Base 320px)
        # Export logic: Blur = Slider * (ExportW / 320)
        base_w = 320.0
        scale = width / base_w 
        scaled_blur = int(self.heatmap_blur_radius * scale)
        if scaled_blur % 2 == 0: scaled_blur += 1 # Ensure odd
        
        heatmap_only_path = f"{base_path}_overlay.png"
        self.heatmap.save_png(heatmap_only_path, width, height, alpha=0.85, blur_size=scaled_blur)
        
        # 2. Save combined with source image (if available)
        source_frame = None
        if self.media_mode:
            if self.media_image is not None:
                source_frame = self.media_image.copy()
            elif self.media_video_cap is not None and self.last_valid_media_frame is not None:
                source_frame = self.last_valid_media_frame.copy()
        elif self.latest_scene_frame is not None:
            source_frame = self.latest_scene_frame.copy()
        
        if source_frame is not None:
            combined_path = f"{base_path}_combined.png"
            self.heatmap.save_combined(combined_path, source_frame, alpha=0.6, blur_size=scaled_blur)
            print(f"Heatmap exported to {combined_path} and {heatmap_only_path}")
        else:
            print(f"Heatmap overlay exported to {heatmap_only_path}")

    # --- Heatmap Control Methods (Added for Stability) ---
    def update_heatmap_res(self, index):
        # 0=0.5K, 1=1K, 2=2K, 3=4K
        resolutions = [320, 640, 1280, 2560] # Proxy widths
        if 0 <= index < len(resolutions):
            self.heatmap_proxy_width = resolutions[index]
            # Reset cache
            self.heatmap_overlay_cache = None
            print(f"Heatmap Proxy Resolution set to width: {self.heatmap_proxy_width}")

    def update_heatmap_blur(self, value):
        self.heatmap_blur_radius = value
        # Ensure odd number
        if self.heatmap_blur_radius % 2 == 0:
            self.heatmap_blur_radius += 1
        if hasattr(self, 'lbl_blur'):
            self.lbl_blur.setText(f"Blur: {self.heatmap_blur_radius}")
        # Reset cache
        self.heatmap_overlay_cache = None

        self.heatmap_opacity = value / 100.0
        if hasattr(self, 'lbl_opacity'):
             self.lbl_opacity.setText(f"Opacity: {value}%")

    def update_heatmap_opacity(self, value):
        self.heatmap_opacity = value / 100.0
        if hasattr(self, 'lbl_opacity'):
             self.lbl_opacity.setText(f"Opacity: {value}%")


    def _draw_qt_text_overlays(self, pixmap):
        """Draws high-quality text and UI elements using QPainter on the video frame."""
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        w = pixmap.width()
        
        # Calculate Scale Factor (Reference 1280px width)
        scale_factor = max(0.8, w / 1280.0)
        
        # Use a clearer variable for scaled font size
        base_font_size = 14
        scaled_font_size = int(base_font_size * scale_factor)
        
        # Configure Font (Roboto -> Segoe UI -> Sans Serif fallbacks)
        font = QFont("Roboto", scaled_font_size)
        if not QFontInfo(font).exactMatch():
            font = QFont("Segoe UI", scaled_font_size)
        font.setStyleStrategy(QFont.PreferAntialias)
        painter.setFont(font)
        
        # --- NO CAMERA MESSAGE / PLACEHOLDER ---
        # If we have no scene frame or it's just black/empty
        if not hasattr(self, 'latest_scene_frame') or self.latest_scene_frame is None:
             if hasattr(self, 'logo_pixmap') and self.logo_pixmap and not self.logo_pixmap.isNull():
                 # Draw Logo at 20% Opacity
                 painter.setOpacity(0.2)
                 
                 # Center logo
                 logo_w = self.logo_pixmap.width()
                 logo_h = self.logo_pixmap.height()
                 
                 # Scale logo if bigger than screen?
                 # Let's keep original unless huge. 
                 # Or scale to 50% of screen min dimension.
                 target_h = int(pixmap.height() * 0.5)
                 scaled_logo = self.logo_pixmap.scaledToHeight(target_h, Qt.SmoothTransformation)
                 
                 lx = (w - scaled_logo.width()) // 2
                 ly = (pixmap.height() - scaled_logo.height()) // 2
                 
                 painter.drawPixmap(lx, ly, scaled_logo)
                 painter.setOpacity(1.0) # Reset
             else:
                 # Fallback Text
                 msg = "No Scene Camera Detected"
                 font_lg = QFont(font)
                 font_lg.setPointSize(int(24 * scale_factor))
                 painter.setFont(font_lg)
                 painter.setPen(QColor(255, 100, 100))
                 painter.drawText(QRect(0, 0, w, pixmap.height()), Qt.AlignCenter, msg)
             
             painter.end()
             return

        # --- Calibration Instructions ---
        if self.is_calibrating:
            msg = ""
            if self.cal_mode == "center":
                if self.center_cal_step == 1:
                    msg = "Step 1: Reposition eye -> Press 'C'"
                else:
                    msg = "Step 2: Align scene center to screen center -> Press 'C'"
            else:
                msg = f"Look at Target {self.calibration_step + 1} / 9"
            
            # Use a modern "Tag" style background
            fm = painter.fontMetrics()
            text_rect = fm.boundingRect(msg)
            padding = int(20 * scale_factor)
            
            # Center the pill at top
            box_w = text_rect.width() + padding*2
            box_h = int(50 * scale_factor)
            box_x = (w - box_w) // 2
            box_y = int(40 * scale_factor)
            
            bg_rect = QRect(box_x, box_y, box_w, box_h)
            
            
            # Text
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(bg_rect, Qt.AlignCenter, msg)

        # --- Fitting Globe Instructions ---
        if self.is_fitting_globe:
            # Replicate the metrics calculation locally to display text
            # (The bars are still drawn by CV2 for now, we overlay the text)
            
            # Metrics
            total_samples = len(self.globe_fitter.ray_lines)
            angles = [ellipse[2] for ellipse in self.globe_fitter.ray_lines]
            angle_coverage = self._calculate_angle_coverage(angles)
            
            elapsed = time.time() - self.fit_start_time
            remaining = max(0, self.fit_duration - elapsed)
            
            confidence_pct = 0
            if total_samples >= 15: confidence_pct += min(50, (total_samples / 60.0) * 50)
            if angle_coverage >= 30: confidence_pct += min(50, (angle_coverage / 80.0) * 50)

            # Panel Geometry (Must match CV2 rects if possible, or we define new clean ones)
            # The CV2 code draws a panel at (20, 180). Let's overlay it nicely.
            
            panel_x = 20
            panel_y = 180
            panel_w = 600 # Fixed in CV2 code, should probably scale? 
            # CV2 code used fixed pixels. Let's stick to matching that for now or overriding it.
            # Actually, user wants CLEAN text.
            
            # Title
            painter.setPen(QColor(0, 255, 255))
            font_title = QFont(font)
            font_title.setBold(True)
            font_title.setPointSize(int(16 * scale_factor))
            painter.setFont(font_title)
            painter.drawText(panel_x + 20, panel_y + 30, "CALIBRATING GLOBE MODEL")
            
            painter.setFont(font)
            painter.setPen(QColor(220, 220, 220))
            
            # Bar Labels (Manual positioning to match the CV2 bars roughly)
            # Bar 1 (Samples) Y=230
            painter.drawText(panel_x + 25, 230 - 5, f"Samples: {total_samples}/60")
            
            # Bar 2 (Coverage) Y=270
            painter.drawText(panel_x + 25, 270 - 5, f"Coverage: {angle_coverage:.0f}%")
            
            # Bar 3 (Confidence) Y=310
            painter.drawText(panel_x + 25, 310 - 5, f"Confidence: {confidence_pct:.0f}%")
            
            # Timer Y=310 + 30
            painter.setPen(QColor(150, 150, 255))
            painter.drawText(panel_x + 25, 340 + 5, f"Time Remaining: {remaining:.1f}s")


        painter.end()


# To handle mouse clicks on the video, we need to subclass QLabel or install an event filter.
# I will patch the MainWindow to handle clicks on the video_label.

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    
    # Simple mouse click handling for calibration
    def video_clicked(event):
        if window.is_calibrating and window.cap_scene is not None:
            # Map widget coords to image coords
            # This is tricky because of AspectRatio scaling. 
            # For this MVP, let's just assume we are clicking roughly.
            # A real app needs proper coordinate transform.
            
            # Mocking calibration for now if clicked anywhere
            # In real usage, we'd get event.pos() and transform it.
            
            # Let's just simulate capturing the current pupil position as a corner if available?
            # Or just clicking the screen.
            
            # For MVP: Let's assume the user positions the pupil at the corner and presses a key.
            pass

    # Better approach for MVP: Use Key presses for calibration
    # 'C' to capture current pupil position as next corner.
    
    window.show()
    sys.exit(app.exec())
