import PyInstaller.__main__
import os
import shutil

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, 'dist')
BUILD_DIR = os.path.join(BASE_DIR, 'build')
SPEC_FILE = os.path.join(BASE_DIR, 'DrawingAnalysis.spec')

# 2. Cleanup previous builds
print("Cleaning up previous builds...")
if os.path.exists(DIST_DIR):
    shutil.rmtree(DIST_DIR)
if os.path.exists(BUILD_DIR):
    shutil.rmtree(BUILD_DIR)
if os.path.exists(SPEC_FILE):
    os.remove(SPEC_FILE)

# 3. Define Assets to bundle
# Format: (Source Path, Destination Path in Dist)
# Note: On Windows, separator is ';'
add_data = [
    ('assets', 'assets'),
    ('eyetracker_base', 'eyetracker_base'),
    ('drawing_analysis/app/styles.py', 'drawing_analysis/app'),
    ('drawing_analysis/app/calibration.py', 'drawing_analysis/app'),
    ('drawing_analysis/app/camera_thread.py', 'drawing_analysis/app'),
    ('drawing_analysis/app/export_mesh.py', 'drawing_analysis/app'),
    ('drawing_analysis/app/export_ntopo.py', 'drawing_analysis/app'),
    ('drawing_analysis/app/gaze_mapping.py', 'drawing_analysis/app'),
    ('drawing_analysis/app/globe_fitting.py', 'drawing_analysis/app'),
    ('drawing_analysis/app/head_tracker.py', 'drawing_analysis/app'),
    ('drawing_analysis/app/heatmap.py', 'drawing_analysis/app'),
    ('drawing_analysis/app/pupil_adapter.py', 'drawing_analysis/app'),
    ('drawing_analysis/app/session.py', 'drawing_analysis/app'),
    ('drawing_analysis/app/ui.py', 'drawing_analysis/app'),
    ('drawing_analysis/app/video_recorder.py', 'drawing_analysis/app'),
]

# Convert to PyInstaller format string
add_data_args = []
for src, dst in add_data:
    if os.path.exists(src):
        add_data_args.append(f'--add-data={src}{os.pathsep}{dst}')
    else:
        print(f"WARNING: Source path not found: {src}")

# 4. Run PyInstaller
print("Starting PyInstaller Build...")
PyInstaller.__main__.run([
    'drawing_analysis/app/main.py',
    '--name=DrawingAnalysis',
    '--onedir',          # Directory based bundle (faster, easier to debug)
    '--windowed',        # No console window
    '--icon=app_icon_oval.ico',
    '--hidden-import=mediapipe',
    '--hidden-import=cv2',
    '--hidden-import=numpy',
    '--hidden-import=PySide6',
    '--hidden-import=sklearn.utils._typedefs', # Often missed by PyInstaller
    '--hidden-import=sklearn.neighbors._partition_nodes',
    '--hidden-import=tkinter',
    # '--debug=all',     # Enable for verbose debug output
] + add_data_args)

print(f"Build Complete! Executable should be in: {os.path.join(DIST_DIR, 'DrawingAnalysis', 'DrawingAnalysis.exe')}")
