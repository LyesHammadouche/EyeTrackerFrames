
MODERN_THEME_QSS = """
/* --- Global Reset & Fonts --- */
QWidget {
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    font-size: 13px;
    color: #333333; /* Default text for light mode */
}

/* --- Main Window & Central Area (Light Mode) --- */
QMainWindow, QWidget#CentralWidget {
    background-color: #F5F5F7; /* Light Gray/White Smoke */
}

/* --- Menus & Dialogs --- */
QMenuBar {
    background-color: #e0e0e0;
    color: #333;
}
QMenuBar::item:selected {
    background-color: #d0d0d0;
}
QMenu {
    background-color: #2d2d2d; /* Dark menu to match docks? Or Light? User said "Barres blanches... ça ne va pas". Let's match Dark theme */
    color: #f0f0f0;
    border: 1px solid #444;
}
QMenu::item:selected {
    background-color: #00e676;
    color: #000;
}
QMessageBox {
    background-color: #F5F5F7; /* Keep message box light for readability key with system buttons */
}
QLabel {
    color: #333; /* Default label */
}

/* Dock Separator (The gap between panels) */
QMainWindow::separator {
    background-color: #1E1E1E;
    width: 4px; 
    height: 4px;
}
QMainWindow::separator:hover {
    background-color: #00e676;
}

/* --- Dock Widgets (Dark Sidebars) --- */
QDockWidget {
    titlebar-close-icon: url(close.png);
    titlebar-normal-icon: url(float.png);
    color: #ffffff;
}

QDockWidget::title {
    text-align: left;
    background: #181818;
    color: #f0f0f0;
    padding: 8px;
    font-weight: bold;
    border-bottom: 1px solid #333;
}

/* The internal widget of the dock needs to be dark */
QWidget#DockContent {
    background-color: #1E1E1E;
    color: #e0e0e0;
}

/* Labels inside Docks */
QWidget#DockContent QLabel {
    color: #e0e0e0;
}

/* --- Scroll Areas --- */
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollBar:vertical {
    border: none;
    background: #2b2b2b;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #555;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* --- Buttons (Pill / Modern Shape) --- */
QPushButton {
    background-color: #4a4a4a; /* Darker gray - more visible */
    border: 1px solid #5a5a5a;
    border-radius: 6px;
    padding: 6px 12px;
    color: #f0f0f0; /* White text for contrast */
    font-weight: 600;
}
QPushButton:hover {
    background-color: #5a5a5a; /* Lighter on hover */
    border-color: #00e676; /* Green accent on hover */
}
QPushButton:pressed {
    background-color: #3a3a3a;
}
QPushButton:disabled {
    background-color: #2a2a2a;
    color: #666;
    border: 1px solid #333;
}

/* Primary Action Buttons (in dark panels) */
QWidget#DockContent QPushButton {
    background-color: #3d3d3d; /* Slightly lighter than dock background */
    border: 1px solid #555555;
    color: #f0f0f0;
}
QWidget#DockContent QPushButton:hover {
    background-color: #4d4d4d;
    border-color: #00e676; /* Green Accent */
}
QWidget#DockContent QPushButton:checked {
    background-color: #2e7d32;
    border-color: #00e676;
}
QWidget#DockContent QPushButton:disabled {
    background-color: #2a2a2a;
    color: #555;
    border-color: #333;
}

/* --- Inputs (Combo, Spin, LineEdit) --- */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 4px;
    background-color: #ffffff;
    color: #333;
}
/* Inputs in Dark Docks */
QWidget#DockContent QComboBox, 
QWidget#DockContent QSpinBox, 
QWidget#DockContent QDoubleSpinBox, 
QWidget#DockContent QLineEdit {
    background-color: #2d2d2d;
    border: 1px solid #444;
    color: #f0f0f0;
}

QWidget#DockContent QComboBox:hover, 
QWidget#DockContent QLineEdit:hover {
    border-color: #00e676;
}

/* Dropdown List (AbstractItemView) Fix for Dark Mode */
QWidget#DockContent QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    color: #f0f0f0;
    selection-background-color: #00e676;
    selection-color: #000000;
    border: 1px solid #444;
}

/* --- Checkboxes & Radios --- */
QCheckBox, QRadioButton {
    color: #333;
}
QWidget#DockContent QCheckBox, 
QWidget#DockContent QRadioButton {
    color: #f0f0f0; /* White text on dark background */
}
QWidget#DockContent QCheckBox::indicator, 
QWidget#DockContent QRadioButton::indicator {
    border: 1px solid #777;
    background: #333;
    width: 14px;
    height: 14px;
    border-radius: 2px;
}
QWidget#DockContent QCheckBox::indicator:checked {
    background: #00e676;
    border-color: #00e676;
    image: url(check.png); /* or just simple color */
}

/* --- Sliders --- */
QSlider::groove:horizontal {
    border: 1px solid #bbb;
    height: 4px;
    background: #e0e0e0;
    margin: 2px 0;
    border-radius: 2px;
}
QWidget#DockContent QSlider::groove:horizontal {
    border: 1px solid #3d3d3d;
    background: #2a2a2a;
}

QSlider::handle:horizontal {
    background: #00e676;
    border: 1px solid #00e676;
    width: 6px; /* Narrow width for "Tick" look */
    height: 16px; /* Taller than groove */
    margin: -6px 0; /* Center vertically */
    border-radius: 1px; /* Slight rounding but mostly rectangular */
}

/* --- Group Boxes --- */
QGroupBox {
    border: 1px solid #ddd;
    border-radius: 6px;
    margin-top: 1.2em;
    padding-top: 10px;
    font-weight: bold;
}
QWidget#DockContent QGroupBox {
    border: 1px solid #3d3d3d;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #333;
}
QWidget#DockContent QGroupBox::title {
    color: #00e676;
}

/* --- Tabs (if used) --- */
QTabWidget::pane { 
    border: 1px solid #ccc; 
}
QTabBar::tab {
    background: #e0e0e0;
    padding: 8px 12px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #fff;
    border-bottom: 2px solid #00e676;
}
"""
