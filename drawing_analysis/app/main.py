import sys
import os
from PySide6.QtWidgets import QApplication
from ui import MainWindow

if __name__ == "__main__":
    # Ensure we are in the correct directory so relative paths work
    if getattr(sys, 'frozen', False):
        # If frozen (PyInstaller), the executable is the anchor
        app_dir = os.path.dirname(sys.executable)
    else:
        # If running from source, this file is the anchor
        app_dir = os.path.dirname(os.path.abspath(__file__))
        
    os.chdir(app_dir)
    
    app = QApplication(sys.argv)
    
    # Set Window Icon
    # Assuming 'interface elements' is in the same directory as the executable/script
    from PySide6.QtGui import QIcon
    icon_path = os.path.join(app_dir, "interface elements", "Eye-fb-round.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        print(f"Icon not found at: {icon_path}")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
