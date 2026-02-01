import os
import sys
import subprocess

def create_shortcut():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # VBScript expects paths to be absolute
    main_script = os.path.join(root_dir, "drawing_analysis", "app", "main.py")
    icon_path = os.path.join(root_dir, "app_icon.ico")
    shortcut_path = os.path.join(root_dir, "EyeTracker.lnk")
    
    python_exe = sys.executable
    pythonw_exe = python_exe.replace("python.exe", "pythonw.exe")
    if not os.path.exists(pythonw_exe):
        pythonw_exe = python_exe

    # VBScript Content
    # To handle spaces in paths, we wrap the path in quotes.
    # In VBScript, a double quote inside a string is written as "".
    # So to get: Arguments = "C:\Path\File.py"
    # We write: Arguments = """C:\Path\File.py"""
    
    vbs_content = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{shortcut_path}"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{pythonw_exe}"
oLink.Arguments = \"""{main_script}\"""
oLink.WorkingDirectory = "{root_dir}"
oLink.IconLocation = "{icon_path}"
oLink.Description = "Eye Tracking Drawing Analysis"
oLink.Save
"""
    
    vbs_path = os.path.join(root_dir, "make_shortcut.vbs")
    
    try:
        with open(vbs_path, "w") as f:
            f.write(vbs_content)
        
        print("Running VBScript...")
        subprocess.run(["cscript", "//Nologo", vbs_path], check=True)
        print(f"Shortcut created: {shortcut_path}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists(vbs_path):
            os.remove(vbs_path)

if __name__ == "__main__":
    create_shortcut()
