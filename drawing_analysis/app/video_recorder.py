import cv2
import os
import time

class VideoRecorder:
    def __init__(self, output_dir="sessions", fps=30):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.writer = None
        self.fps = fps
        self.is_recording = False
        self.filename = None
        
    def start(self, width, height, session_id=None):
        if session_id is None:
            session_id = time.strftime("%Y%m%d_%H%M%S")
            
        self.filename = os.path.join(self.output_dir, f"session_{session_id}.mp4")
        
        # Define codec
        # mp4v is a safe bet for generic .mp4 on Windows/Linux
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        try:
            self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                print(f"Failed to open video writer: {self.filename}")
                self.writer = None
                return False
                
            self.is_recording = True
            print(f"Started recording video: {self.filename}")
            return True
        except Exception as e:
            print(f"Error starting video recorder: {e}")
            self.writer = None
            return False

    def write_frame(self, frame):
        if self.is_recording and self.writer is not None:
            # frame should be BGR (standard OpenCV)
            try:
                self.writer.write(frame)
            except Exception as e:
                print(f"Error writing frame: {e}")

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        
        saved_file = self.filename
        if self.is_recording:
            print(f"Stopped recording. Saved to {saved_file}")
            
        self.is_recording = False
        self.filename = None
        return saved_file
