import json
import time
import os
import pickle

class Session:
    def __init__(self, output_dir="sessions"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.is_running = False
        self.start_time = 0
        self.samples = [] # List of (timestamp, x_mm, y_mm, pupil_data)
        self.session_id = None
        
    def start(self):
        self.is_running = True
        self.start_time = time.time()
        self.samples = []
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        print(f"Session {self.session_id} started.")
        
    def stop(self):
        self.is_running = False
        duration = time.time() - self.start_time
        print(f"Session stopped. Duration: {duration:.2f}s. Samples: {len(self.samples)}")
        self.save_raw_data()
        
    def add_sample(self, timestamp, x_mm, y_mm, pupil_data):
        if not self.is_running:
            return
        self.samples.append({
            't': timestamp,
            'x': x_mm,
            'y': y_mm,
            'pupil': pupil_data
        })
        
    def save_raw_data(self):
        """
        Save raw samples to disk immediately for safety.
        """
        if not self.session_id:
            return
            
        filename = os.path.join(self.output_dir, f"session_{self.session_id}_raw.pkl")
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.samples, f)
            print(f"Raw data saved to {filename}")
        except Exception as e:
            print(f"Failed to save raw data: {e}")

    def get_duration(self):
        if self.is_running:
            return time.time() - self.start_time
        return 0
