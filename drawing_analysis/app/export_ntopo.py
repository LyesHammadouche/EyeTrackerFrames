import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def export_to_ntopology(session_data, heatmap_grid, width_mm, height_mm, output_path):
    """
    Export session data to a JSON format compatible with nTopology.
    
    Args:
        session_data (list): List of sample dicts.
        heatmap_grid (np.ndarray): 2D heatmap array.
        width_mm (float): Drawing width.
        height_mm (float): Drawing height.
        output_path (str): Output file path.
    """
    
    # 1. Format Gaze Path
    gaze_path = []
    for sample in session_data:
        gaze_path.append({
            "x": sample['x'],
            "y": sample['y'],
            "t": sample['t']
        })
        
    # 2. Format Heatmap (Flattened for JSON or keep as 2D list)
    # nTopology often imports CSV points or Voxel grids. 
    # We'll provide a structured grid format.
    
    rows, cols = heatmap_grid.shape
    heatmap_data = {
        "resolution_x": cols,
        "resolution_y": rows,
        "cell_size_x": width_mm / cols,
        "cell_size_y": height_mm / rows,
        "values": heatmap_grid.tolist() # 2D array
    }
    
    export_obj = {
        "metadata": {
            "width_mm": width_mm,
            "height_mm": height_mm,
            "total_samples": len(session_data)
        },
        "gaze_path": gaze_path,
        "heatmap": heatmap_data
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(export_obj, f, indent=2, cls=NumpyEncoder)
        print(f"nTopology JSON exported to {output_path}")
    except Exception as e:
        print(f"Failed to export JSON: {e}")
