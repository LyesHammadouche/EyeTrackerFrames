import numpy as np

def export_heightfield_obj(heatmap_grid, width_mm, height_mm, z_min_mm=0.0, z_max_mm=30.0, output_path="relief.obj"):
    """
    Export heatmap grid as a triangulated OBJ heightfield.
    
    Args:
        heatmap_grid (np.ndarray): 2D array of values (normalized 0-1 recommended).
        width_mm (float): Physical width.
        height_mm (float): Physical height.
        z_min_mm (float): Minimum Z height.
        z_max_mm (float): Maximum Z height.
        output_path (str): Output filename.
    """
    if heatmap_grid.ndim != 2:
        raise ValueError("grid must be a 2D array")

    h, w = heatmap_grid.shape
    
    # Normalize grid to 0-1 if not already (robustness)
    g_min = np.min(heatmap_grid)
    g_max = np.max(heatmap_grid)
    if g_max > g_min:
        g = (heatmap_grid - g_min) / (g_max - g_min)
    else:
        g = heatmap_grid # Flat or empty

    xs = np.linspace(0.0, width_mm, w, dtype=np.float32)
    ys = np.linspace(height_mm, 0.0, h, dtype=np.float32) # Top-to-bottom for image coords
    zs = z_min_mm + g * (z_max_mm - z_min_mm)

    # Build vertices
    # We can write directly to file to save memory for large grids
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# OBJ generated from heatmap\n")
            f.write(f"# Width: {width_mm}mm, Height: {height_mm}mm\n")
            
            # Write Vertices
            for i in range(h):
                for j in range(w):
                    f.write(f"v {xs[j]:.4f} {ys[i]:.4f} {zs[i, j]:.4f}\n")
            
            # Write Faces
            # 1-based indexing
            def vid(i, j):
                return i * w + j + 1

            for i in range(h - 1):
                for j in range(w - 1):
                    v1 = vid(i, j)
                    v2 = vid(i, j + 1)
                    v3 = vid(i + 1, j)
                    v4 = vid(i + 1, j + 1)

                    # Two triangles per quad
                    f.write(f"f {v1} {v3} {v4}\n")
                    f.write(f"f {v1} {v4} {v2}\n")
                    
        print(f"OBJ exported to {output_path}")
        
    except Exception as e:
        print(f"Failed to export OBJ: {e}")
