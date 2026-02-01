import numpy as np
import random
import math

class GlobeFitter:
    def __init__(self):
        self.ray_lines = [] # Stores ((cx, cy), (major, minor), angle)
        self.stored_intersections = []
        self.max_rays = 300 # Increased from 100 for better stats
        self.max_intersections = 3000
        
        # Angular coverage tracking (8 sectors: N, NE, E, SE, S, SW, W, NW)
        self.angle_sectors = [0] * 8  # Count of samples per sector
        
    def get_angular_coverage(self):
        """Returns coverage percentage (0-100) based on how many sectors have data."""
        filled_sectors = sum(1 for count in self.angle_sectors if count >= 3)  # At least 3 samples per sector
        return (filled_sectors / 8) * 100
    
    def get_quality_metrics(self):
        """Returns dict with quality metrics for auto-stop decision."""
        return {
            'ray_count': len(self.ray_lines),
            'intersection_count': len(self.stored_intersections),
            'angular_coverage': self.get_angular_coverage(),
            'sectors': self.angle_sectors.copy(),
            'is_ready': len(self.ray_lines) >= 50 and self.get_angular_coverage() >= 60
        }
        
    def add_pupil_data(self, pupil_data):
        """
        Add a pupil observation.
        pupil_data should contain:
        - center: (x, y)
        - axes: (major, minor) or (width, height)
        - angle: degrees
        - confidence: float (0-1)
        """
        if not pupil_data:
            return
        
        # FILTER 1: Reject low confidence detections (blinks, occlusions)
        confidence = pupil_data.get('confidence', 1.0)
        if confidence < 0.65: # Stricter for better accuracy
            return
            
        center = pupil_data['center']
        axes = pupil_data['axes']
        angle = pupil_data['angle']
        
        # FILTER 2: Validate ellipse
        # Reject too small
        if len(axes) < 2 or min(axes) < 3: # Relaxed from 5
            return
            
        # FILTER 3: UNSTABLE ANGLE CHECK (Crucial for Orlosky method)
        # If ellipse is nearly circular, the angle is noise.
        major = max(axes)
        minor = min(axes)
        if major == 0: return
        ratio = minor / major
        if ratio > 0.85: # Stricter - need clearly elliptical pupils
            return
        
        # Format for processing: ((cx, cy), (major, minor), angle)
        # Note: cv2.fitEllipse returns (center, axes, angle) where axes is (minor, major) or (width, height)
        # We store it exactly as provided to be consistent with consumption
        ellipse = (center, axes, angle)
        
        self.ray_lines.append(ellipse)
        
        # Track angular sector (which direction is the eye looking?)
        # Map angle (0-180 from ellipse) to 8 sectors (0-360)
        # Ellipse angle is major axis orientation, so gaze is perpendicular (+90)
        gaze_angle = (angle + 90) % 360
        sector_index = int(gaze_angle / 45) % 8
        self.angle_sectors[sector_index] += 1
        
        # Prune if too many
        if len(self.ray_lines) > self.max_rays:
            self.ray_lines.pop(0)
            
    def compute_average_intersection(self, width, height, min_angle_diff=10):
        """
        Computes the average intersection point of normal vectors from the collected pupil ellipses.
        Uses deterministic all-pairs algorithm with robust outlier filtering.
        """
        if len(self.ray_lines) < 5: # Need reasonable amount
            print(f"AutoFit: Not enough elliptical rays ({len(self.ray_lines)})")
            return None

        intersections = []

        # DETERMINISTIC: Use all valid pairs instead of random sample
        # Limit total comparisons to avoid freezing if max_rays is huge
        # Recent 300 rays = 45000 pairs, doable.
        
        for i in range(len(self.ray_lines)):
            for j in range(i + 1, len(self.ray_lines)):
                line1 = self.ray_lines[i]
                line2 = self.ray_lines[j]

                angle1 = line1[2]
                angle2 = line2[2]

                # Increased threshold from 2° to 10° for better numerical stability
                # Relaxed back to 5° to allow smaller movements
                angle_diff = abs(angle1 - angle2)
                if angle_diff >= min_angle_diff:
                    intersection = self.find_line_intersection(line1, line2)
                    
                    # Ensure the intersection is within the frame bounds (with margin)
                    margin = 500 # Relaxed margin
                    if intersection and (-margin <= intersection[0] < width + margin) and (-margin <= intersection[1] < height + margin):
                        intersections.append(intersection)
                        self.stored_intersections.append(intersection) 

        print(f"AutoFit: Found {len(intersections)} intersections from {len(self.ray_lines)} rays.")

        # Prune stored intersections
        if len(self.stored_intersections) > self.max_intersections:
            self.stored_intersections = self.stored_intersections[-self.max_intersections:]

        if len(self.stored_intersections) < 5:
            print("AutoFit: No valid intersections found.")
            return None
        
        # OUTLIER FILTERING: Use MAD (Median Absolute Deviation) for robustness
        filtered_intersections = self._filter_outliers_mad(self.stored_intersections, threshold=2.0) # Stricter MAD

        if not filtered_intersections:
            print("AutoFit: All intersections filtered out as outliers.")
            # Fallback to pure median of all
            filtered_intersections = self.stored_intersections
            
        print(f"AutoFit: Using {len(filtered_intersections)} filtered points.")

        # Compute the median intersection point from filtered data (More robust than mean)
        avg_x = np.median([pt[0] for pt in filtered_intersections])
        avg_y = np.median([pt[1] for pt in filtered_intersections])
        
        return (avg_x, avg_y)
    
    def _filter_outliers_mad(self, intersections, threshold=3.0):
        """
        Filter outliers using Median Absolute Deviation (MAD) method.
        MAD is more robust to outliers than standard deviation.
        
        Args:
            intersections: List of (x, y) tuples
            threshold: Number of MADs away from median to consider outlier (default: 3.0)
        
        Returns:
            Filtered list of (x, y) tuples
        """
        if len(intersections) < 4:
            return intersections
        
        xs = np.array([pt[0] for pt in intersections])
        ys = np.array([pt[1] for pt in intersections])
        
        # Compute median
        median_x = np.median(xs)
        median_y = np.median(ys)
        
        # Compute MAD (Median Absolute Deviation)
        mad_x = np.median(np.abs(xs - median_x))
        mad_y = np.median(np.abs(ys - median_y))
        
        # Avoid division by zero
        if mad_x == 0:
            mad_x = 1e-6
        if mad_y == 0:
            mad_y = 1e-6
        
        # Filter points within threshold * MAD
        filtered = []
        for pt in intersections:
            x_dev = abs(pt[0] - median_x) / mad_x
            y_dev = abs(pt[1] - median_y) / mad_y
            
            if x_dev <= threshold and y_dev <= threshold:
                filtered.append(pt)
        
        # Return filtered list, or original if too few points remain
        return filtered if len(filtered) >= 3 else intersections

    def find_line_intersection(self, ellipse1, ellipse2):
        """
        Computes the intersection of two lines that are orthogonal to the surface of given ellipses.
        The gaze vector (normal to pupil) corresponds to the MINOR axis of the projected ellipse.
        Since ellipse angle usually denotes the major axis, we rotate by 90 degrees.
        """
        (cx1, cy1), axes1, angle1 = ellipse1
        (cx2, cy2), axes2, angle2 = ellipse2

        # Ensure we treat the angle as the major axis angle. 
        # Calculate normal angle (projection of gaze vector)
        # Gaze vector projects to the connection between pupil center and globe center.
        # This line aligns with the MINOR axis.
        # If 'angle' is major axis orientation, we need angle + 90.
        
        angle1_rad = np.deg2rad(angle1 + 90)
        angle2_rad = np.deg2rad(angle2 + 90)

        # Direction vectors (length doesn't matter for infinite line intersection, but using 1.0 for simplicity)
        dx1, dy1 = np.cos(angle1_rad), np.sin(angle1_rad)
        dx2, dy2 = np.cos(angle2_rad), np.sin(angle2_rad)

        # Line equations in parametric form:
        # P1 + t1 * V1 = P2 + t2 * V2
        # t1 * V1 - t2 * V2 = P2 - P1
        # [dx1  -dx2] [t1] = [cx2 - cx1]
        # [dy1  -dy2] [t2]   [cy2 - cy1]
        
        A = np.array([[dx1, -dx2], [dy1, -dy2]])
        B = np.array([cx2 - cx1, cy2 - cy1])

        if np.abs(np.linalg.det(A)) < 1e-6:
            return None

        try:
            t = np.linalg.solve(A, B)
            t1 = t[0]
            
            intersection_x = cx1 + t1 * dx1
            intersection_y = cy1 + t1 * dy1
            
            return (intersection_x, intersection_y)
        except np.linalg.LinAlgError:
            return None

    def estimate_radius(self, globe_center, pupil_center):
        """
        Estimates radius based on observed pupil distance from center.
        This roughly corresponds to max_observed_distance logic but smoothed.
        """
        dist = math.sqrt((pupil_center[0] - globe_center[0])**2 + (pupil_center[1] - globe_center[1])**2)
        # The pupil moves on the surface of the sphere.
        # The projected distance from center (in 2D image) is roughly r * sin(theta).
        # We want 'r'. If we assume extreme eye movements reach ~45 degrees, then dist approx 0.707 * r.
        # Orlosky just took 'max_observed_distance' as the radius proxy. 
        # Let's return the raw distance, the UI can max-pool it.
        return dist
