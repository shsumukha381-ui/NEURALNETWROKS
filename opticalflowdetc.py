
import cv2
import numpy as np


class OpticalFlowDetector:

    
    def __init__(self, motion_threshold=2.0, high_motion_threshold=5.0):
        """
        Args:
            motion_threshold: Minimum motion intensity to consider (pixels/frame)
            high_motion_threshold: High motion intensity threshold for alerts
        """
        self.motion_threshold = motion_threshold
        self.high_motion_threshold = high_motion_threshold
        self.prev_gray = None
        
        # Farneback optical flow parameters (optimized for RTX 3050)
        self.flow_params = dict(
            pyr_scale=0.5,      # Image pyramid scale
            levels=3,           # Number of pyramid layers
            winsize=15,         # Averaging window size
            iterations=3,       # Iterations at each pyramid level
            poly_n=5,           # Polynomial expansion neighborhood
            poly_sigma=1.2,     # Gaussian standard deviation
            flags=0
        )
        
        # Grid parameters for region analysis
        self.grid_rows = 8
        self.grid_cols = 8
        
    def calculate_optical_flow(self, frame):
        """
        Calculate dense optical flow using Farneback algorithm
        
        Args:
            frame: Current BGR frame
            
        Returns:
            flow: Optical flow array (H, W, 2) with dx, dy motion vectors
            motion_magnitude: Motion intensity map (H, W)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize previous frame
        if self.prev_gray is None:
            self.prev_gray = gray
            return None, None
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, **self.flow_params
        )
        
        # Calculate motion magnitude (speed)
        motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Update previous frame
        self.prev_gray = gray
        
        return flow, motion_magnitude
    
    def create_motion_heatmap(self, motion_magnitude, frame_shape):
        """
        Create color-coded heatmap overlay
        
        Args:
            motion_magnitude: Motion intensity map
            frame_shape: Original frame shape (H, W, C)
            
        Returns:
            heatmap: BGR heatmap image
            heatmap_overlay: Semi-transparent overlay
        """
        if motion_magnitude is None:
            return None, None
        
        # Normalize motion magnitude to 0-255
        motion_norm = np.clip(motion_magnitude * 20, 0, 255).astype(np.uint8)
        
        # Apply Gaussian blur for smoother heatmap
        motion_smooth = cv2.GaussianBlur(motion_norm, (15, 15), 0)
        
        # Create color heatmap (Blue -> Green -> Yellow -> Red)
        heatmap = cv2.applyColorMap(motion_smooth, cv2.COLORMAP_JET)
        
        # Create semi-transparent overlay (30% opacity)
        h, w = frame_shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        return heatmap_resized, motion_smooth
    
    def analyze_motion_regions(self, motion_magnitude, frame_shape):
        """
        Analyze motion in grid regions to detect high-motion areas
        
        Args:
            motion_magnitude: Motion intensity map
            frame_shape: Original frame shape
            
        Returns:
            high_motion_regions: List of (row, col, intensity) tuples
            avg_motion: Average motion intensity across frame
            max_motion: Maximum motion intensity
        """
        if motion_magnitude is None:
            return [], 0.0, 0.0
        
        h, w = frame_shape[:2]
        region_h = h // self.grid_rows
        region_w = w // self.grid_cols
        
        high_motion_regions = []
        motion_values = []
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Extract region
                y1 = row * region_h
                y2 = (row + 1) * region_h
                x1 = col * region_w
                x2 = (col + 1) * region_w
                
                region = motion_magnitude[y1:y2, x1:x2]
                avg_intensity = np.mean(region)
                motion_values.append(avg_intensity)
                
                # Check if high motion region
                if avg_intensity > self.high_motion_threshold:
                    high_motion_regions.append((row, col, avg_intensity))
        
        avg_motion = np.mean(motion_values)
        max_motion = np.max(motion_values)
        
        return high_motion_regions, avg_motion, max_motion
    
    def detect_suspicious_activity(self, motion_magnitude, skeleton_count, frame_shape):
        """
        Detect suspicious activity based on motion and skeleton detection
        
        Logic:
        - High motion + No skeletons = Suspicious (crowded scene)
        - High motion + Few skeletons = Suspicious (occlusion)
        - Low motion = Normal
        
        Args:
            motion_magnitude: Motion intensity map
            skeleton_count: Number of detected skeletons
            frame_shape: Original frame shape
            
        Returns:
            is_suspicious: Boolean
            reason: String explanation
            motion_score: Float 0-1
        """
        if motion_magnitude is None:
            return False, "No motion data", 0.0
        
        # Analyze motion regions
        high_motion_regions, avg_motion, max_motion = self.analyze_motion_regions(
            motion_magnitude, frame_shape
        )
        
        # Calculate motion score (0-1)
        motion_score = min(avg_motion / 10.0, 1.0)
        
        # Detection logic
        is_suspicious = False
        reason = "Normal"
        
        # Case 1: High motion but no skeletons detected (crowded/occluded)
        if len(high_motion_regions) >= 3 and skeleton_count == 0:
            is_suspicious = True
            reason = f"High motion ({len(high_motion_regions)} regions) but no skeletons detected"
        
        # Case 2: Very high motion with few skeletons (partial occlusion)
        elif max_motion > self.high_motion_threshold * 1.5 and skeleton_count < 2:
            is_suspicious = True
            reason = f"Very high motion (max: {max_motion:.1f}) with limited skeleton detection"
        
        # Case 3: Widespread motion (potential crowd activity)
        elif len(high_motion_regions) >= 5:
            is_suspicious = True
            reason = f"Widespread motion detected ({len(high_motion_regions)} regions)"
        
        return is_suspicious, reason, motion_score
    
    def draw_motion_overlay(self, frame, motion_magnitude, heatmap, 
                           high_motion_regions, skeleton_count, 
                           is_suspicious, motion_score):
        """
        Draw motion heatmap and information overlay on frame
        
        Args:
            frame: Original BGR frame
            motion_magnitude: Motion intensity map
            heatmap: Color heatmap
            high_motion_regions: List of high-motion regions
            skeleton_count: Number of detected skeletons
            is_suspicious: Suspicious activity flag
            motion_score: Motion intensity score
            
        Returns:
            frame_with_overlay: Frame with heatmap and info
        """
        if heatmap is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Blend heatmap with original frame (30% opacity)
        overlay = frame.copy()
        cv2.addWeighted(heatmap, 0.3, overlay, 0.7, 0, overlay)
        
        # Draw grid and highlight high-motion regions
        region_h = h // self.grid_rows
        region_w = w // self.grid_cols
        
        for row, col, intensity in high_motion_regions:
            x1 = col * region_w
            y1 = row * region_h
            x2 = (col + 1) * region_w
            y2 = (row + 1) * region_h
            
            # Draw red rectangle around high-motion region
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw intensity value
            cv2.putText(overlay, f"{intensity:.1f}", 
                       (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw motion info panel
        panel_h = 150
        cv2.rectangle(overlay, (0, h - panel_h), (300, h), (0, 0, 0), -1)
        
        y_offset = h - panel_h + 25
        cv2.putText(overlay, "OPTICAL FLOW ANALYSIS", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(overlay, f"Motion Score: {motion_score:.2f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(overlay, f"High Motion Regions: {len(high_motion_regions)}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(overlay, f"Skeletons Detected: {skeleton_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        status_color = (0, 0, 255) if is_suspicious else (0, 255, 0)
        status_text = "SUSPICIOUS" if is_suspicious else "NORMAL"
        cv2.putText(overlay, f"Status: {status_text}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        return overlay
    
    def reset(self):
        """Reset optical flow detector (clear previous frame)"""
        self.prev_gray = None


def test_optical_flow():
    """Test optical flow detector with webcam"""
    print("Testing Optical Flow Detector...")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    detector = OpticalFlowDetector(motion_threshold=2.0, high_motion_threshold=5.0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate optical flow
        flow, motion_magnitude = detector.calculate_optical_flow(frame)
        
        if motion_magnitude is not None:
            # Create heatmap
            heatmap, _ = detector.create_motion_heatmap(motion_magnitude, frame.shape)
            
            # Analyze motion
            high_motion_regions, avg_motion, max_motion = detector.analyze_motion_regions(
                motion_magnitude, frame.shape
            )
            
            # Detect suspicious activity (assume 0 skeletons for testing)
            is_suspicious, reason, motion_score = detector.detect_suspicious_activity(
                motion_magnitude, 0, frame.shape
            )
            
            # Draw overlay
            frame_with_overlay = detector.draw_motion_overlay(
                frame, motion_magnitude, heatmap, high_motion_regions,
                0, is_suspicious, motion_score
            )
            
            cv2.imshow("Optical Flow Detection", frame_with_overlay)
        else:
            cv2.imshow("Optical Flow Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_optical_flow()
