    """
SafetyNet AI - Real-Time Inference Module
Live behavioral anomaly detection with visual alerts
Target: 30+ FPS on NVIDIA RTX 3050
"""wad
import torch
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from datetime import datetime
import os
import winsound
import threading
from evidence_hasher import EvidenceHasher

# Explicit save directory
SAVE_DIR = os.path.join(os.getcwd(), "recordings")
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"✓ Recordings directory: {os.path.abspath(SAVE_DIR)}")


class SimpleAnomalyModel(torch.nn.Module):
    """
    Simplified LSTM model (matches training)
    """
    def __init__(self, input_size=99, hidden_size=64, num_layers=2):
        super(SimpleAnomalyModel, self).__init__()
        
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.classifier(last_step)


def get_risk_category(score):
    """
    Decision-Making Logic: Categorize risk level based on model output
    """
    if score > 0.85:
        return "CRITICAL ANOMALY", (0, 0, 255), "RED"  # Red in BGR
    elif score > 0.50:
        return "SUSPICIOUS ACTIVITY", (0, 165, 255), "ORANGE"  # Orange in BGR
    else:
        return "NORMAL", (0, 255, 0), "GREEN"  # Green in BGR


class SafetyNetInference:
    """
    Real-time anomaly detection system with live camera feed
    """
    def __init__(self, mode="SCHOOL", weights_path=None):
        """
        Args:
            mode: "SCHOOL" or "SHOP"
            weights_path: Path to trained model weights (optional)
        """
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize evidence hasher
        self.evidence_hasher = EvidenceHasher()
        
        # Video recording state
        self.video_writer = None
        self.recording = False
        self.recording_start_time = None
        self.current_video_path = None
        self.anomaly_frames = []
        
        # Load model (99 features, matching training)
        self.model = SimpleAnomalyModel(input_size=99, hidden_size=64, num_layers=2).to(self.device)
        
        if weights_path is None:
            weights_path = f"{mode.lower()}_weights.pth"
        
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"OK Loaded weights from {weights_path}")
        else:
            print(f"Warning: No weights found at {weights_path}")
            print("Running with untrained model (for testing only)")
        
        self.model.eval()
        
        # MediaPipe Pose (Heavy Model)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=2,  # Heavy model for accuracy
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Temporal memory (30 frames = 1 second)
        self.person_memory = deque(maxlen=30)
        
        # Alert logging
        self.alert_log = []
        
        # Display configuration
        self.display_title = {
            "SCHOOL": "SafetyNet AI: School Security",
            "SHOP": "SafetyNet AI: Retail Monitor"
        }[mode]
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_time = datetime.now()
        
        # Sound alert control (prevent spam)
        self.last_beep_time = None  # Start as None so first beep always plays
        self.beep_cooldown = 2.0  # Minimum 2 seconds between beeps
    
    def run(self):
        """
        Main inference loop with live camera feed
        """
        cap = cv2.VideoCapture(0)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n{'='*60}")
        print(f"{self.display_title}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Target: 30+ FPS")
        print(f"Video: {frame_width}x{frame_height} @ {fps} FPS")
        print(f"Save Directory: {os.path.abspath(SAVE_DIR)}")
        print(f"\nPress 'Q' to quit\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            current_time = datetime.now()
            current_fps = 1 / (current_time - self.last_time).total_seconds()
            self.fps_history.append(current_fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            self.last_time = current_time
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            risk_score = 0.0
            is_anomaly = False
            
            if results.pose_landmarks:
                # Extract 99 raw coordinates (33 landmarks × 3 coordinates)
                raw_coords = []
                for lm in results.pose_landmarks.landmark:
                    raw_coords.extend([lm.x, lm.y, lm.z])
                raw_coords = np.array(raw_coords)
                
                # Add to memory (using raw 99 features)
                self.person_memory.append(raw_coords)
                
                # Predict anomaly when we have 30 frames
                if len(self.person_memory) == 30:
                    input_tensor = torch.FloatTensor(
                        np.array([list(self.person_memory)])
                    ).to(self.device)
                    
                    with torch.no_grad():
                        risk_score = self.model(input_tensor).item()
                    
                    # Different beep thresholds for different modes
                    beep_threshold = 0.50 if self.mode == "SHOP" else 0.85
                    
                    # Check if anomaly detected
                    if risk_score > beep_threshold:
                        is_anomaly = True
                        self._log_alert(risk_score, current_time)
                        self._play_warning_sound(risk_score)
                
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            
            # Get risk category
            status, color, alert_level = get_risk_category(risk_score)
            
            # Handle video recording for anomalies
            if is_anomaly:
                if not self.recording:
                    # Start recording
                    self._start_recording(frame_width, frame_height, fps)
                
                # Write frame to video
                if self.video_writer is not None:
                    self.video_writer.write(frame)
                    self.anomaly_frames.append({
                        'timestamp': current_time.isoformat(),
                        'risk_score': risk_score
                    })
            else:
                if self.recording:
                    # Stop recording and hash evidence
                    self._stop_recording_and_hash()
            
            # Draw UI overlay
            self._draw_ui(frame, status, color, risk_score, avg_fps, alert_level)
            
            # Display
            cv2.imshow(self.display_title, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up - stop recording if still active
        if self.recording:
            self._stop_recording_and_hash()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print alert summary
        if self.alert_log:
            print(f"\n{'='*60}")
            print(f"Alert Summary: {len(self.alert_log)} critical events detected")
            print(f"{'='*60}")
            for alert in self.alert_log[-10:]:  # Show last 10
                print(f"[{alert['time']}] Risk: {alert['score']:.2%}")
    
    def _draw_ui(self, frame, status, color, risk_score, fps, alert_level):
        """
        Draw the UI overlay with status, risk score, and FPS
        """
        h, w = frame.shape[:2]
        
        # Top banner
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
        
        # Title
        cv2.putText(frame, self.display_title, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Status with color coding
        cv2.putText(frame, f"Status: {status}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Risk confidence
        cv2.putText(frame, f"Risk Confidence: {risk_score:.2%}", (10, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # FPS counter (top right)
        fps_color = (0, 255, 0) if fps >= 30 else (0, 165, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # Alert level indicator (top right)
        indicator_x = w - 50
        indicator_color = color
        cv2.circle(frame, (indicator_x, 70), 20, indicator_color, -1)
        cv2.circle(frame, (indicator_x, 70), 20, (255, 255, 255), 2)
        
        # Critical alert flash
        if alert_level == "RED":
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
    
    def _log_alert(self, score, timestamp):
        """
        Log critical alerts for administrative review
        """
        self.alert_log.append({
            'time': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'score': score,
            'mode': self.mode
        })
    
    def _play_warning_sound(self, risk_score=0):
        """
        Play warning beep when critical anomaly detected (non-blocking)
        Uses cooldown to prevent sound spam
        """
        current_time = datetime.now()
        
        # Check cooldown (allow first beep immediately)
        should_beep = False
        if self.last_beep_time is None:
            should_beep = True
        else:
            time_since_last_beep = (current_time - self.last_beep_time).total_seconds()
            if time_since_last_beep >= self.beep_cooldown:
                should_beep = True
        
        if should_beep:
            beep_threshold = "50%" if self.mode == "SHOP" else "85%"
            print(f"🔊 BEEP! {self.mode} ({beep_threshold}) - Risk: {risk_score:.1%}")
            # Play beep in separate thread to avoid blocking video feed
            def beep_thread():
                try:
                    # Play triple beep for better noticeability
                    winsound.Beep(1500, 200)  # High pitch
                    winsound.Beep(1000, 300)  # Medium pitch
                    winsound.Beep(1500, 200)  # High pitch again
                except Exception as e:
                    print(f"⚠️ Beep error: {e}")
            
            threading.Thread(target=beep_thread, daemon=True).start()
            self.last_beep_time = current_time
    
    def _start_recording(self, width, height, fps):
        """
        Start recording anomaly video
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.mode}_anomaly_{timestamp}.mp4"
        self.current_video_path = os.path.join(SAVE_DIR, filename)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.current_video_path, fourcc, fps, (width, height))
        
        self.recording = True
        self.recording_start_time = datetime.now()
        self.anomaly_frames = []
        
        print(f"\n🎥 Started recording: {filename}")
        print(f"   Path: {os.path.abspath(self.current_video_path)}")
    
    def _stop_recording_and_hash(self):
        """
        Stop recording and hash the video evidence
        CRITICAL: Release video writer BEFORE hashing
        """
        if not self.recording or self.video_writer is None:
            return
        
        # Release video writer immediately
        self.video_writer.release()
        self.video_writer = None
        self.recording = False
        
        recording_duration = (datetime.now() - self.recording_start_time).total_seconds()
        
        print(f"🎥 Stopped recording: {os.path.basename(self.current_video_path)}")
        print(f"   Duration: {recording_duration:.1f} seconds")
        print(f"   Frames: {len(self.anomaly_frames)}")
        print(f"   Absolute path: {os.path.abspath(self.current_video_path)}")
        
        # Calculate statistics
        if self.anomaly_frames:
            risk_scores = [f['risk_score'] for f in self.anomaly_frames]
            avg_risk = sum(risk_scores) / len(risk_scores)
            max_risk = max(risk_scores)
            
            # Prepare incident info for evidence hasher
            incident_info = {
                "mode": self.mode,
                "anomaly_type": f"{self.mode}_Anomaly",
                "recording_start": self.recording_start_time.isoformat(),
                "recording_end": datetime.now().isoformat(),
                "duration_seconds": round(recording_duration, 2),
                "total_frames": len(self.anomaly_frames),
                "average_risk": round(avg_risk, 4),
                "maximum_risk": round(max_risk, 4)
            }
            
            # Hash evidence (async, after file is released)
            print(f"🔐 Hashing evidence...")
            self.evidence_hasher.hash_evidence_async(self.current_video_path, incident_info)
        
        # Reset state
        self.current_video_path = None
        self.recording_start_time = None
        self.anomaly_frames = []


if __name__ == "__main__":
    print("\n=== SafetyNet AI Inference ===\n")
    print("Select Mode:")
    print("1. SCHOOL (Ragging/Bullying Detection)")
    print("2. SHOP (Shoplifting Detection)")
    
    choice = input("\nEnter choice (1/2): ").strip()
    mode = "SCHOOL" if choice == "1" else "SHOP"
    
    # Optional: specify custom weights path
    custom_weights = input(f"\nWeights path (press Enter for default '{mode.lower()}_weights.pth'): ").strip()
    weights_path = custom_weights if custom_weights else None
    
    system = SafetyNetInference(mode=mode, weights_path=weights_path)
    system.run()
