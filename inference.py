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
        
        print(f"\n{'='*60}")
        print(f"{self.display_title}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Target: 30+ FPS")
        print(f"\nPress 'Q' to quit\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            current_time = datetime.now()
            fps = 1 / (current_time - self.last_time).total_seconds()
            self.fps_history.append(fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            self.last_time = current_time
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            risk_score = 0.0
            
            if results.pose_landmarks:
                # Extract 99 raw coordinates (33 landmarks × 3 coordinates)
                raw_coords = []
                for lm in results.pose_landmarks.landmark:
                    raw_coords.extend([lm.x, lm.y, lm.z])
                raw_coords = np.array(raw_coords)
