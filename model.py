
import torch
import torch.nn as nn
import numpy as np


class AdvancedAnomalyModel(nn.Module):
 
    def __init__(self, input_size=105, hidden_size=128, num_layers=3):
        super(AdvancedAnomalyModel, self).__init__()
        
        # Bidirectional LSTM with Dropout for regularization
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classification head with progressive dimensionality reduction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Tensor of shape (batch_size, sequence_length=30, features=105)
        Returns:
            Tensor of shape (batch_size, 1) with anomaly probability
        """
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]  # Take final hidden state
        return self.classifier(last_time_step)


def compute_advanced_features(current_landmarks, prev_landmarks, other_persons_pos):
    """
    Feature Engineering Engine: Fuses 99 raw coordinates with 6 physics-based features
    
    Args:
        current_landmarks: np.array of shape (99,) - Current frame's x,y,z coordinates
        prev_landmarks: np.array of shape (99,) - Previous frame's coordinates
        other_persons_pos: List of np.array - Hip centroids of other detected persons
    
    Returns:
        np.array of shape (105,) - Complete feature vector
    """
    # Feature 1 & 2: Wrist Velocity (Euclidean distance change)
    # Left wrist (landmark 15): indices 45-47 (x,y,z)
    v_left_wrist = np.linalg.norm(
        current_landmarks[15*3:15*3+2] - prev_landmarks[15*3:15*3+2]
    )
    
    # Right wrist (landmark 16): indices 48-50 (x,y,z)
    v_right_wrist = np.linalg.norm(
        current_landmarks[16*3:16*3+2] - prev_landmarks[16*3:16*3+2]
    )
    
    # Feature 3: Social Proximity (minimum distance to nearest person)
    # Using hip centroid (landmark 23): indices 69-71
    my_center = current_landmarks[23*3:23*3+2]
    min_dist = 999.0
    
    if len(other_persons_pos) > 0:
        for other_center in other_persons_pos:
            dist = np.linalg.norm(my_center - other_center)
            if dist < min_dist:
                min_dist = dist
    
    # Feature 4 & 5: Acceleration/Jerk (rate of velocity change)
    # Placeholder for future implementation - requires velocity history
    acceleration_left = 0.0
    acceleration_right = 0.0
    
    # Feature 6: Reserved for future expansion (e.g., posture angle)
    reserved_feature = 0.0
    
    # Concatenate: 99 raw + 6 engineered = 105 total features
    extra_features = [
        v_left_wrist,
        v_right_wrist,
        min_dist,
        acceleration_left,
        acceleration_right,
        reserved_feature
    ]
    
    return np.concatenate([current_landmarks, extra_features])


def get_risk_category(score):
    """
    Decision-Making Logic: Categorize risk level based on model output
    
    Args:
        score: Float between 0.0 and 1.0 (model prediction)
    
    Returns:
        Tuple of (status_text, color_bgr, alert_level)
    """
    if score > 0.85:
        return "CRITICAL ANOMALY", (0, 0, 255), "RED"  # Red in BGR
    elif score > 0.50:
        return "SUSPICIOUS ACTIVITY", (0, 165, 255), "ORANGE"  # Orange in BGR
    else:
        return "NORMAL", (0, 255, 0), "GREEN"  # Green in BGR
