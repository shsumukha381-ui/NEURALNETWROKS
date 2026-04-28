"""
SafetyNet AI - Multi-Class Data Collector
Collects pose landmark sequences for multiple anomaly classes
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime

print("="*70)
print("SafetyNet AI - Multi-Class Data Collector")
print("="*70)

# Class registry file
CLASS_REGISTRY_FILE = "class_registry.json"

def load_class_registry():
    """Load existing class registry or create new one"""
    if os.path.exists(CLASS_REGISTRY_FILE):
        with open(CLASS_REGISTRY_FILE, 'r') as f:
            return json.load(f)
    return {"classes": [], "class_to_id": {}, "id_to_class": {}}

def save_class_registry(registry):
    """Save class registry to file"""
    with open(CLASS_REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"✓ Class registry saved: {len(registry['classes'])} classes")

def add_class_to_registry(registry, class_name):
    """Add new class to registry if it doesn't exist"""
    if class_name not in registry['classes']:
        class_id = len(registry['classes'])
        registry['classes'].append(class_name)
        registry['class_to_id'][class_name] = class_id
        registry['id_to_class'][str(class_id)] = class_name
        save_class_registry(registry)
        print(f"✓ New class added: '{class_name}' (ID: {class_id})")
    else:
        print(f"✓ Class already exists: '{class_name}' (ID: {registry['class_to_id'][class_name]})")

def display_existing_classes(registry):
    """Display all existing classes"""
    if not registry['classes']:
        print("\nNo classes registered yet.")
        return
    
    print("\nExisting Classes:")
    print("-" * 40)
    for i, class_name in enumerate(registry['classes']):
        print(f"  {i}. {class_name}")
    print("-" * 40)

# Select mode
print("\nSelect Mode:")
print("1. SCHOOL (ragging, bullying, fights)")
print("2. SHOP (shoplifting, loitering)")
print("3. GENERAL (custom anomalies)")

mode_choice = input("\nEnter choice (1-3): ").strip()
mode_map = {"1": "SCHOOL", "2": "SHOP", "3": "GENERAL"}
mode = mode_map.get(mode_choice, "GENERAL")

print(f"\n✓ Mode selected: {mode}")

# Load class registry
registry = load_class_registry()
display_existing_classes(registry)

# Get class label
print("\nEnter Anomaly Class:")
print("  Examples: Fighting, Falling, Shoplifting, Vandalism, Loitering")
print("  Or select existing class by number")

class_input = input("\nClass name or number: ").strip()

# Check if input is a number (selecting existing class)
if class_input.isdigit() and 0 <= int(class_input) < len(registry['classes']):
    class_label = registry['classes'][int(class_input)]
    print(f"✓ Selected existing class: '{class_label}'")
else:
    class_label = class_input.title()  # Capitalize first letter
    print(f"✓ Using class: '{class_label}'")
    add_class_to_registry(registry, class_label)

# Create data directory
data_dir = os.path.join("MP_Data", mode, class_label)
os.makedirs(data_dir, exist_ok=True)

# Count existing sequences
existing_sequences = len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]) if os.path.exists(data_dir) else 0

print(f"\n{'='*70}")
print(f"Data Collection Setup:")
print(f"  Mode: {mode}")
print(f"  Class: {class_label}")
print(f"  Directory: {data_dir}")
print(f"  Existing sequences: {existing_sequences}")
print(f"{'='*70}")

# How many sequences to collect
num_sequences = int(input("\nHow many sequences to collect? (default 30): ").strip() or "30")
frames_per_sequence = 30

print(f"\n✓ Will collect {num_sequences} sequences of {frames_per_sequence} frames each")
print(f"✓ Total frames: {num_sequences * frames_per_sequence}")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print(f"\n{'='*70}")
print("Starting Data Collection")
print(f"{'='*70}")
print("\nInstructions:")
print("  - Perform the action: " + class_label)
print("  - Press SPACE to start recording a sequence")
print("  - Press Q to quit")
print(f"{'='*70}\n")

sequence_num = existing_sequences
collected = 0

while collected < num_sequences:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Process pose
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    # Draw skeleton
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
        )
    
    # UI overlay
    cv2.rectangle(frame, (0,0), (w,120), (0,0,0), -1)
    cv2.putText(frame, f"SafetyNet AI - Data Collector", (10,25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Mode: {mode} | Class: {class_label}", (10,55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Collected: {collected}/{num_sequences}", (10,85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, "Press SPACE to record | Q to quit", (10,110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    
    cv2.imshow('Data Collection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n✓ Collection stopped by user")
        break
    
    elif key == ord(' '):
        # Start recording sequence
        print(f"\nRecording sequence {sequence_num + 1}...")
        
        sequence_dir = os.path.join(data_dir, str(sequence_num))
        os.makedirs(sequence_dir, exist_ok=True)
        
        frame_num = 0
        recording = True
        
        while recording and frame_num < frames_per_sequence:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            
            if results.pose_landmarks:
                # Extract coordinates
                coords = []
                for lm in results.pose_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                coords = np.array(coords, dtype=np.float32)
                
                # Save frame
                np.save(os.path.join(sequence_dir, f"{frame_num}.npy"), coords)
                frame_num += 1
                
                # Draw skeleton
                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
                )
            
            # Recording UI
            cv2.rectangle(frame, (0,0), (w,120), (0,0,255), -1)
            cv2.putText(frame, "RECORDING...", (10,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
            cv2.putText(frame, f"Frame: {frame_num}/{frames_per_sequence}", (10,80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, f"Class: {class_label}", (10,110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(33)  # ~30 FPS
        
        if frame_num == frames_per_sequence:
            collected += 1
            sequence_num += 1
            print(f"✓ Sequence {sequence_num} saved ({frame_num} frames)")
        else:
            print(f"⚠️ Incomplete sequence (only {frame_num} frames)")

cap.release()
cv2.destroyAllWindows()
pose.close()

print(f"\n{'='*70}")
print("Data Collection Complete!")
print(f"  Mode: {mode}")
print(f"  Class: {class_label}")
print(f"  Sequences collected: {collected}")
print(f"  Total sequences: {sequence_num}")
print(f"  Data saved to: {data_dir}")
print(f"{'='*70}\n")

# Display registry summary
display_existing_classes(registry)

print("\nNext Steps:")
print(f"  1. Collect data for other classes")
print(f"  2. Run training: python train_multiclass.py")
print(f"  3. Test model: python app.py")
