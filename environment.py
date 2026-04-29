
import sys
import os

print("="*70)
print("SafetyNet AI - Environment Diagnostic")
print("="*70)

# Check 1: Python version
print("\n[1] Python Version:")
print(f"    {sys.version}")
if sys.version_info < (3, 8):
    print("    ⚠️  WARNING: Python 3.8+ recommended")
else:
    print("    ✓ OK")

# Check 2: OneDrive path
print("\n[2] Project Location:")
current_path = os.getcwd()
print(f"    {current_path}")
if "OneDrive" in current_path:
    print("    ⚠️  WARNING: Project is in OneDrive folder!")
    print("    This can cause file-locking and access violation errors")
    print("    Recommendation: Move to C:\\Projects\\ or similar")
else:
    print("    ✓ OK - Not in OneDrive")

# Check 3: NumPy version
print("\n[3] NumPy Version:")
try:
    import numpy as np
    print(f"    {np.__version__}")
    
    major, minor = map(int, np.__version__.split('.')[:2])
    if major >= 2:
        print("    ⚠️  WARNING: NumPy 2.0+ detected!")
        print("    This can cause ABI compatibility issues with MediaPipe")
        print("    Recommendation: pip install numpy==1.24.3")
    else:
        print("    ✓ OK - Compatible version")
except ImportError:
    print("    ❌ ERROR: NumPy not installed")

# Check 4: OpenCV version
print("\n[4] OpenCV Version:")
try:
    import cv2
    print(f"    {cv2.__version__}")
    print("    ✓ OK")
except ImportError:
    print("    ❌ ERROR: OpenCV not installed")

# Check 5: MediaPipe version
print("\n[5] MediaPipe Version:")
try:
    import mediapipe as mp
    print(f"    {mp.__version__}")
    print("    ✓ OK")
except ImportError:
    print("    ❌ ERROR: MediaPipe not installed")
except Exception as e:
    print(f"    ❌ ERROR loading MediaPipe: {e}")

# Check 6: MediaPipe initialization test
print("\n[6] MediaPipe Initialization Test:")
try:
    import mediapipe as mp
    print("    Attempting to initialize Pose...")
    
    pose = mp.solutions.pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("    ✓ SUCCESS - MediaPipe initialized without crash")
    pose.close()
    
except Exception as e:
    print(f"    ❌ ERROR: {e}")
    print("    This is the 0xC0000005 error!")

# Check 7: Camera access
print("\n[7] Camera Access Test:")
try:
    import cv2
    
    # Try DirectShow backend
    print("    Testing DirectShow backend...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("    ✓ SUCCESS - Camera accessible with DirectShow")
        else:
            print("    ⚠️  WARNING: Camera opened but can't read frames")
        cap.release()
    else:
        print("    ⚠️  WARNING: DirectShow failed, trying default...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("    ✓ OK - Camera accessible with default backend")
            cap.release()
        else:
            print("    ❌ ERROR: Cannot access camera")
            
except Exception as e:
    print(f"    ❌ ERROR: {e}")

# Summary
print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

print("\nIf you see 0xC0000005 errors, try these fixes in order:")
print("\n1. DOWNGRADE NUMPY:")
print("   pip uninstall numpy")
print("   pip install numpy==1.24.3")

print("\n2. REINSTALL MEDIAPIPE:")
print("   pip uninstall mediapipe")
print("   pip install mediapipe==0.10.9")

print("\n3. MOVE PROJECT OUT OF ONEDRIVE:")
if "OneDrive" in current_path:
    print("   Current: " + current_path)
    print("   Move to: C:\\Projects\\SafetyNet\\")

print("\n4. USE DIRECTSHOW BACKEND:")
print("   Already implemented in fixed data_collector.py")

print("\n5. RUN AS ADMINISTRATOR:")
print("   Right-click Python script -> Run as Administrator")

print("\n6. UPDATE VISUAL C++ REDISTRIBUTABLES:")
print("   Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")

print("\n" + "="*70)
