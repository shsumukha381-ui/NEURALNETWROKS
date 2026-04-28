import gradio as gr
import torch
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import tempfile
import os
import shutil
from datetime import datetime
import winsound
import threading
from optical_flow_detector import OpticalFlowDetector
from evidence_hasher import EvidenceHasher
import hashlib
import json

# Define permanent recordings folder
RECORDINGS_FOLDER = r'D:\hacthontest12314\recordings'
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
print(f"✓ Recordings folder: {RECORDINGS_FOLDER}")

print("="*70)
print("SafetyNet AI - Secure Admin Interface")
print("="*70)

# Initialize evidence hasher with proper path
SECURITY_LOG_PATH = os.path.join(RECORDINGS_FOLDER, "security_log.json")
evidence_hasher = EvidenceHasher(log_file=SECURITY_LOG_PATH)
print(f"✓ Evidence hasher initialized")
print(f"✓ Security log: {SECURITY_LOG_PATH}")

# ==================== USER AUTHENTICATION ====================
USER_DB_FILE = "users.json"

def initialize_user_db():
    """Initialize user database with default admin account"""
    if not os.path.exists(USER_DB_FILE):
        default_users = {
            "admin": {
                "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
                "role": "admin",
                "created": datetime.now().isoformat(),
                "last_login": None
            }
        }
        with open(USER_DB_FILE, 'w') as f:
            json.dump(default_users, f, indent=2)
        print("✓ User database initialized")
        print("  Default credentials: admin / admin123")
    else:
        print("✓ User database loaded")

initialize_user_db()

def authenticate_user(username, password):
    """Authenticate user credentials"""
    try:
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
        
        if username not in users:
            return False, "Invalid username or password"
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if users[username]["password_hash"] == password_hash:
            # Update last login
            users[username]["last_login"] = datetime.now().isoformat()
            with open(USER_DB_FILE, 'w') as f:
                json.dump(users, f, indent=2)
            
            return True, f"Welcome, {username}!"
        else:
            return False, "Invalid username or password"
    
    except Exception as e:
        return False, f"Authentication error: {e}"

# ==================== MODEL LOADING ====================
class EnhancedModel(torch.nn.Module):
    def __init__(self, input_size=120, hidden_size=128, num_layers=3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = torch.nn.Sequential(torch.nn.Linear(hidden_size*2, hidden_size), torch.nn.Tanh(), torch.nn.Linear(hidden_size, 1))
        self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_size*2, 128), torch.nn.ReLU(), torch.nn.Dropout(0.4), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(64, 1), torch.nn.Sigmoid())
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn, dim=1)
        return self.classifier(context)

def engineer_features(sequence):
    enhanced = []
    for i, frame in enumerate(sequence):
        features = list(frame)
        if i > 0:
            v_lw = np.linalg.norm(frame[45:47] - sequence[i-1][45:47])
            v_rw = np.linalg.norm(frame[48:50] - sequence[i-1][48:50])
            v_le = np.linalg.norm(frame[39:41] - sequence[i-1][39:41])
            v_re = np.linalg.norm(frame[42:44] - sequence[i-1][42:44])
        else:
            v_lw = v_rw = v_le = v_re = 0
        shoulder_width = np.linalg.norm(frame[33:35] - frame[36:38])
        hip_width = np.linalg.norm(frame[69:71] - frame[72:74])
        arm_elevation_l = frame[34] - frame[46]
        arm_elevation_r = frame[37] - frame[49]
        motion = np.sum(np.abs(frame - sequence[i-1])) if i > 0 else 0
        all_features = features + [v_lw, v_rw, v_le, v_re, 0, 0, 0, 0, shoulder_width, hip_width, 0, arm_elevation_l, arm_elevation_r, motion] + [0]*7
        enhanced.append(all_features[:120])
    return np.array(enhanced)

device = torch.device("cpu")
models = {}
model_info = {}

print("\nLoading models...")
for mode in ["SCHOOL", "SHOP"]:
    path = f"{mode.lower()}_enhanced_weights.pth"
    if os.path.exists(path):
        models[mode] = EnhancedModel(120, 128, 3)
        models[mode].load_state_dict(torch.load(path, map_location=device))
        models[mode].eval()
        
        file_size = os.path.getsize(path) / (1024*1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(path))
        model_info[mode] = {
            'size': file_size,
            'modified': mod_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"  {mode}: Loaded ({file_size:.1f} MB, updated {model_info[mode]['modified']})")

if not models:
    print("\nWARNING: No models found!")
else:
    print(f"\n{len(models)} model(s) ready")
print("="*70 + "\n")

# ==================== VIDEO PROCESSING ====================
def process(video, mode, threshold, progress=gr.Progress()):
    if not video:
        return None, "ERROR: Please upload a video file"
    
    if mode not in models:
        return None, f"ERROR: {mode} model not found. Please train the model first."
    
    progress(0, desc="Initializing...")
    
    print(f"\n{'='*70}")
    print(f"Testing {mode} Model")
    print(f"Threshold: {threshold}%")
    print(f"{'='*70}")
    
    try:
        progress(0.05, desc="Opening video...")
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            return None, "ERROR: Cannot open video file"
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo: {w}x{h}, {fps}fps, {total} frames, {total/fps:.1f}s")
        
        progress(0.1, desc="Creating output...")
        out_path = tempfile.mktemp(suffix='.mp4')
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        progress(0.15, desc="Initializing detection...")
        pose = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5)
        mp_draw = mp.solutions.drawing_utils
        optical_flow = OpticalFlowDetector(motion_threshold=2.0, high_motion_threshold=5.0)
        
        mem = deque(maxlen=30)
        scores = []
        anomalies = 0
        suspicious = 0
        normal = 0
        n = 0
        optical_flow_alerts = 0
        skeleton_failures = 0
        threshold_decimal = threshold / 100.0
        last_beep_time = datetime.now()
        beep_cooldown = 2.0
        
        print(f"Processing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            n += 1
            progress(0.15 + (n / total) * 0.75, desc=f"Frame {n}/{total}")
            
            risk = 0.0
            skeleton_detected = False
            flow, motion_magnitude = optical_flow.calculate_optical_flow(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            
            if results.pose_landmarks:
                skeleton_detected = True
                coords = []
                for lm in results.pose_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                coords = np.array(coords, dtype=np.float32)
                mem.append(coords)
                
                if len(mem) == 30:
                    sequence = np.array(list(mem))
                    sequence_120 = engineer_features(sequence)
                    tensor = torch.FloatTensor(np.array([sequence_120])).to(device)
                    with torch.no_grad():
                        risk = models[mode](tensor).item()
                    scores.append(risk)
                    
                    beep_threshold = 0.50 if mode == "SHOP" else threshold_decimal
                    
                    if risk > threshold_decimal:
                        anomalies += 1
                    elif risk > 0.5:
                        suspicious += 1
                    else:
                        normal += 1
                    
                    if risk > beep_threshold:
                        current_time = datetime.now()
                        if (current_time - last_beep_time).total_seconds() >= beep_cooldown:
                            def beep_thread():
                                try:
                                    winsound.Beep(1500, 200)
                                    winsound.Beep(1000, 300)
                                    winsound.Beep(1500, 200)
                                except: pass
                            threading.Thread(target=beep_thread, daemon=True).start()
                            last_beep_time = current_time
                
                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
                )
            
            optical_flow_suspicious = False
            motion_score = 0.0
            
            if motion_magnitude is not None:
                skeleton_count = 1 if skeleton_detected else 0
                optical_flow_suspicious, reason, motion_score = optical_flow.detect_suspicious_activity(
                    motion_magnitude, skeleton_count, frame.shape
                )
                
                if optical_flow_suspicious and not skeleton_detected:
                    optical_flow_alerts += 1
                    skeleton_failures += 1
                    risk = max(risk, motion_score * 0.7)
                
                heatmap, _ = optical_flow.create_motion_heatmap(motion_magnitude, frame.shape)
                if heatmap is not None:
                    frame = cv2.addWeighted(frame, 0.8, heatmap, 0.2, 0)
            
            if risk > threshold_decimal:
                status = "CRITICAL"
                color = (0, 0, 255)
            elif risk > 0.5 or optical_flow_suspicious:
                status = "SUSPICIOUS"
                color = (0, 165, 255)
            else:
                status = "NORMAL"
                color = (0, 255, 0)
            
            cv2.rectangle(frame, (0,0), (w,140), (0,0,0), -1)
            cv2.putText(frame, f"SafetyNet AI - {mode}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Status: {status}", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Risk: {risk:.1%}", (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Frame: {n}/{total}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, f"Motion: {motion_score:.2f}", (10,135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, f"Threshold: {threshold}%", (w-200,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            
            if risk > threshold_decimal:
                cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 10)
            
            out.write(frame)
        
        progress(0.9, desc="Finalizing...")
        
        cap.release()
        out.release()
        pose.close()
        
        avg = np.mean(scores) if scores else 0
        mx = max(scores) if scores else 0
        mn = min(scores) if scores else 0
        
        permanent_video_path = None
        final_output_path = out_path
        
        if anomalies > 0 or optical_flow_alerts > 0:
            progress(0.92, desc="Saving to recordings...")
            
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                permanent_filename = f"incident_{mode}_{timestamp}.mp4"
                permanent_video_path = os.path.join(RECORDINGS_FOLDER, permanent_filename)
                
                print(f"\n📁 Copying video to: {permanent_video_path}")
                shutil.copy2(out_path, permanent_video_path)
                
                if not os.path.exists(permanent_video_path):
                    raise FileNotFoundError(f"Copy failed: {permanent_video_path}")
                
                file_size_mb = os.path.getsize(permanent_video_path) / (1024 * 1024)
                print(f"✓ Video saved ({file_size_mb:.2f} MB)")
                
                progress(0.95, desc="Hashing evidence...")
                print(f"🔐 Generating SHA-256 hash...")
                
                if mx > threshold_decimal:
                    anomaly_type = f"{mode}_Critical_Anomaly"
                elif optical_flow_alerts > 0:
                    anomaly_type = f"{mode}_Optical_Flow_Anomaly"
                else:
                    anomaly_type = f"{mode}_Suspicious_Activity"
                
                incident_info = {
                    "mode": mode,
                    "anomaly_type": anomaly_type,
                    "detection_threshold": threshold,
                    "average_risk": round(avg, 4),
                    "maximum_risk": round(mx, 4),
                    "minimum_risk": round(mn, 4),
                    "critical_frames": anomalies,
                    "suspicious_frames": suspicious,
                    "normal_frames": normal,
                    "optical_flow_alerts": optical_flow_alerts,
                    "skeleton_failures": skeleton_failures,
                    "total_frames": total,
                    "video_duration_seconds": round(total/fps, 2),
                    "video_fps": fps,
                    "video_resolution": f"{w}x{h}"
                }
                
                evidence_hasher.hash_evidence_async(permanent_video_path, incident_info)
                
            except Exception as e:
                print(f"❌ Error saving/hashing: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*70}")
        print(f"Complete! Avg: {avg:.1%}, Max: {mx:.1%}, Critical: {anomalies}")
        if permanent_video_path and os.path.exists(permanent_video_path):
            print(f"Saved: {os.path.abspath(permanent_video_path)}")
        print(f"{'='*70}\n")
        
        video_location_info = ""
        if permanent_video_path and os.path.exists(permanent_video_path):
            video_location_info = f"\n📁 VIDEO SAVED:\n- Path: {os.path.abspath(permanent_video_path)}\n- Size: {os.path.getsize(permanent_video_path) / (1024*1024):.2f} MB\n- Hash: Logged in {SECURITY_LOG_PATH}\n"
        
        report = f"""{'='*60}
SAFETYNET AI - DETECTION REPORT
{'='*60}

MODEL: {mode} | Threshold: {threshold}%
Video: {total} frames, {total/fps:.1f}s, {w}x{h}
{video_location_info}
RISK ANALYSIS:
- Average: {avg:.2%}
- Maximum: {mx:.2%}
- Minimum: {mn:.2%}

CLASSIFICATION:
- Critical (>{threshold}%): {anomalies} frames ({100*anomalies/len(scores) if scores else 0:.1f}%)
- Suspicious (50-{threshold}%): {suspicious} frames ({100*suspicious/len(scores) if scores else 0:.1f}%)
- Normal (<50%): {normal} frames ({100*normal/len(scores) if scores else 0:.1f}%)

OPTICAL FLOW:
- Alerts: {optical_flow_alerts}
- Skeleton Failures: {skeleton_failures}

SUMMARY:
"""
        
        if mx > threshold_decimal:
            report += f"🚨 HIGH RISK - {anomalies} critical anomalies!\n"
        elif mx > 0.5:
            report += f"⚠️ MODERATE RISK - Suspicious activity detected.\n"
        else:
            report += f"✅ LOW RISK - Normal behavior.\n"
        
        report += f"\n{'='*60}\n"
        report += f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"{'='*60}"
        
        progress(1.0, desc="Done!")
        return final_output_path, report
        
    except Exception as e:
        import traceback
        error = f"ERROR: {str(e)}\n\n{traceback.format_exc()}"
        print(error)
        return None, error

# ==================== VIEW SECURITY LOG ====================
def view_security_log():
    """View all logged incidents"""
    try:
        if not os.path.exists(SECURITY_LOG_PATH):
            return "No security log found. Process a video with anomalies to create one."
        
        with open(SECURITY_LOG_PATH, 'r') as f:
            log_data = json.load(f)
        
        if not log_data.get("incidents"):
            return "No incidents logged yet."
        
        report = f"""{'='*70}
SECURITY LOG - INCIDENT HISTORY
{'='*70}

Total Incidents: {len(log_data['incidents'])}
Log Created: {log_data.get('created', 'Unknown')}
Last Updated: {log_data.get('last_updated', 'Unknown')}

{'='*70}
INCIDENTS:
{'='*70}

"""
        
        for i, incident in enumerate(log_data["incidents"], 1):
            report += f"\n[{i}] {incident.get('timestamp', 'Unknown')}\n"
            report += f"    File: {incident.get('filename', 'Unknown')}\n"
            report += f"    Type: {incident.get('anomaly_type', 'Unknown')}\n"
            report += f"    Risk: {incident.get('maximum_risk', 0):.2%}\n"
            report += f"    SHA-256: {incident.get('sha256_hash', 'Unknown')[:64]}...\n"
            report += f"    Size: {incident.get('file_size_mb', 0)} MB\n"
            report += "-" * 70 + "\n"
        
        return report
    
    except Exception as e:
        return f"Error reading security log: {e}"

# ==================== GRADIO INTERFACE ====================
custom_css = """
.login-container {
    max-width: 500px;
    margin: 50px auto;
    padding: 40px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}

.login-title {
    text-align: center;
    color: white;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 10px;
}

.login-subtitle {
    text-align: center;
    color: rgba(255,255,255,0.9);
    font-size: 16px;
    margin-bottom: 30px;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 20px;
    color: white;
    text-align: center;
}

.stat-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 10px;
}

.footer {
    text-align: center;
    padding: 20px;
    color: #666;
    font-size: 14px;
}
"""

def create_login_interface():
    """Create beautiful login interface"""
    with gr.Blocks(title="SafetyNet AI - Login") as login_demo:
        gr.HTML("""
        <div class="login-container">
            <div class="login-title">🛡️ SafetyNet AI</div>
            <div class="login-subtitle">Secure Admin Access</div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=2):
                username_input = gr.Textbox(
                    label="👤 Username",
                    placeholder="Enter your username",
                    max_lines=1
                )
                password_input = gr.Textbox(
                    label="🔒 Password",
                    placeholder="Enter your password",
                    type="password",
                    max_lines=1
                )
                login_btn = gr.Button("🚀 Login", variant="primary", size="lg")
                login_status = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown("""
                ---
                **Default Credentials:**
                - Username: `admin`
                - Password: `admin123`
                
                ⚠️ Change default password after first login!
                """)
            with gr.Column(scale=1):
                pass
        
        def login_handler(username, password):
            success, message = authenticate_user(username, password)
            if success:
                return f"✅ {message}\n\nRedirecting to main interface..."
            else:
                return f"❌ {message}"
        
        login_btn.click(
            fn=login_handler,
            inputs=[username_input, password_input],
            outputs=[login_status]
        )
    
    return login_demo

def create_main_interface():
    """Create main detection interface"""
    with gr.Blocks(title="SafetyNet AI - Admin Panel") as main_demo:
        gr.HTML("""
        <div class="main-header">
            <h1>🛡️ SafetyNet AI - Admin Control Panel</h1>
            <p>Advanced Fighting Detection & Evidence Management System</p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: Video Detection
            with gr.Tab("🎥 Video Detection"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Test Video")
                        
                        mode = gr.Radio(
                            ["SCHOOL", "SHOP"],
                            value="SCHOOL",
                            label="Detection Mode",
                            info="Select environment type"
                        )
                        
                        threshold = gr.Slider(
                            minimum=50,
                            maximum=95,
                            value=85,
                            step=5,
                            label="Detection Threshold (%)",
                            info="Risk % above this = Critical"
                        )
                        
                        test_btn = gr.Button("🚀 Analyze Video", variant="primary", size="lg")
                        
                        if model_info:
                            model_status = "📊 Models Loaded:\n"
                            for m, info in model_info.items():
                                model_status += f"✓ {m}: {info['size']:.1f} MB\n"
                            gr.Markdown(f"```\n{model_status}```")
                    
                    with gr.Column():
                        video_output = gr.Video(label="Annotated Result")
                        report_output = gr.Textbox(
                            label="Detection Report",
                            lines=25,
                            max_lines=35
                        )
                
                gr.Markdown("""
                ### 📖 How to Use:
                1. **Upload Video**: Select a video file to analyze
                2. **Choose Mode**: SCHOOL or SHOP environment
                3. **Set Threshold**: Adjust sensitivity (85% recommended)
                4. **Analyze**: Click to process and view results
                
                ### 🎯 Understanding Results:
                - 🔴 **Critical**: Risk > Threshold (Fighting detected)
                - 🟠 **Suspicious**: Risk 50-Threshold% (Uncertain)
                - 🟢 **Normal**: Risk < 50% (Normal behavior)
                
                ### 💾 Evidence Storage:
                - Videos with anomalies are automatically saved
                - SHA-256 hash generated for integrity
                - Location: `D:\\hacthontest12314\\recordings`
                """)
                
                # Connect button to processing function (must be inside the Tab)
                test_btn.click(
                    fn=process,
                    inputs=[video_input, mode, threshold],
                    outputs=[video_output, report_output]
                )
            
            # Tab 2: Security Log
            with gr.Tab("📋 Security Log"):
                gr.Markdown("### 🔐 Incident History & Evidence Chain")
                
                refresh_btn = gr.Button("🔄 Refresh Log", variant="secondary")
                log_output = gr.Textbox(
                    label="Security Log",
                    lines=30,
                    max_lines=40
                )
                
                gr.Markdown(f"""
                ### 📁 Storage Information:
                - **Recordings Folder**: `{RECORDINGS_FOLDER}`
                - **Security Log**: `{SECURITY_LOG_PATH}`
                - **Hash Algorithm**: SHA-256
                
                All incident videos are cryptographically hashed for evidence integrity.
                """)
                
                refresh_btn.click(
                    fn=view_security_log,
                    outputs=[log_output]
                )
            
            # Tab 3: System Info
            with gr.Tab("ℹ️ System Info"):
                gr.Markdown("""
                # 🛡️ SafetyNet AI - System Information
                
                ## 📊 System Status
                """)
                
                system_info = f"""
**Models Loaded**: {len(models)}
**Recordings Folder**: {RECORDINGS_FOLDER}
**Security Log**: {SECURITY_LOG_PATH}
**Hash Algorithm**: SHA-256
**Detection Modes**: SCHOOL, SHOP

## 🔧 Model Details:
"""
                for mode, info in model_info.items():
                    system_info += f"\n**{mode} Model**:\n"
                    system_info += f"- Size: {info['size']:.1f} MB\n"
                    system_info += f"- Updated: {info['modified']}\n"
                
                gr.Markdown(system_info)
                
                gr.Markdown("""
                ## 🔐 Security Features:
                - ✅ Admin authentication
                - ✅ SHA-256 evidence hashing
                - ✅ Tamper-proof logging
                - ✅ Chain of custody tracking
                - ✅ Automatic evidence archival
                
                ## 📞 Support:
                For technical support or questions, contact your system administrator.
                """)
        
        gr.HTML("""
        <div class="footer">
            <p>🛡️ SafetyNet AI v2.0 | Secure Evidence Management System</p>
            <p>© 2026 SafetyNet AI | All Rights Reserved</p>
        </div>
        """)
    
# ==================== COMBINED INTERFACE WITH LOGIN ====================
def create_app():
    """Create app with login gate"""
    with gr.Blocks(title="SafetyNet AI - Secure Admin Panel") as app:
        # Session state
        logged_in = gr.State(False)
        
        # Login Section
        with gr.Column(visible=True) as login_section:
            gr.HTML("""
            <div style="max-width: 500px; margin: 50px auto; padding: 40px; 
                 background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3);">
                <div style="text-align: center; color: white; font-size: 32px; 
                     font-weight: bold; margin-bottom: 10px;">
                    🛡️ SafetyNet AI
                </div>
                <div style="text-align: center; color: rgba(255,255,255,0.9); 
                     font-size: 16px; margin-bottom: 30px;">
                    Secure Admin Access
                </div>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    pass
                with gr.Column(scale=2):
                    username_input = gr.Textbox(
                        label="👤 Username",
                        placeholder="Enter your username",
                        max_lines=1
                    )
                    password_input = gr.Textbox(
                        label="🔒 Password",
                        placeholder="Enter your password",
                        type="password",
                        max_lines=1
                    )
                    login_btn = gr.Button("🚀 Login", variant="primary", size="lg")
                    login_status = gr.Textbox(label="Status", interactive=False)
                    
                    gr.Markdown("""
                    ---
                    **Default Credentials:**
                    - Username: `admin`
                    - Password: `admin123`
                    
                    ⚠️ Change default password after first login!
                    """)
                with gr.Column(scale=1):
                    pass
        
        # Main Interface Section (hidden by default)
        with gr.Column(visible=False) as main_section:
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 padding: 30px; border-radius: 15px; margin-bottom: 20px; 
                 color: white; text-align: center;">
                <h1>�️ SafetyNet AI - Admin Control Panel</h1>
                <p>Advanced Fighting Detection & Evidence Management System</p>
            </div>
            """)
            
            with gr.Tabs():
                # Tab 1: Video Detection
                with gr.Tab("🎥 Video Detection"):
                    with gr.Row():
                        with gr.Column():
                            video_input = gr.Video(label="Upload Test Video")
                            
                            mode = gr.Radio(
                                ["SCHOOL", "SHOP"],
                                value="SCHOOL",
                                label="Detection Mode",
                                info="Select environment type"
                            )
                            
                            threshold = gr.Slider(
                                minimum=50,
                                maximum=95,
                                value=85,
                                step=5,
                                label="Detection Threshold (%)",
                                info="Risk % above this = Critical"
                            )
                            
                            test_btn = gr.Button("🚀 Analyze Video", variant="primary", size="lg")
                            
                            if model_info:
                                model_status = "📊 Models Loaded:\n"
                                for m, info in model_info.items():
                                    model_status += f"✓ {m}: {info['size']:.1f} MB\n"
                                gr.Markdown(f"```\n{model_status}```")
                        
                        with gr.Column():
                            video_output = gr.Video(label="Annotated Result")
                            report_output = gr.Textbox(
                                label="Detection Report",
                                lines=25,
                                max_lines=35
                            )
                    
                    gr.Markdown("""
                    ### 📖 How to Use:
                    1. **Upload Video**: Select a video file to analyze
                    2. **Choose Mode**: SCHOOL or SHOP environment
                    3. **Set Threshold**: Adjust sensitivity (85% recommended)
                    4. **Analyze**: Click to process and view results
                    
                    ### 🎯 Understanding Results:
                    - 🔴 **Critical**: Risk > Threshold (Fighting detected)
                    - 🟠 **Suspicious**: Risk 50-Threshold% (Uncertain)
                    - 🟢 **Normal**: Risk < 50% (Normal behavior)
                    
                    ### 💾 Evidence Storage:
                    - Videos with anomalies are automatically saved
                    - SHA-256 hash generated for integrity
                    - Location: `D:\\hacthontest12314\\recordings`
                    """)
                    
                    # Connect button to processing function
                    test_btn.click(
                        fn=process,
                        inputs=[video_input, mode, threshold],
                        outputs=[video_output, report_output]
                    )
                
                # Tab 2: Security Log
                with gr.Tab("📋 Security Log"):
                    gr.Markdown("### 🔐 Incident History & Evidence Chain")
                    
                    refresh_btn = gr.Button("🔄 Refresh Log", variant="secondary")
                    log_output = gr.Textbox(
                        label="Security Log",
                        lines=30,
                        max_lines=40
                    )
                    
                    gr.Markdown(f"""
                    ### 📁 Storage Information:
                    - **Recordings Folder**: `{RECORDINGS_FOLDER}`
                    - **Security Log**: `{SECURITY_LOG_PATH}`
                    - **Hash Algorithm**: SHA-256
                    
                    All incident videos are cryptographically hashed for evidence integrity.
                    """)
                    
                    refresh_btn.click(
                        fn=view_security_log,
                        outputs=[log_output]
                    )
                
                # Tab 3: System Info
                with gr.Tab("ℹ️ System Info"):
                    gr.Markdown("""
                    # 🛡️ SafetyNet AI - System Information
                    
                    ## 📊 System Status
                    """)
                    
                    system_info = f"""
**Models Loaded**: {len(models)}
**Recordings Folder**: {RECORDINGS_FOLDER}
**Security Log**: {SECURITY_LOG_PATH}
**Hash Algorithm**: SHA-256
**Detection Modes**: SCHOOL, SHOP

## 🔧 Model Details:
"""
                    for mode_name, info in model_info.items():
                        system_info += f"\n**{mode_name} Model**:\n"
                        system_info += f"- Size: {info['size']:.1f} MB\n"
                        system_info += f"- Updated: {info['modified']}\n"
                    
                    gr.Markdown(system_info)
                    
                    gr.Markdown("""
                    ## 🔐 Security Features:
                    - ✅ Admin authentication
                    - ✅ SHA-256 evidence hashing
                    - ✅ Tamper-proof logging
                    - ✅ Chain of custody tracking
                    - ✅ Automatic evidence archival
                    
                    ## 📞 Support:
                    For technical support or questions, contact your system administrator.
                    """)
            
            gr.HTML("""
            <div style="text-align: center; padding: 20px; color: #666; font-size: 14px;">
                <p>🛡️ SafetyNet AI v2.0 | Secure Evidence Management System</p>
                <p>© 2026 SafetyNet AI | All Rights Reserved</p>
            </div>
            """)
        
        # Login handler
        def handle_login(username, password):
            success, message = authenticate_user(username, password)
            if success:
                return {
                    login_status: f"✅ {message}",
                    login_section: gr.update(visible=False),
                    main_section: gr.update(visible=True)
                }
            else:
                return {
                    login_status: f"❌ {message}",
                    login_section: gr.update(visible=True),
                    main_section: gr.update(visible=False)
                }
        
        login_btn.click(
            fn=handle_login,
            inputs=[username_input, password_input],
            outputs=[login_status, login_section, main_section]
        )
    
    return app

# ==================== LAUNCH ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🛡️ SafetyNet AI - Secure Admin Interface")
    print("="*70)
    print("\n🔐 Authentication Required")
    print("Default credentials: admin / admin123")
    print("\n🌐 Starting server on http://localhost:7860")
    print("="*70 + "\n")
    
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
