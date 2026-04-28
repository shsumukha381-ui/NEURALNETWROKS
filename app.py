
import gradio as gr
import torch
import torch.nn as nn
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
import json
from optical_flow_detector import OpticalFlowDetector
from evidence_hasher import EvidenceHasher

# Try to import Gemini summarizer (optional)
try:
    from gemini_summarizer import GeminiSummarizer
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    GeminiSummarizer = None

# ── Recordings folder ────────────────────────────────────────────────────────
RECORDINGS_FOLDER = r'C:\Users\lenovo\OneDrive\Desktop\hacathonbang\recordings'
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)

print("=" * 70)
print("SafetyNet AI - Multi-Class Testing Interface")
print("=" * 70)

# ── Evidence / Gemini ─────────────────────────────────────────────────────────
evidence_hasher = EvidenceHasher()
print("✓ Evidence hasher initialized")

gemini_summarizer = None
if GEMINI_AVAILABLE:
    try:
        gemini_summarizer = GeminiSummarizer()
        if gemini_summarizer.enabled:
            print("✓ Gemini AI summarizer initialized")
        else:
            print("⚠️  Gemini AI summarizer not available (API key not configured)")
    except Exception as e:
        print(f"⚠️  Gemini init error: {e}")
else:
    print("⚠️  Gemini AI not installed")

# ── Model architecture (must match train_multiclass_images.py) ────────────────
class MultiClassModel(nn.Module):
    def __init__(self, input_size=120, hidden_size=128, num_layers=3, num_classes=8):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.3, bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn, dim=1)
        return self.classifier(context)


# ── Feature engineering (must match training) ─────────────────────────────────
def engineer_features(sequence):
    enhanced = []
    for i, frame in enumerate(sequence):
        features = list(frame)
        if i > 0:
            v_lw = np.linalg.norm(frame[45:47] - sequence[i - 1][45:47])
            v_rw = np.linalg.norm(frame[48:50] - sequence[i - 1][48:50])
            v_le = np.linalg.norm(frame[39:41] - sequence[i - 1][39:41])
            v_re = np.linalg.norm(frame[42:44] - sequence[i - 1][42:44])
        else:
            v_lw = v_rw = v_le = v_re = 0
        shoulder_width = np.linalg.norm(frame[33:35] - frame[36:38])
        hip_width      = np.linalg.norm(frame[69:71] - frame[72:74])
        arm_elev_l     = frame[34] - frame[46]
        arm_elev_r     = frame[37] - frame[49]
        motion         = np.sum(np.abs(frame - sequence[i - 1])) if i > 0 else 0
        all_f = features + [v_lw, v_rw, v_le, v_re, 0, 0, 0, 0,
                            shoulder_width, hip_width, 0,
                            arm_elev_l, arm_elev_r, motion] + [0] * 7
        enhanced.append(all_f[:120])
    return np.array(enhanced)


# ── Load model ────────────────────────────────────────────────────────────────
WEIGHTS_PATH = 'multiclass_weights.pth'
INFO_PATH    = 'multiclass_training_info.json'

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model      = None
class_names = []
model_info  = {}

print(f"\nDevice: {device}")
print("Loading multi-class model...")

if os.path.exists(INFO_PATH):
    with open(INFO_PATH, 'r') as f:
        info = json.load(f)
    class_names = info.get('class_names', [])
    num_classes = info.get('num_classes', len(class_names))
    model_info  = {
        'num_classes':   num_classes,
        'class_names':   class_names,
        'best_val_acc':  info.get('best_val_acc', 0),
        'training_date': info.get('training_date', 'unknown'),
        'size_mb':       round(os.path.getsize(WEIGHTS_PATH) / (1024 * 1024), 1)
                         if os.path.exists(WEIGHTS_PATH) else 0
    }

    if os.path.exists(WEIGHTS_PATH):
        model = MultiClassModel(120, 128, 3, num_classes).to(device)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        model.eval()
        print(f"✓ Model loaded: {num_classes} classes — {model_info['size_mb']} MB")
        print(f"  Classes: {', '.join(class_names)}")
        print(f"  Best val accuracy: {model_info['best_val_acc']:.2f}%")
    else:
        print(f"❌ Weights not found: {WEIGHTS_PATH}")
        print("   Run START_IMAGE_TRAINING.bat first")
else:
    print(f"❌ Training info not found: {INFO_PATH}")
    print("   Run START_IMAGE_TRAINING.bat first")

print("=" * 70 + "\n")

# ── Class colour map ──────────────────────────────────────────────────────────
# BGR colours for each class label
CLASS_COLORS = {
    'NormalVideos': (0, 200, 0),
    'Fighting':     (0, 0, 255),
    'Assault':      (0, 0, 220),
    'Abuse':        (0, 60, 255),
    'Arrest':       (255, 140, 0),
    'Arson':        (0, 100, 255),
    'Burglary':     (180, 0, 255),
    'Explosion':    (0, 0, 180),
}

NORMAL_CLASS = 'NormalVideos'

def get_class_color(class_name):
    return CLASS_COLORS.get(class_name, (0, 165, 255))

def is_anomaly(class_name):
    return class_name != NORMAL_CLASS


# ── Main processing function ──────────────────────────────────────────────────
def process(video, threshold, progress=gr.Progress()):
    if not video:
        return None, "ERROR: Please upload a video file.", "No video provided."

    if model is None:
        return None, (
            "ERROR: Model not loaded.\n"
            "Run START_IMAGE_TRAINING.bat to train the model first."
        ), ""

    threshold_decimal = threshold / 100.0

    progress(0, desc="Initializing...")
    print(f"\n{'=' * 70}")
    print(f"Processing video — threshold: {threshold}%")
    print(f"Classes: {', '.join(class_names)}")
    print(f"{'=' * 70}")

    try:
        # ── Open video ────────────────────────────────────────────────────────
        progress(0.05, desc="Opening video...")
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            return None, "ERROR: Cannot open video file.", ""

        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {w}x{h} @ {fps}fps — {total} frames ({total/fps:.1f}s)")

        # ── Output writer ─────────────────────────────────────────────────────
        progress(0.1, desc="Creating output...")
        out_path = tempfile.mktemp(suffix='.mp4')
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # ── MediaPipe + Optical Flow ──────────────────────────────────────────
        progress(0.15, desc="Initializing detectors...")
        pose     = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5)
        mp_draw  = mp.solutions.drawing_utils
        opt_flow = OpticalFlowDetector(motion_threshold=2.0, high_motion_threshold=5.0)

        # ── State ─────────────────────────────────────────────────────────────
        mem            = deque(maxlen=30)
        frame_results  = []   # list of (class_name, confidence) per scored frame
        anomaly_counts = {c: 0 for c in class_names}
        n              = 0

        last_beep_time = datetime.now()
        beep_cooldown  = 2.0

        # ── Frame loop ────────────────────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            n += 1
            progress(0.15 + (n / max(total, 1)) * 0.75,
                     desc=f"Frame {n}/{total}")

            predicted_class = NORMAL_CLASS
            confidence      = 0.0
            all_probs       = None
            motion_score    = 0.0

            # Optical flow
            flow, motion_magnitude = opt_flow.calculate_optical_flow(frame)

            # Pose extraction
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                coords = []
                for lm in results.pose_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                coords = np.array(coords, dtype=np.float32)
                mem.append(coords)

                if len(mem) == 30:
                    seq    = np.array(list(mem))
                    seq120 = engineer_features(seq)
                    tensor = torch.FloatTensor(seq120[np.newaxis]).to(device)

                    with torch.no_grad():
                        probs      = model(tensor)[0].cpu().numpy()   # (num_classes,)
                        pred_idx   = int(np.argmax(probs))
                        confidence = float(probs[pred_idx])
                        predicted_class = class_names[pred_idx]
                        all_probs  = probs

                    frame_results.append((predicted_class, confidence))

                    if is_anomaly(predicted_class) and confidence >= threshold_decimal:
                        anomaly_counts[predicted_class] += 1

                        # Beep alert
                        now = datetime.now()
                        if (now - last_beep_time).total_seconds() >= beep_cooldown:
                            print(f"🔊 ALERT! {predicted_class} — {confidence:.1%}")
                            def _beep():
                                try:
                                    winsound.Beep(1500, 200)
                                    winsound.Beep(1000, 300)
                                    winsound.Beep(1500, 200)
                                except Exception:
                                    pass
                            threading.Thread(target=_beep, daemon=True).start()
                            last_beep_time = now

                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            # Optical flow overlay
            if motion_magnitude is not None:
                skel_count = 1 if results.pose_landmarks else 0
                of_suspicious, _, motion_score = opt_flow.detect_suspicious_activity(
                    motion_magnitude, skel_count, frame.shape
                )
                heatmap, _ = opt_flow.create_motion_heatmap(motion_magnitude, frame.shape)
                if heatmap is not None:
                    frame = cv2.addWeighted(frame, 0.8, heatmap, 0.2, 0)

            # ── Draw UI ───────────────────────────────────────────────────────
            color = get_class_color(predicted_class)
            alert = is_anomaly(predicted_class) and confidence >= threshold_decimal

            # Top banner
            banner_h = 40 + len(class_names) * 22 + 20
            cv2.rectangle(frame, (0, 0), (w, banner_h), (0, 0, 0), -1)

            cv2.putText(frame, "SafetyNet AI - Multi-Class Detection",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Detected class + confidence
            det_text = f"DETECTED: {predicted_class.upper()}  {confidence:.0%}"
            cv2.putText(frame, det_text,
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            # Per-class probability bars
            if all_probs is not None:
                y = 75
                for i, cname in enumerate(class_names):
                    p     = float(all_probs[i])
                    bar_w = int(p * 160)
                    bcol  = get_class_color(cname)
                    cv2.rectangle(frame, (10, y - 12), (10 + bar_w, y), bcol, -1)
                    cv2.putText(frame, f"{cname}: {p:.0%}",
                                (180, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                (220, 220, 220), 1)
                    y += 22

            # Motion score bottom-left
            cv2.putText(frame, f"Motion: {motion_score:.2f}  Frame: {n}/{total}",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            # Threshold indicator top-right
            cv2.putText(frame, f"Threshold: {threshold}%",
                        (w - 190, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Red border on alert
            if alert:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)

            out.write(frame)

            if n % 100 == 0:
                top = sorted(anomaly_counts.items(), key=lambda x: -x[1])
                print(f"  Frame {n}/{total} — top: {top[:3]}")

        # ── Wrap up ───────────────────────────────────────────────────────────
        progress(0.9, desc="Finalizing...")
        cap.release()
        out.release()
        pose.close()

        # Stats
        total_anomaly_frames = sum(anomaly_counts.values())
        scored_frames        = len(frame_results)
        normal_frames        = sum(1 for c, _ in frame_results if c == NORMAL_CLASS)

        # Dominant anomaly class
        top_class      = max(anomaly_counts, key=anomaly_counts.get) \
                         if total_anomaly_frames > 0 else NORMAL_CLASS
        top_class_conf = np.mean([conf for cls, conf in frame_results
                                  if cls == top_class]) if total_anomaly_frames > 0 else 0.0

        # ── Save + hash if anomalies found ────────────────────────────────────
        permanent_path = out_path
        ai_summary     = "ℹ️ No anomalies detected."

        if total_anomaly_frames > 0:
            progress(0.92, desc="Saving evidence...")
            try:
                ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
                perm_name = f"incident_{top_class}_{ts}.mp4"
                permanent_path = os.path.join(RECORDINGS_FOLDER, perm_name)
                shutil.move(out_path, permanent_path)
                print(f"✓ Saved: {permanent_path}")

                incident_info = {
                    "anomaly_type":        top_class,
                    "detection_threshold": threshold,
                    "total_frames":        total,
                    "scored_frames":       scored_frames,
                    "anomaly_frames":      total_anomaly_frames,
                    "normal_frames":       normal_frames,
                    "class_breakdown":     anomaly_counts,
                    "video_fps":           fps,
                    "video_resolution":    f"{w}x{h}",
                    "duration_seconds":    round(total / fps, 2),
                }
                evidence_hasher.hash_evidence_async(permanent_path, incident_info)

                # Gemini summary
                if gemini_summarizer and gemini_summarizer.enabled:
                    progress(0.97, desc="Generating AI summary...")
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as ex:
                        fut = ex.submit(
                            gemini_summarizer.summarize_incident,
                            video_path=permanent_path,
                            mode="MULTICLASS",
                            anomaly_type=top_class
                        )
                        try:
                            ai_summary = fut.result(timeout=60)
                        except concurrent.futures.TimeoutError:
                            ai_summary = "⚠️ AI summary timed out."
                        except Exception as e:
                            ai_summary = f"⚠️ AI summary error: {e}"
                elif not GEMINI_AVAILABLE:
                    ai_summary = "⚠️ Gemini not installed."
                else:
                    ai_summary = "⚠️ Gemini API key not configured."

            except Exception as e:
                print(f"❌ Save/hash error: {e}")
                permanent_path = out_path

        # ── Build report ──────────────────────────────────────────────────────
        progress(1.0, desc="Done!")

        pct = lambda x: f"{100 * x / scored_frames:.1f}%" if scored_frames else "0%"

        report = f"""{'=' * 60}
SAFETYNET AI — MULTI-CLASS RESULTS
{'=' * 60}

MODEL
  Weights : {WEIGHTS_PATH}
  Classes : {num_classes}  ({', '.join(class_names)})
  Val Acc : {model_info.get('best_val_acc', 0):.2f}%
  Trained : {model_info.get('training_date', 'unknown')}
  Size    : {model_info.get('size_mb', 0)} MB

VIDEO
  Resolution : {w}x{h}
  FPS        : {fps}
  Frames     : {total}  ({total/fps:.1f}s)
  Threshold  : {threshold}%

FRAME BREAKDOWN
  Scored frames  : {scored_frames}
  Normal frames  : {normal_frames}  ({pct(normal_frames)})
  Anomaly frames : {total_anomaly_frames}  ({pct(total_anomaly_frames)})

CLASS DETECTIONS  (confidence ≥ {threshold}%)
"""
        for cname in class_names:
            cnt = anomaly_counts[cname]
            if cnt > 0:
                report += f"  {cname:<18}: {cnt} frames  ({pct(cnt)})\n"

        report += f"""
TOP ANOMALY
  Class      : {top_class}
  Avg conf   : {top_class_conf:.1%}

"""
        if total_anomaly_frames > 0:
            report += f"[ALERT] {top_class} detected — {total_anomaly_frames} frames flagged.\n"
            if permanent_path and os.path.exists(permanent_path):
                report += f"Video saved: {os.path.abspath(permanent_path)}\n"
        else:
            report += "[OK] No anomalies detected above threshold.\n"

        report += f"\n{'=' * 60}\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 60}"

        print(report)
        return permanent_path, report, ai_summary

    except Exception as e:
        import traceback
        err = f"ERROR: {e}\n\n{traceback.format_exc()}"
        print(err)
        return None, err, "❌ Error during processing."


# ── Gradio UI ─────────────────────────────────────────────────────────────────
model_status_md = ""
if model is None:
    model_status_md = "⚠️ **Model not loaded** — run `START_IMAGE_TRAINING.bat` first."
else:
    model_status_md = (
        f"✅ **Model ready** — {model_info['num_classes']} classes · "
        f"{model_info['size_mb']} MB · "
        f"Val acc {model_info['best_val_acc']:.1f}% · "
        f"Trained {model_info['training_date']}\n\n"
        f"**Classes:** {', '.join(class_names)}"
    )

with gr.Blocks(title="SafetyNet AI") as demo:
    gr.Markdown("# SafetyNet AI — Multi-Class Anomaly Detection")
    gr.Markdown(model_status_md)

    with gr.Row():
        # ── Left column ───────────────────────────────────────────────────────
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Test Video", sources=["upload"])

            threshold = gr.Slider(
                minimum=50, maximum=95, value=70, step=5,
                label="Detection Threshold (%)",
                info="Minimum confidence to flag a frame as anomaly"
            )

            test_btn = gr.Button("▶  Analyse Video", variant="primary", size="lg")

            gr.Markdown("""
**Threshold guide**
- **70%** — balanced (recommended)
- **60%** — more sensitive, more alerts
- **85%** — strict, only high-confidence events

**Classes detected**
- NormalVideos · Fighting · Assault · Abuse
- Arrest · Arson · Burglary · Explosion
""")

        # ── Right column ──────────────────────────────────────────────────────
        with gr.Column(scale=2):
            video_output = gr.Video(label="Annotated Output")

            report_output = gr.Textbox(
                label="Detection Report",
                lines=22,
                max_lines=35
            )

            ai_summary_output = gr.Textbox(
                label="🤖 AI Narrative Summary (Gemini)",
                lines=10,
                max_lines=20,
                placeholder="AI-generated incident summary will appear here..."
            )

    test_btn.click(
        fn=process,
        inputs=[video_input, threshold],
        outputs=[video_output, report_output, ai_summary_output]
    )

print("Starting interface on http://localhost:7860")
print("=" * 70 + "\n")
demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
