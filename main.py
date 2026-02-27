import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import cv2
import time
from datetime import datetime
import os
import winsound
from collections import deque
import statistics

import math

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ===============================
# IMPORT YOUR AI MODULES
# ===============================
from modules.face_module import FaceDetector
from modules.fatigue_module import FatigueDetector
from modules.headpose_module import HeadPoseDetector
from modules.risk_module import RiskEngine
from modules.attention_module import AttentionAnalyzer
from modules.phone_module import PhoneDetector
from modules.gaze_module import GazeEstimator
from modules.hand_module import HandDetector
from modules.trip_report import TripReport

# ===============================
# INITIALIZE AI MODULES
# ===============================
face_detector = FaceDetector()
fatigue_detector = FatigueDetector()
headpose_detector = HeadPoseDetector()
risk_engine = RiskEngine()
attention_analyzer = AttentionAnalyzer()
phone_detector = PhoneDetector(model_path="yolov8s.pt", conf_threshold=0.6)
gaze_estimator = GazeEstimator()
hand_detector = HandDetector()
trip_report = TripReport()
# ===============================
# GLOBALS
# ===============================
camera_running = False
cap = None
session_start_time = None
session_end_time = None


risk_history = deque(maxlen=300)

# ===============================
# ALERT SYSTEM
# ===============================
last_alert_time = 0
ALERT_COOLDOWN = 3

def trigger_alert(level):
    global last_alert_time
    current_time = time.time()

    if current_time - last_alert_time < ALERT_COOLDOWN:
        return

    if level == "HIGH":
        winsound.Beep(1200, 700)
    elif level == "MEDIUM":
        winsound.Beep(900, 300)

    last_alert_time = current_time

REPORT_FOLDER = "reports"
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ===============================
# FRAME PROCESSING
# ===============================
def process_frame(frame):
    global max_risk_score, high_risk_count, medium_risk_count, phone_detect_count

    import math

    h, w, _ = frame.shape
    landmarks = face_detector.detect(frame)

    face_bbox = None
    face_center = None
    face_width = None

    phone_detected = False
    phone_near_face = False
    hand_on_face = False

    risk_score = 0
    risk_level = "LOW"
    driver_state = "UNKNOWN"

    # =====================================================
    # 1️⃣ Face Detection + BBox
    # =====================================================
    if landmarks:
        xs = [int(lm.x * w) for lm in landmarks.landmark]
        ys = [int(lm.y * h) for lm in landmarks.landmark]

        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        pad_x = int((x2 - x1) * 0.6)
        pad_top = int((y2 - y1) * 0.3)
        pad_bottom = int((y2 - y1) * 0.8)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_top)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_bottom)

        face_bbox = (x1, y1, x2, y2)

        face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        face_width = x2 - x1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # =====================================================
    # 2️⃣ Phone Detection
    # =====================================================
    phone_detected, _, phone_boxes = phone_detector.detect(frame, face_bbox)

    for (px1, py1, px2, py2, conf) in phone_boxes:
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)

    # =====================================================
    # 3️⃣ Hand Detection
    # =====================================================
    hand_boxes = hand_detector.detect(frame)

    # =====================================================
    # 4️⃣ Hand قريب من الوجه؟
    # =====================================================
    if face_center and hand_boxes:

        for (hx1, hy1, hx2, hy2) in hand_boxes:
            hand_center = ((hx1 + hx2) // 2, (hy1 + hy2) // 2)

            distance = math.sqrt(
                (face_center[0] - hand_center[0])**2 +
                (face_center[1] - hand_center[1])**2
            )

            threshold = face_width * 0.6

            if distance < threshold:
                hand_on_face = True
                break

    # =====================================================
    # 5️⃣ Phone قريب من الوجه؟
    # =====================================================
    if face_center and phone_boxes:

        for (px1, py1, px2, py2, conf) in phone_boxes:
            phone_center = ((px1 + px2) // 2, (py1 + py2) // 2)

            distance_face_phone = math.sqrt(
                (face_center[0] - phone_center[0])**2 +
                (face_center[1] - phone_center[1])**2
            )

            threshold = face_width * 0.6

            if distance_face_phone < threshold:
                phone_near_face = True
                break

    # =====================================================
    # 6️⃣ Risk Evaluation
    # =====================================================
    if landmarks:

        ear, _ = fatigue_detector.check_fatigue(landmarks, w, h)
        perclos, _, _ = attention_analyzer.update(ear)
        yaw, pitch, _ = headpose_detector.get_head_pose(landmarks, w, h)
        gaze_direction = gaze_estimator.estimate(frame)

        result = risk_engine.evaluate(
            ear, yaw, pitch, perclos,
            phone_detected=phone_detected,
            gaze_direction=gaze_direction,
            hand_on_face=hand_on_face,
            phone_near_face=phone_near_face
        )
        # store the report in reports folder
        trip_report.update(result)


        driver_state = result["driver_state"]
        risk_level = result["risk_level"]
        risk_score = result["risk_score"]




        risk_history.append(risk_score)

    return frame, driver_state, risk_level, risk_score, phone_detected

# ===============================
# VIDEO LOOP
# ===============================
def video_loop(video_source=0):
    global camera_running, cap

    cap = cv2.VideoCapture(video_source)

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        frame, state, level, score, phone = process_frame(frame)

        avg_risk = int(statistics.mean(risk_history)) if risk_history else 0

        state_var.set(state)
        level_var.set(level)
        score_var.set(f"{score} (Avg:{avg_risk})")
        phone_var.set("YES" if phone else "NO")

        trigger_alert(level)

        if level == "HIGH":
            level_label.config(fg="#FF3B3B")
        elif level == "MEDIUM":
            level_label.config(fg="#FFD93B")
        else:
            level_label.config(fg="#00FF88")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    if cap:
        cap.release()


# ===============================
# GUI DESIGN
# ===============================
root = tk.Tk()
root.title("Industrial Driver Monitoring AI")
root.geometry("1200x750")
root.configure(bg="#0F172A")

# ===============================
# LEFT SIDE (VIDEO + GRAPH)
# ===============================
left_frame = tk.Frame(root, bg="#0F172A")
left_frame.pack(side="left", padx=20, pady=20)

video_label = tk.Label(left_frame, bg="#000000")
video_label.pack()

# ===== Live Graph =====
graph_frame = tk.Frame(left_frame, bg="#0F172A")
graph_frame.pack(pady=15)

fig = Figure(figsize=(7, 3), dpi=100)
ax = fig.add_subplot(111)
ax.set_ylim(0, 100)
ax.set_facecolor("#1E293B")
ax.set_title("Live Risk Trend", fontsize=11)

# Threshold Lines
ax.axhline(35)
ax.axhline(70)

line, = ax.plot([], [])

canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.get_tk_widget().pack()

def update_graph():
    if len(risk_history) > 0:
        x_data = list(range(len(risk_history)))
        y_data = list(risk_history)

        line.set_data(x_data, y_data)
        ax.set_xlim(0, max(50, len(risk_history)))
        canvas.draw_idle()

    root.after(500, update_graph)

update_graph()

# ===============================
# RIGHT PANEL
# ===============================
panel = tk.Frame(root, bg="#111827", width=350)
panel.pack(side="right", fill="y")

panel_title = tk.Label(panel, text="RISK MONITOR",
                       font=("Arial", 18, "bold"),
                       bg="#111827", fg="#00FFAA")
panel_title.pack(pady=20)

state_var = tk.StringVar(value="-")
level_var = tk.StringVar(value="-")
score_var = tk.StringVar(value="0")
phone_var = tk.StringVar(value="NO")

def create_info(label_text, var):
    frame = tk.Frame(panel, bg="#111827")
    frame.pack(pady=10)

    tk.Label(frame, text=label_text,
             font=("Arial", 12),
             bg="#111827", fg="#9CA3AF").pack()

    value = tk.Label(frame, textvariable=var,
                     font=("Arial", 16, "bold"),
                     bg="#111827", fg="#FFFFFF")
    value.pack()
    return value

state_label = create_info("Driver State", state_var)
level_label = create_info("Risk Level", level_var)
score_label = create_info("Risk Score", score_var)
phone_label = create_info("Phone Detected", phone_var)

btn_frame = tk.Frame(panel, bg="#111827")
btn_frame.pack(pady=40)

def open_camera():
    global camera_running, session_start_time
    if camera_running:
        return
    camera_running = True
    session_start_time = time.time()
    thread = threading.Thread(target=video_loop, args=(0,))
    thread.daemon = True
    thread.start()

def stop_camera():
    global camera_running, session_end_time

    if not camera_running:
        return

    camera_running = False
    session_end_time = time.time()

    # 🔥 إنهاء الرحلة
    trip_report.end_trip()

    # 🔥 اسم ملف بالوقت
    filename = os.path.join(
        REPORT_FOLDER,
        f"trip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    trip_report.save_to_file(filename)

    messagebox.showinfo("Session Ended",f"Trip JSON saved in:\n{filename}")

def exit_app():
    if camera_running:
        stop_camera()
    root.destroy()

tk.Button(btn_frame, text="Open Camera",
          width=18, height=2,
          bg="#00FFAA", fg="black",
          command=open_camera).pack(pady=5)

tk.Button(btn_frame, text="Stop Session",
          width=18, height=2,
          bg="#FF3B3B", fg="white",
          command=stop_camera).pack(pady=5)

tk.Button(btn_frame, text="Exit",
          width=18, height=2,
          bg="#374151", fg="white",
          command=exit_app).pack(pady=5)

root.mainloop()