import time
from collections import deque

class AttentionAnalyzer:

    def __init__(self, window_seconds=60, fps=30):

        self.window_size = window_seconds * fps
        self.eye_history = deque(maxlen=self.window_size)

        self.blink_count = 0
        self.prev_eye_closed = False

        # Adaptive EAR
        self.calibration_frames = []
        self.calibrated = False
        self.ear_threshold = 0.2
        self.calibration_duration = 10  
        self.start_time = time.time()

    def update(self, ear):

        current_time = time.time()

        # ==========================
        # 1️⃣ Calibration Phase
        # ==========================
        if not self.calibrated:
            self.calibration_frames.append(ear)

            if current_time - self.start_time > self.calibration_duration:
                avg_ear = sum(self.calibration_frames) / len(self.calibration_frames)
                self.ear_threshold = avg_ear * 0.75   # 75% من الطبيعي
                self.calibrated = True

        eye_closed = ear < self.ear_threshold

        # ==========================
        # 2️⃣ History for PERCLOS
        # ==========================
        self.eye_history.append(1 if eye_closed else 0)

        if eye_closed and not self.prev_eye_closed:
            self.blink_count += 1

        self.prev_eye_closed = eye_closed

        if len(self.eye_history) > 0:
            perclos = sum(self.eye_history) / len(self.eye_history)
        else:
            perclos = 0

        return perclos, self.blink_count, self.ear_threshold