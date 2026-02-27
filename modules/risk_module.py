import time
import config


class RiskEngine:

    def __init__(self):

        self.risk_score = 0
        self.max_score = config.MAX_RISK_SCORE
        self.last_time = time.time()

        # Angle thresholds
        self.yaw_allowed = config.YAW_ALLOWED
        self.pitch_allowed = config.PITCH_ALLOWED

        # Timers
        self.distraction_timer = 0.0
        self.gaze_timer = 0.0
        self.eye_closed_timer = 0.0
        self.phone_timer = 0.0
        self.hand_timer = 0.0   # ✅ NEW

        # Episode Counters
        self.distraction_events = 0
        self.sleep_events = 0
        self.phone_events = 0
        self.hand_events = 0    # ✅ NEW

        # Episode flags
        self.distraction_active = False
        self.sleep_active = False
        self.phone_active_flag = False
        self.hand_active_flag = False   # ✅ NEW

        # Adaptive factors
        self.repeat_factor = 1.0
        self.escalation_factor = 1.0

        # Config
        self.sleep_full_time = config.SLEEP_FULL_TIME
        self.micro_sleep_time = config.MICRO_SLEEP_TIME
        self.recovery_time = config.RECOVERY_TIME

        # Head Grace Time
        self.head_grace_time = 0.6

    # ==========================================================
    # MAIN EVALUATION FUNCTION
    # ==========================================================
    def evaluate(self, ear, yaw, pitch, perclos,
             phone_detected=False,
             gaze_direction="FORWARD",
             hand_on_face=False,
             phone_near_face=False):

        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        abs_yaw = abs(yaw)
        abs_pitch = abs(pitch)

        # ======================================================
        # 1️⃣ Dynamic Head Threshold
        # ======================================================
        if abs_yaw > 50:
            dynamic_full_time = 1.8
        elif abs_yaw > 35:
            dynamic_full_time = 2.5
        elif abs_yaw > self.yaw_allowed:
            dynamic_full_time = 3.5
        else:
            dynamic_full_time = None

        distracted_head = dynamic_full_time is not None

        # ======================================================
        # 2️⃣ Phone Tracking
        # ======================================================
        if phone_detected:
            self.phone_timer += dt
            if not self.phone_active_flag:
                self.phone_events += 1
                self.phone_active_flag = True
        else:
            self.phone_timer = max(0, self.phone_timer - dt * 2)
            self.phone_active_flag = False

        # 🔥 لو الفون قريب من الوجه نعتبره أخطر
        distracted_phone = phone_detected
        danger_phone = phone_near_face

        # ======================================================
        # 3️⃣ Hand On Face Tracking ✅ NEW
        # ======================================================
        if hand_on_face:
            self.hand_timer += dt
            if not self.hand_active_flag:
                self.hand_events += 1
                self.hand_active_flag = True
        else:
            self.hand_timer = max(0, self.hand_timer - dt * 2)
            self.hand_active_flag = False

        distracted_hand = self.hand_timer > 0.7  # لازم تستمر شوية

        # ======================================================
        # 4️⃣ Gaze Down Tracking
        # ======================================================
        if gaze_direction == "DOWN":
            self.gaze_timer += dt
        else:
            self.gaze_timer = 0.0

        distracted_gaze = self.gaze_timer > 0.9

        # ======================================================
        # 5️⃣ Unified Distraction
        # ======================================================
        distracted_now = (
            distracted_head or
            distracted_phone or
            distracted_gaze or
            distracted_hand or
            danger_phone
        )

        if distracted_now:
            self.distraction_timer += dt

            if not self.distraction_active:
                self.distraction_events += 1
                self.distraction_active = True
        else:
            self.distraction_timer = max(0, self.distraction_timer - dt * 2)
            self.distraction_active = False

        # ======================================================
        # 6️⃣ Sleep Detection
        # ======================================================
        if ear < 0.18:
            self.eye_closed_timer += dt
        else:
            self.eye_closed_timer = 0.0

        micro_sleep = self.eye_closed_timer > self.micro_sleep_time
        drowsy_state = perclos > 0.35

        if micro_sleep and not self.sleep_active:
            self.sleep_events += 1
            self.sleep_active = True
        elif not micro_sleep:
            self.sleep_active = False

        # ======================================================
        # 7️⃣ Adaptive Escalation
        # ======================================================
        if distracted_now or micro_sleep or drowsy_state:
            self.escalation_factor += dt * 0.25
            self.escalation_factor = min(self.escalation_factor, 2.0)
        else:
            self.escalation_factor = max(1.0, self.escalation_factor - dt)

        # ======================================================
        # 8️⃣ Risk Update Logic
        # ======================================================

        if micro_sleep:
            rate = self.max_score / 1.8
            self.risk_score += rate * dt * self.escalation_factor

        elif drowsy_state:
            rate = self.max_score / self.sleep_full_time
            self.risk_score += rate * dt * self.escalation_factor

        # 🔥 أخطر حالة بعد النوم
        elif danger_phone:
            rate = self.max_score / 1.5
            self.risk_score += rate * dt * self.escalation_factor

        elif distracted_phone:
            rate = self.max_score / 3.0
            self.risk_score += rate * dt * self.escalation_factor

        elif distracted_hand:
            rate = self.max_score / 4.0
            self.risk_score += rate * dt * self.escalation_factor

        elif distracted_gaze:
            rate = self.max_score / 4.5
            self.risk_score += rate * dt * self.escalation_factor

        elif distracted_head:

            if self.distraction_timer > self.head_grace_time:

                effective_time = self.distraction_timer - self.head_grace_time
                intensity = min(effective_time / dynamic_full_time, 1.0)

                self.repeat_factor += dt * 0.2
                self.repeat_factor = min(self.repeat_factor, 1.8)

                self.risk_score += (
                    intensity * 30 *
                    self.repeat_factor *
                    self.escalation_factor *
                    dt
                )

        else:
            decay_rate = self.max_score / self.recovery_time
            self.risk_score -= decay_rate * dt
            self.repeat_factor = max(1.0, self.repeat_factor - dt * 1.2)

        # ======================================================
        # Clamp
        # ======================================================
        self.risk_score = max(0, min(self.max_score, self.risk_score))

        # ======================================================
        # Driver State
        # ======================================================
        if micro_sleep:
            driver_state = "MICRO_SLEEP"
        elif drowsy_state:
            driver_state = "DROWSY"
        elif danger_phone:
            driver_state = "PHONE_NEAR_FACE"
        elif distracted_phone:
            driver_state = "PHONE_DISTRACTION"
        elif distracted_hand:
            driver_state = "HAND_ON_FACE"
        elif distracted_gaze:
            driver_state = "GAZE_DOWN"
        elif distracted_head:
            driver_state = "HEAD_DISTRACTION"
        else:
            driver_state = "SAFE"

        # ======================================================
        # Risk Level
        # ======================================================
        if self.risk_score >= 70:
            risk_level = "HIGH"
        elif self.risk_score >= 35:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "driver_state": driver_state,
            "risk_level": risk_level,
            "risk_score": int(self.risk_score),

            "distraction_time": round(self.distraction_timer, 2),
            "eye_closed_time": round(self.eye_closed_timer, 2),
            "gaze_down_time": round(self.gaze_timer, 2),
            "phone_time": round(self.phone_timer, 2),
            "hand_time": round(self.hand_timer, 2),

            "repeat_factor": round(self.repeat_factor, 2),
            "escalation_factor": round(self.escalation_factor, 2),

            "distraction_events": self.distraction_events,
            "sleep_events": self.sleep_events,
            "phone_events": self.phone_events,
            "hand_events": self.hand_events
        }