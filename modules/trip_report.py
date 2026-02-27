import time
import json


class TripReport:

    def __init__(self):
        self.start_time = time.time()
        self.end_time = None

        self.max_risk = 0

        # 🔥 Timeline storage
        self.timeline = []
        self.last_logged_second = -1

    def update(self, risk_data):

        current_time = time.time()
        elapsed = int(current_time - self.start_time)

        # نسجل مرة واحدة بس كل ثانية
        if elapsed != self.last_logged_second:
            self.last_logged_second = elapsed

            entry = {
                "second": elapsed,
                "risk_score": risk_data["risk_score"],
                "risk_level": risk_data["risk_level"],
                "driver_state": risk_data["driver_state"],

                "eye_closed_time": risk_data["eye_closed_time"],
                "distraction_time": risk_data["distraction_time"],
                "phone_time": risk_data["phone_time"],
                "hand_time": risk_data["hand_time"],

                "escalation_factor": risk_data["escalation_factor"],
                "repeat_factor": risk_data["repeat_factor"]
            }

            self.timeline.append(entry)

        # تحديث أقصى ريسك
        self.max_risk = max(self.max_risk, risk_data["risk_score"])

    def end_trip(self):
        self.end_time = time.time()

    def save_to_file(self, filename="trip_timeline.json"):

        report = {
            "trip_duration_sec": int(self.end_time - self.start_time),
            "max_risk_score": self.max_risk,
            "timeline": self.timeline
        }

        with open(filename, "w") as f:
            json.dump(report, f, indent=4)

        print("Trip timeline saved to", filename)