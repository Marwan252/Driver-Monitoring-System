import numpy as np
from scipy.spatial import distance as dist


class FatigueDetector:

    def __init__(self):
        self.EAR_THRESHOLD = 0.22
        self.FRAME_THRESHOLD = 25

        self.frame_counter = 0
        self.drowsy = False

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def calculate_EAR(self, eye_points):
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])

        return (A + B) / (2.0 * C)

    def check_fatigue(self, landmarks, w, h):

        left_eye = []
        right_eye = []

        for idx in self.LEFT_EYE:
            lm = landmarks.landmark[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            left_eye.append((x, y))

        for idx in self.RIGHT_EYE:
            lm = landmarks.landmark[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            right_eye.append((x, y))

        left_ear = self.calculate_EAR(left_eye)
        right_ear = self.calculate_EAR(right_eye)

        ear = (left_ear + right_ear) / 2.0

        if ear < self.EAR_THRESHOLD:
            self.frame_counter += 1
            if self.frame_counter >= self.FRAME_THRESHOLD:
                self.drowsy = True
        else:
            self.frame_counter = 0
            self.drowsy = False

        return ear, self.drowsy