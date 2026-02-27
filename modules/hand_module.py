import mediapipe as mp
import cv2

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        hand_boxes = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                landmark_style = self.mp_draw.DrawingSpec(
                    color=(0, 255, 255),
                    thickness=2,
                    circle_radius=3
                )

                connection_style = self.mp_draw.DrawingSpec(
                    color=(0, 200, 0),
                    thickness=2
                )

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=landmark_style,
                    connection_drawing_spec=connection_style
                )

                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)

                hand_boxes.append((x1, y1, x2, y2))

        return hand_boxes