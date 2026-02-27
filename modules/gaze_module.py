import cv2
import mediapipe as mp
import numpy as np


class GazeEstimator:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1
        )

        # Iris landmark indices
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

    def estimate(self, frame):

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return "UNKNOWN"

        mesh = results.multi_face_landmarks[0]

        # Get left iris center
        left_iris_points = []
        for idx in self.LEFT_IRIS:
            x = int(mesh.landmark[idx].x * w)
            y = int(mesh.landmark[idx].y * h)
            left_iris_points.append((x, y))

        left_iris_center = np.mean(left_iris_points, axis=0)

        # Eye corners (approx)
        left_eye_outer = mesh.landmark[33]
        left_eye_inner = mesh.landmark[133]

        outer_x = int(left_eye_outer.x * w)
        inner_x = int(left_eye_inner.x * w)

        iris_x = left_iris_center[0]

        # Normalize iris position
        ratio = (iris_x - outer_x) / (inner_x - outer_x + 1e-6)

        # Direction classification
        if ratio < 0.35:
            return "RIGHT"
        elif ratio > 0.65:
            return "LEFT"
        else:
            # Check vertical
            iris_y = left_iris_center[1]
            eye_top = int(mesh.landmark[159].y * h)
            eye_bottom = int(mesh.landmark[145].y * h)

            vertical_ratio = (iris_y - eye_top) / (eye_bottom - eye_top + 1e-6)

            if vertical_ratio > 0.65:
                return "DOWN"
            else:
                return "FORWARD"