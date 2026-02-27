import cv2
import numpy as np

class HeadPoseDetector:

    def get_head_pose(self, landmarks, w, h):

        # 2D image points من MediaPipe
        image_points = np.array([
            (landmarks.landmark[1].x * w, landmarks.landmark[1].y * h),     # Nose tip
            (landmarks.landmark[33].x * w, landmarks.landmark[33].y * h),   # Left eye
            (landmarks.landmark[263].x * w, landmarks.landmark[263].y * h), # Right eye
            (landmarks.landmark[61].x * w, landmarks.landmark[61].y * h),   # Left mouth
            (landmarks.landmark[291].x * w, landmarks.landmark[291].y * h), # Right mouth
            (landmarks.landmark[199].x * w, landmarks.landmark[199].y * h)  # Chin
        ], dtype="double")

        # 3D model points ثابتة
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (-30.0, -30.0, -30.0),    # Left eye
            (30.0, -30.0, -30.0),     # Right eye
            (-40.0, 30.0, -30.0),     # Left mouth
            (40.0, 30.0, -30.0),      # Right mouth
            (0.0, 60.0, -50.0)        # Chin
        ])

        focal_length = w
        center = (w / 2, h / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        # تصحيح الإشارات لو محتاج
        yaw = -yaw
        return yaw, pitch, roll