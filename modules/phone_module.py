from ultralytics import YOLO
import torch


class PhoneDetector:

    def __init__(self, model_path="yolov8s.pt", conf_threshold=0.4):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.conf_threshold = conf_threshold
        self.phone_class_id = 67  # COCO class id for 'cell phone'

        # Temporal smoothing
        self.phone_frame_count = 0
        self.phone_required_frames = 3

        self.last_boxes = []
        self.last_confidence = 0.0

    # =====================================================
    # ROI-BASED DETECTION
    # =====================================================
    def detect(self, frame, face_bbox=None):

        h, w, _ = frame.shape

        # ==========================
        # If face detected → use ROI
        # ==========================
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox

            # Expand ROI (include chest area)
            pad_x = int((fx2 - fx1) * 0.5)
            pad_y_top = int((fy2 - fy1) * 0.3)
            pad_y_bottom = int((fy2 - fy1) * 1.2)

            x1 = max(0, fx1 - pad_x)
            y1 = max(0, fy1 - pad_y_top)
            x2 = min(w, fx2 + pad_x)
            y2 = min(h, fy2 + pad_y_bottom)

            roi = frame[y1:y2, x1:x2]
        else:
            # fallback to full frame
            x1, y1 = 0, 0
            roi = frame

        # ==========================
        # Run YOLO on ROI
        # ==========================
        results = self.model.predict(
            source=roi,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            imgsz=640
        )

        phone_detected = False
        confidence = 0.0
        boxes = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id == self.phone_class_id:
                    phone_detected = True
                    confidence = max(confidence, conf)

                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])

                    # Convert ROI coords → original frame coords
                    global_x1 = bx1 + x1
                    global_y1 = by1 + y1
                    global_x2 = bx2 + x1
                    global_y2 = by2 + y1

                    boxes.append((global_x1, global_y1,
                                  global_x2, global_y2, conf))

        # ==========================
        # Temporal smoothing
        # ==========================
        if phone_detected:
            self.phone_frame_count += 1
            self.last_boxes = boxes
            self.last_confidence = confidence
        else:
            self.phone_frame_count = 0

        stable_phone_detected = self.phone_frame_count >= self.phone_required_frames

        if stable_phone_detected:
            return True, self.last_confidence, self.last_boxes
        else:
            return False, 0.0, []