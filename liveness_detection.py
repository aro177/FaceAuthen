import cv2
from PIL import Image
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import dlib
from scipy.spatial import distance as dist

class LivenessDetector:
    def __init__(self, yolo_model='yolo-face.pt', output_dir='output', session_id='random_session', predictor_path="shape_predictor_68_face_landmarks.dat"):
        self.yolo = YOLO(yolo_model)
        self.output_dir = Path(output_dir) / session_id
        self.images_dir = self.output_dir / 'tracked_frames'
        self.json_dir = self.output_dir / 'tracking_jsons'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Dlib model not found at {predictor_path}")
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Thresholds
        self.EAR_THRESHOLD = 0.25
        self.EAR_BLINK_THRESHOLD = 0.21
        self.SMILE_MAR_THRESHOLD = 0.35
        self.SMILE_CORNER_THRESHOLD = 0.03
        
        # State tracking
        self.reset()
        self.session_id = session_id
        self.saved_track_log = []
        
        # Landmarks indices
        self.LEFT_EYE_START = 36
        self.LEFT_EYE_END = 42
        self.RIGHT_EYE_START = 42
        self.RIGHT_EYE_END = 48

    def reset(self):
        """Reset state"""
        self.smile_detected = False
        self.blink_detected = False
        self.ear_history = []
        self.eye_states = []
        self.smile_frames = 0
        self.neutral_frames = 0
        self.total_blinks = 0
        self.frame_count = 0
        self.smile_confidence = 0.0
        self.blink_confidence = 0.0
        self.consecutive_closed = 0
        self.was_open = True
        self.baseline_ear = None
        self.baseline_mar = None
        self.calibration_frames = []
        self.is_calibrated = False

    def _get_landmarks(self, gray, bbox):
        """Extract 68 landmarks từ YOLO bbox"""
        x1, y1, x2, y2 = map(int, bbox)
        dlib_rect = dlib.rectangle(x1, y1, x2, y2)
        shape = self.predictor(gray, dlib_rect)
        landmarks = []
        for i in range(68):
            landmarks.append((shape.part(i).x, shape.part(i).y))
        return np.array(landmarks)

    def _eye_aspect_ratio(self, eye):
        """EAR calculation"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C) if C > 0 else 0
        return ear

    def detect_smile(self, landmarks):
        # Get mouth landmarks (48-67)
        left_corner = np.array(landmarks[48])
        right_corner = np.array(landmarks[54])
        upper_lip_top = np.array(landmarks[51])
        upper_lip_bottom = np.array(landmarks[62])
        lower_lip_top = np.array(landmarks[66])
        lower_lip_bottom = np.array(landmarks[57])

        # Calculate mouth dimensions
        mouth_width = dist.euclidean(left_corner, right_corner)
        mouth_opening = dist.euclidean(upper_lip_bottom, lower_lip_top)
        mar = mouth_opening / mouth_width if mouth_width > 0 else 0

        # Mouth corner elevation
        mouth_center_y = (upper_lip_top[1] + lower_lip_bottom[1]) / 2
        left_corner_elevation = mouth_center_y - left_corner[1]
        right_corner_elevation = mouth_center_y - right_corner[1]
        avg_corner_elevation = (left_corner_elevation + right_corner_elevation) / 2
        normalized_elevation = avg_corner_elevation / mouth_width if mouth_width > 0 else 0

        # Calibration
        if not self.is_calibrated and self.frame_count <= 15:
            self.calibration_frames.append({
                'mar': mar, 'elevation': normalized_elevation, 'width': mouth_width
            })
            if len(self.calibration_frames) >= 10:
                self.baseline_mar = np.mean([f['mar'] for f in self.calibration_frames])
                self.baseline_elevation = np.mean([f['elevation'] for f in self.calibration_frames])
                self.baseline_width = np.mean([f['width'] for f in self.calibration_frames])
                self.is_calibrated = True

        # Smile detection criteria
        is_smiling = False
        smile_score = 0

        if self.is_calibrated:
            mar_increase = mar - self.baseline_mar
            elevation_increase = normalized_elevation - self.baseline_elevation
            width_increase = (mouth_width - self.baseline_width) / self.baseline_width if self.baseline_width > 0 else 0

            if mar_increase > 0.05: smile_score += 1
            if elevation_increase > 0.02: smile_score += 1
            if width_increase > 0.05: smile_score += 1
            if mar > 0.12: smile_score += 1

            is_smiling = smile_score >= 2
            message = f"MAR: {mar:.3f} | Elev: {normalized_elevation:.3f} | Score: {smile_score}/4"
        else:
            is_smiling = mar > 0.15 and normalized_elevation > 0.02
            message = f"MAR: {mar:.3f} | Elev: {normalized_elevation:.3f} | Calibrating..."

        # Track smile frames
        if is_smiling:
            self.smile_frames += 1
            self.neutral_frames = 0
        else:
            self.neutral_frames += 1
            if self.neutral_frames > 10:
                self.smile_frames = max(0, self.smile_frames - 1)

        confidence = min(1.0, smile_score / 3) if self.is_calibrated else 0.3
        if self.smile_frames >= 8:
            self.smile_detected = True
            self.smile_confidence = confidence

        return is_smiling, confidence, message

    def detect_blink(self, landmarks):
        left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
        right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]

        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        self.ear_history.append(ear)
        if len(self.ear_history) > 50:
            self.ear_history.pop(0)

        # Adaptive threshold
        if len(self.ear_history) >= 15:
            sorted_ears = sorted(self.ear_history)
            baseline_ear = sorted_ears[int(len(sorted_ears) * 0.75)]
            adaptive_threshold = baseline_ear * 0.70
            blink_threshold = max(adaptive_threshold, self.EAR_BLINK_THRESHOLD)
        else:
            blink_threshold = self.EAR_THRESHOLD
            baseline_ear = 0.30

        is_closed = ear < blink_threshold
        blink_detected_now = False

        if is_closed:
            self.consecutive_closed += 1
            if self.was_open and self.consecutive_closed >= 2:
                self.was_open = False
        else:
            if not self.was_open and self.consecutive_closed >= 2:
                self.total_blinks += 1
                self.blink_detected = True
                self.blink_confidence = 1.0
                blink_detected_now = True
            self.was_open = True
            self.consecutive_closed = 0

        if is_closed:
            current_state = "NHAM MAT" if self.consecutive_closed >= 3 else "DANG NHAM"
        else:
            current_state = "MO MAT"

        confidence = 1.0 if self.blink_detected else 0.0
        message = f"EAR: {ear:.3f} | Thresh: {blink_threshold:.3f} | {current_state} | Blinks: {self.total_blinks}"
        return blink_detected_now, confidence, message

    def _process_landmarks_result(self, frame, bbox):
        """Process landmarks"""
        self.frame_count += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = self._get_landmarks(gray, bbox)
        
        # Detect smile & blink
        smile_now, smile_conf, smile_msg = self.detect_smile(landmarks)
        blink_now, blink_conf, blink_msg = self.detect_blink(landmarks)
        
        return smile_msg, blink_msg

    def analyze_video(self, video_path):
        """Main function - tracking với vid_stride=30"""
        results = self.yolo.track(
            source=video_path,
            stream=True,  # Memory efficient cho video dài
            vid_stride=30,
            persist=True,
            save=False,   # Không save video, tự handle frames
            project=self.output_dir,
            name='tracking_results',
            imgsz=640,
            conf=0.5,
            save_txt=False  # Không dùng save_txt mặc định
        )

        # Danh sách lưu tracking data
        all_tracks = []

        frame_idx = 0
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                # Save frame as image
                frame_path = self.images_dir / f'frame_{frame_idx:06d}.jpg'
                original_frame = result.orig_img  # Lấy frame gốc từ result
                Image.fromarray(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)).save(frame_path)
        
                # Extract tracking data cho frame này
                frame_tracks = []
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]
                track_ids = result.boxes.id.int().cpu().numpy() if result.boxes.id is not None else []
                classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []
        
                for i, (bbox, tid) in enumerate(zip(boxes, track_ids)):
                    track_data = {
                        "trackingId": int(tid),
                        "framePath": str(frame_path.relative_to(self.output_dir)),
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],  # x1,y1,x2,y2
                        "class": int(classes[i]) if i < len(classes) else None,
                        "conf": float(result.boxes.conf[i]) if i < len(result.boxes.conf) else None
                    }
                    smile_msg, blink_msg = self._process_landmarks_result(original_frame, bbox) #For debug
                    frame_tracks.append(track_data)
        
                # Save JSON per frame
                frame_json_path = self.json_dir / f'frame_{frame_idx:06d}.json'
                with open(frame_json_path, 'w') as f:
                    json.dump(frame_tracks, f, indent=2)
        
                all_tracks.extend(frame_tracks)

                
                
            frame_idx += 1  # Tăng theo stride thực tế

            result = {
                "message": "",
                "is_live": self.smile_detected and self.blink_detected,
                "smile": self.smile_detected,
                "blink": self.blink_detected
            }
        
            # Build message
            if result["is_live"]:
                result["message"] = "XÁC THỰC THÀNH CÔNG!"

        return result