import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import os, glob
import cv2
import numpy as np
from numpy.linalg import norm
import pickle
import grpc
from datetime import datetime
import uuid
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Any
from ultralytics import YOLO
from pathlib import Path


class TFServingClient:
    def __init__(self, host: str = "127.0.0.1:8500", model_name: str = "facenet", input_name: str = "input"):
        self.channel = grpc.insecure_channel(host)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.model_name = model_name
        self.input_name = input_name
        print("Here 3")

    def predict(self, input_data: np.ndarray, input_name: str = None):
        input_name = input_name or self.input_name
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = "serving_default"
        request.inputs[input_name].CopyFrom(tf.make_tensor_proto(input_data, shape=input_data.shape))
        response = self.stub.Predict(request, timeout=10.0)
        return response

    def get_embedding(self, input_data: np.ndarray, input_name: str = None) -> np.ndarray:
        input_name = input_name or self.input_name
        response = self.predict(input_data, input_name)
        embedding = tf.make_ndarray(response.outputs[list(response.outputs.keys())[0]])
        return embedding[0]


class FaceRecognitionSystem:
    """
    - Face capture: detect + crop largest face using YOLO only
    - Face embedding: FaceNet TF SavedModel via TF Serving
    - Recognition: cosine distance with a simple mean-embedding database
    """

    def __init__(self, tfserving_host: str = "127.0.0.1:8500", model_name: str = "facenet", input_name: str = "input", img_size: int = 160, thresh: float = 0.3, yolo_model_path="yolo-face.pt"):
        self.img_size = img_size
        self.thresh = thresh
        self.input_name = input_name
        self.tf_client = TFServingClient(host=tfserving_host, model_name=model_name, input_name=input_name)
        print("Here 2")
        self.yolo_model_path = yolo_model_path
        self.yolo = None
        self.database: dict[str, np.ndarray] = {}
        print("Here 4")

    def _get_yolo(self):
        if self.yolo is None:
            self.yolo = YOLO(self.yolo_model_path)
            print("Here 5")
        return self.yolo

    @staticmethod
    def l2_normalize(x: np.ndarray) -> np.ndarray:
        n = norm(x)
        return x / (n if n > 0 else 1e-12)

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b))

    def preprocess(self, rgb_face: np.ndarray) -> np.ndarray:
        face = cv2.resize(rgb_face, (self.img_size, self.img_size))
        face = face.astype("float32")
        mean, std = face.mean(), face.std()
        face = (face - mean) / (std if std > 0 else 1e-6)
        return face

    def detect_face_largest_xywh(self, bgr_img: np.ndarray) -> tuple[int, int, int, int] | None:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        yolo = self._get_yolo()
        results = yolo(rgb, imgsz=640, conf=0.5)
        if not results[0].boxes or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return None
        areas = []
        for (x1, y1, x2, y2) in boxes:
            w, h = x2 - x1, y2 - y1
            areas.append(w * h)
        idx = np.argmax(areas)
        x1, y1, x2, y2 = boxes[idx]
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        return (x, y, w, h)

    def crop_face(self, bgr_img: np.ndarray, xywh: tuple[int, int, int, int]) -> np.ndarray | None:
        x, y, w, h = xywh
        H, W = bgr_img.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        if x2 <= x or y2 <= y:
            return None
        face_bgr = bgr_img[y:y2, x:x2]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return face_rgb

    def get_embedding(self, rgb_face: np.ndarray) -> np.ndarray:
        face = self.preprocess(rgb_face)
        face_batch = np.expand_dims(face, 0).astype(np.float32)
        emb = self.tf_client.get_embedding(face_batch)
        return self.l2_normalize(emb)

    def add_identity_from_images(self, user_id: str, img_paths: list[str]) -> bool:
        embs = []
        for p in img_paths:
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            xywh = self.detect_face_largest_xywh(bgr)
            if xywh is None:
                continue
            face_rgb = self.crop_face(bgr, xywh)
            if face_rgb is None:
                continue
            emb = self.get_embedding(face_rgb)
            embs.append(emb)
        if not embs:
            return False
        mean_emb = self.l2_normalize(np.mean(embs, axis=0))
        self.database[user_id] = mean_emb
        return True

    def _calculate_embedding_quality(self, user_id: str) -> float:
        if user_id not in self.database:
            return 0.0
        return 1.0

    def _find_best_match(self, emb: np.ndarray) -> tuple[str, float]:
        if not self.database:
            return "Unknown", 1.0

        best_name = "Unknown"
        best_dist = 1e9

        for name, db_emb in self.database.items():
            d = self.cosine_distance(emb, db_emb)
            if d < best_dist:
                best_dist = d
                best_name = name

        if best_dist > self.thresh:
            return "Unknown", float(best_dist)
        return best_name, float(best_dist)


class PersistentFaceRecognitionSystem(FaceRecognitionSystem):
    """MTCNN + FaceNet with JSON persistence"""

    def __init__(self, db_file: str = "face_identities.json", tfserving_host: str = "localhost:8500", model_name: str = "facenet-2", input_name: str = "input_1", **kwargs):
        super().__init__(tfserving_host=tfserving_host, model_name=model_name, input_name=input_name, **kwargs)
        print("Here 1")
        self.db_file = Path(db_file)
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_database_from_json()

    def _load_database_from_json(self):
        self.database.clear()
        if not self.db_file.exists():
            self._json_store = {"users": [], "face_identities": []}
            print("✅ Loaded 0 identities from JSON")
            return
        try:
            with open(self.db_file, "r", encoding="utf-8") as f:
                self._json_store = json.load(f)
        except Exception:
            self._json_store = {"users": [], "face_identities": []}
        identities = self._json_store.get("face_identities", [])
        for identity in identities:
            embedding = np.array(identity.get("embedding", []), dtype=np.float32)
            if embedding.size:
                self.database[str(identity.get("user_id"))] = embedding
        print(f"✅ Loaded {len(self.database)} identities from JSON")

    def _save_json_store(self):
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(self._json_store, f, ensure_ascii=False, indent=2)

    def _upsert_user(self, user_id: str, email: str | None = None):
        users = self._json_store.setdefault("users", [])
        for u in users:
            if u.get("id") == user_id:
                if email is not None:
                    u["email"] = email
                return
        users.append({"id": user_id, "email": email})

    def _save_embedding(self, name: str, user_id: str, embedding: np.ndarray, num_images: int, quality: float, email: str | None = None):
        now = datetime.now().isoformat()
        self._upsert_user(user_id, email=email)
        identities = self._json_store.setdefault("face_identities", [])
        record = None
        for item in identities:
            if item.get("user_id") == user_id:
                record = item
                break
        payload = {
            "name": name,
            "user_id": user_id,
            "embedding": embedding.tolist(),
            "num_images_used": num_images,
            "embedding_quality": int(quality * 1000),
            "created_at": now,
            "updated_at": now,
        }
        if record is None:
            identities.append(payload)
        else:
            record.update(payload)
            record["created_at"] = record.get("created_at", now)
        self._save_json_store()
        print(f"💾 Saved {user_id} to JSON")

    def add_identity_from_images(self, name: str, user_id: str, img_paths: list[str], email: str | None = None) -> bool:
        success = super().add_identity_from_images(user_id, img_paths)
        if success and user_id in self.database:
            quality = self._calculate_embedding_quality(user_id)
            self._save_embedding(name, user_id, self.database[user_id], len(img_paths), quality, email=email)
        return success

    def get_database_stats(self) -> dict:
        identities = self._json_store.get("face_identities", [])
        qualities = [int(x.get("embedding_quality", 0)) for x in identities]
        avg_quality = sum(qualities) / len(qualities) if qualities else 0
        return {"total_identities": len(identities), "in_memory_cache": len(self.database), "avg_quality": avg_quality, "persistent": True}

    def clear_database(self) -> bool:
        self.database.clear()
        self._json_store = {"users": [], "face_identities": []}
        try:
            self._save_json_store()
            print("🗑️  Cleared entire database!")
            return True
        except Exception as e:
            print(f"❌ Clear failed: {e}")
            return False

    def process_face_jsons(
        self,
        json_pattern: str = "output/*/tracking_jsons/frame_*.json",
        vote_min: int = 3,
        max_dist_threshold: float = 0.6,
    ) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int], Dict[str, Any]]]:
        """
        Đọc tất cả file JSON → crop face → embed → recognize → vote per (user_id, track_id).

        Args:
            json_pattern: pattern tìm file JSON, ví dụ: "liveness_frames/*/track_*.json"
            vote_min: tối thiểu bao nhiêu lần tên giống nhau để công nhận vote
            max_dist_threshold: distance > ngưỡng này → coi như "Unknown"

        Returns:
            raw_results: kết quả chi tiết per frame.
            voted_summary: dict key=(user_id, track_id), value={final_name, vote_count, etc.}
        """
        raw_results = []

        paths = sorted(glob.glob(json_pattern))
        if not paths:
            print("❌ Không tìm thấy file JSON theo pattern:", json_pattern)
            return raw_results, {}

        votes = {}
        best_dist = None
        best_name = None
        for json_path in paths:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if isinstance(meta, list):
                if not meta:
                    continue
                meta = meta[0]

            track_id = meta["trackingId"]
            frame_path = meta["framePath"] # str path relative
            bbox = meta["bbox"]  # list [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox  # unpack trực tiếp từ list
        
            if not frame_path or frame_path.strip() == "":
                raw_results.append({
                    "json_path": json_path,
                    "track_id": track_id,
                    "error": "image_not_found",
                    "name": None,
                    "distance": None,
                    "frame_path": frame_path,
                    "bbox": bbox,
                })
                continue

            image_path = Path(json_path).parent.parent / frame_path
            
            # Đọc ảnh
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                raw_results.append({
                    "json_path": json_path,
                    "track_id": track_id,
                    "error": "read_failed",
                    "name": None,
                    "distance": None,
                    "frame_path": frame_path,
                    "bbox": bbox,
                })
                continue

            # Crop khuôn mặt
            xywh = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            face_rgb = self.crop_face(bgr, xywh)
            if face_rgb is None:
                raw_results.append({
                    "json_path": json_path,
                    "track_id": track_id,
                    "error": "face_crop_failed",
                    "name": None,
                    "distance": None,
                    "frame_path": frame_path,
                    "bbox": bbox,
                })
                continue

            # Tính embedding + nhận diện
            emb = self.get_embedding(face_rgb)
            name, dist = self._find_best_match(emb)

            # Nếu dist quá cao → coi như Unknown
            if dist > max_dist_threshold:
                name = "Unknown"

            # Ghi vào raw kết quả
            raw = {
                "json_path": json_path,
                "track_id": track_id,
                "frame_path": frame_path,
                "name": name,
                "distance": dist,
                "bbox": bbox,
                "error": None,
            }
            raw_results.append(raw)
            
            if name != "Unknown":
                votes[name] = votes.get(name, 0) + 1

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_name = name

        final_name = "Unknown"
        vote_count = 0
        reason = "no_vote"
        if votes:
            final_name = max(votes, key=votes.get)
            vote_count = votes[final_name]
            reason = "majority_vote"      
        
        return final_name, reason
