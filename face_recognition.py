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
from sqlalchemy import create_engine, Column, String, LargeBinary, Integer, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import uuid
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Any
from ultralytics import YOLO
from pathlib import Path


class TFServingClient:
    def __init__(self, host: str = "localhost:8500", model_name: str = "facenet", input_name: str = "input"):
        self.channel = grpc.insecure_channel(host)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        
        self.model_name = model_name
        self.input_name = input_name  # ✅ STORE input_name HERE

        print("Here 3")
        
    def predict(self, input_data: np.ndarray, input_name: str = None):
        """Use self.input_name if not provided"""
        input_name = input_name or self.input_name  # ✅ CRITICAL FIX
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = "serving_default"
        
        request.inputs[input_name].CopyFrom(  # ✅ Uses "input"
            tf.make_tensor_proto(input_data, shape=input_data.shape)
        )
        response = self.stub.Predict(request, timeout=10.0)
        return response
    
    def get_embedding(self, input_data: np.ndarray, input_name: str = None) -> np.ndarray:
        input_name = input_name or self.input_name  # ✅ CRITICAL FIX
        response = self.predict(input_data, input_name)
        embedding = tf.make_ndarray(response.outputs[list(response.outputs.keys())[0]])
        return embedding[0]

class FaceRecognitionSystem:
    """
    - Face capture: detect + crop largest face using MTCNN only
    - Face embedding: FaceNet TF SavedModel
    - Recognition: cosine distance with a simple mean-embedding database
    """

    def __init__(
        self,
        tfserving_host: str = "localhost:8500",  # NEW: TF Serving host
        model_name: str = "facenet",
        input_name: str = "input",  # TF Serving input tensor name
        img_size: int = 160,
        thresh: float = 0.3,
        yolo_model_path="yolo-face.pt",
    ):
        self.img_size = img_size
        self.thresh = thresh
        self.input_name = input_name

        # NEW: Use TF Serving instead of local SavedModel
        self.tf_client = TFServingClient(
            host=tfserving_host, 
            model_name=model_name,
            input_name=input_name
        )

        print("Here 2")
        
        # Lazy-load YOLO to avoid native runtime crashes during app startup.
        self.yolo_model_path = yolo_model_path
        self.yolo = None

        self.database: dict[str, np.ndarray] = {}

    def _get_yolo(self):
        if self.yolo is None:
            self.yolo = YOLO(self.yolo_model_path)
        return self.yolo

    # ----------------------------
    # Utils
    # ----------------------------
    @staticmethod
    def l2_normalize(x: np.ndarray) -> np.ndarray:
        n = norm(x)
        return x / (n if n > 0 else 1e-12)

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        # assumes both are L2-normalized
        return float(1.0 - np.dot(a, b))

    def preprocess(self, rgb_face: np.ndarray) -> np.ndarray:
        face = cv2.resize(rgb_face, (self.img_size, self.img_size))
        face = face.astype("float32")
        mean, std = face.mean(), face.std()
        face = (face - mean) / (std if std > 0 else 1e-6)
        return face

    def detect_face_largest_xywh(self, bgr_img: np.ndarray) -> tuple[int, int, int, int] | None:
        """
        Dùng YOLOv8 face model để detect face, trả về (x, y, w, h) của face lớn nhất.
        """
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        yolo = self._get_yolo()
        results = yolo(rgb, imgsz=640, conf=0.5)

        if not results[0].boxes or len(results[0].boxes) == 0:
            return None

        # Lấy tất cả box
        boxes = results[0].boxes.xyxy.cpu().numpy()  # shape (N, 4) [x1,y1,x2,y2]
        confs = results[0].boxes.conf.cpu().numpy()  # confidence

        if len(boxes) == 0:
            return None

        # Sắp xếp theo diện tích (largest)
        areas = []
        for (x1, y1, x2, y2) in boxes:
            w, h = x2 - x1, y2 - y1
            areas.append(w * h)

        idx = np.argmax(areas)
        x1, y1, x2, y2 = boxes[idx]
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        
        return (x, y, w, h)

    def crop_face(self, bgr_img: np.ndarray, xywh: tuple[int, int, int, int]) -> np.ndarray | None:
        """
        Crop face from BGR image using xywh. Returns RGB cropped face (for FaceNet).
        """
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

    # ----------------------------
    # Face embedding
    # ----------------------------
    def get_embedding(self, rgb_face: np.ndarray) -> np.ndarray:
        """
        rgb_face: cropped face RGB
        returns: L2 normalized embedding
        """
        face = self.preprocess(rgb_face)
        # shape (1, H, W, C)
        face_batch = np.expand_dims(face, 0).astype(np.float32)

        # Use TF Serving
        emb = self.tf_client.get_embedding(face_batch)
        return self.l2_normalize(emb)

    # ----------------------------
    # Database building / enrollment
    # ----------------------------
    def add_identity_from_images(self, user_id: str, img_paths: list[str]) -> bool:
        """
        Build mean embedding from multiple images and add into database.
        Returns True if added.
        """
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
        """
        Calculate embedding quality based on embedding variance.
        Lower variance = higher quality (consistent embeddings).
        """
        if user_id not in self.database:
            return 0.0
    
        # For now, return 1.0 for single embedding
        # In future, could compare multiple embeddings per person
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

# PostgreSQL/Supabase setup
DATABASE_URL = "postgresql+psycopg://postgres.kbmxlrqkzgjbtkmlbaei:we-gonna-pass-with-high-score!@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4) 
    email = Column(String(255), unique=True, nullable=False)

class FaceIdentity(Base):
    __tablename__ = 'face_identities'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=True)
    embedding = Column(LargeBinary)
    num_images_used = Column(Integer, default=0)
    embedding_quality = Column(Integer, default=0)
    created_at = Column(String)
    updated_at = Column(String)

class PersistentFaceRecognitionSystem(FaceRecognitionSystem):
    """
    MTCNN + FaceNet with PostgreSQL persistence
    """
    
    def __init__(
        self, 
        tfserving_host: str = "localhost:8500",
        model_name: str = "facenet-2",
        input_name: str = "input_1",
        db_url: str = DATABASE_URL,
        **kwargs
    ):
        # Pass TF Serving params to parent FaceRecognitionSystem
        super().__init__(
            tfserving_host=tfserving_host,
            model_name=model_name,
            input_name=input_name,
            **kwargs
        )
        print("Here 1")
        self.engine = create_engine(db_url, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        self._load_database_from_db()
    
    def _migrate_face_identities_foreign_key(self):
        """PostgreSQL-safe migration: drop FK from users.face_identity_id, add FK to face_identities.user_id"""
        try:
            Base.metadata.create_all(self.engine)

            with self.engine.connect() as conn:
                # 1. Drop cũ: FK từ users -> face_identities
                conn.execute(text("""
                    ALTER TABLE users 
                    DROP CONSTRAINT IF EXISTS fk_users_face_identity;
                """))

                # 2. Add column user_id trong face_identities
                conn.execute(text("""
                    ALTER TABLE face_identities 
                    ADD COLUMN IF NOT EXISTS user_id UUID;
                """))

                # 3. Thêm FK mới: face_identities.user_id -> users(id)
                conn.execute(text("""
                    DO $$ 
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_constraint c 
                            JOIN pg_class t ON c.conrelid = t.oid 
                            WHERE t.relname = 'face_identities' 
                            AND c.conname = 'fk_face_identities_user'
                        ) THEN
                            ALTER TABLE face_identities 
                            ADD CONSTRAINT fk_face_identities_user 
                            FOREIGN KEY (user_id) 
                            REFERENCES users(id) ON DELETE CASCADE;
                        END IF;
                    END $$;
                """))

                conn.commit()
        except Exception as e:
            print(f"❌ Migration safe skip: {e}")
    
    def _get_db_session(self):
        return self.SessionLocal()
    
    def _load_database_from_db(self):
        """Load ALL embeddings from PostgreSQL on startup"""
        self.database.clear()
        session = self._get_db_session()
        try:
            identities = session.query(FaceIdentity).all()
            for identity in identities:
                embedding = pickle.loads(identity.embedding)
                self.database[str(identity.user_id)] = embedding
            print(f"✅ Loaded {len(self.database)} identities from DB")
        except SQLAlchemyError as e:
            print(f"❌ DB load error: {e}")
        finally:
            session.close()
    
    def _save_embedding(self, name: str, user_id: str, embedding: np.ndarray, num_images: int, quality: float):
        """Save/update embedding in PostgreSQL"""
        user_id_uuid = uuid.UUID(user_id)
        session = self._get_db_session()
        try:
            embedding_bytes = pickle.dumps(embedding)
            
            identity = session.query(FaceIdentity).filter(FaceIdentity.user_id == user_id_uuid).first()
            now = datetime.now().isoformat()
            
            if identity:
                # Update
                identity.embedding = embedding_bytes
                identity.num_images_used = num_images
                identity.embedding_quality = int(quality * 1000)
                identity.updated_at = now
            else:
                # Insert new
                identity = FaceIdentity(
                    name=name,
                    embedding=embedding_bytes,
                    num_images_used=num_images,
                    embedding_quality=int(quality * 1000),
                    created_at=now,
                    updated_at=now,
                    user_id=user_id_uuid
                )
                session.add(identity)
            
            session.commit()
            print(f"💾 Saved {name} to DB")
        except SQLAlchemyError as e:
            print(f"❌ DB save error: {e}")
            session.rollback()
        finally:
            session.close()
    
    def add_identity_from_images(self, name: str, user_id: str, img_paths: list[str]) -> bool:
        """Override: Save to both memory AND PostgreSQL"""
        success = super().add_identity_from_images(user_id, img_paths)
        if success and user_id in self.database:
            quality = self._calculate_embedding_quality(user_id)
            self._save_embedding(name, user_id, self.database[user_id], len(img_paths), quality)
        return success
    
    def get_database_stats(self) -> dict:
        """Full database statistics"""
        session = self._get_db_session()
        try:
            total = session.query(FaceIdentity).count()
            avg_quality = session.query(FaceIdentity).avg(FaceIdentity.embedding_quality) or 0
            return {
                "total_identities": total,
                "in_memory_cache": len(self.database),
                "avg_quality": avg_quality,
                "persistent": True
            }
        finally:
            session.close()
    
    def clear_database(self) -> bool:
        """⚠️ DANGER: Clear entire database"""
        self.database.clear()
        session = self._get_db_session()
        try:
            session.query(FaceIdentity).delete()
            session.commit()
            print("🗑️  Cleared entire database!")
            return True
        except SQLAlchemyError as e:
            session.rollback()
            print(f"❌ Clear failed: {e}")
            return False
        finally:
            session.close()

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