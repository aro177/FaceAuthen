from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import cv2
import tempfile
import shutil
from typing import Optional, List
from pydantic import BaseModel
import uuid
from pathlib import Path
from liveness_detection import LivenessDetector
from face_recognition import PersistentFaceRecognitionSystem
import json
from datetime import datetime
from uuid import UUID
import multiprocessing as mp
import numpy as np
import base64
from PIL import Image
import threading

try:
    mp.set_start_method('spawn')
except RuntimeError:
    # Already set by parent process/runtime (common with uvicorn workers).
    pass

app = FastAPI(title="Face Authentication API", version="1.0.0")

# Config
SAVE_N_FRAMES = 200
TEMP_DIR = Path("temp_videos")
TEMP_DIR.mkdir(exist_ok=True)

# Global Face Recognition System
face_recog_system = None
liveness_detector = None
liveness_lock = threading.Lock()

class FaceResult(BaseModel):
    name: str
    dist: float
    confidence: float  # (1 - dist) * 100
    box: List[int]     # [x, y, w, h]

class RecognitionResponse(BaseModel):
    faces: List[FaceResult]
    total_faces: int
    image_width: int
    image_height: int

class AuthResponse(BaseModel):
    success: bool
    message: str
    name: Optional[str] = None
    confidence: Optional[float] = None
    liveness_passed: bool = False
    reason: str = ""

class FaceRecogResponse(BaseModel):
    success: bool
    name: str
    confidence: float
    message: str
    reason: str
    details: dict = {}

def init_system():
    global face_recog_system, liveness_detector
    if face_recog_system is None:
        print("🔄 Initializing PersistentFaceRecognitionSystem with TF Serving...")
        
        # ✅ UPDATED: TF Serving parameters instead of local model path
        face_recog_system = PersistentFaceRecognitionSystem()
        print("OK")
        print(f"✅ TF Serving connected: {os.getenv('TF_SERVING_HOST', 'localhost:8500')}")
        print(f"✅ Database loaded with {len(face_recog_system.database)} identities")
    if liveness_detector is None:
        print("🔄 Initializing shared LivenessDetector...")
        liveness_detector = LivenessDetector()
        print("✅ LivenessDetector initialized")
    return face_recog_system

@app.on_event("startup")
async def startup_event():
    init_system()

@app.get("/")
async def root():
    return {
        "message": "Face Authentication API (TF Serving)",
        "endpoints": {
            "/auth/video": "POST - Upload video for authentication",
            "/health": "GET - Health check",
            "/enroll/{user_name}/{user_id}": "POST - Enroll new user"
        },
        "tf_serving": os.getenv("TF_SERVING_HOST", "localhost:8500")
    }

@app.get("/health")
async def health_check():
    if not face_recog_system:
        return {"status": "initializing", "database_size": 0}
    
    return {
        "status": "healthy",
        "tf_serving_host": getattr(face_recog_system, 'tfserving_host', 'unknown'),
        "database_size": len(face_recog_system.database),
        "identities": list(face_recog_system.database.keys())[:5],  # First 5
        "timestamp": datetime.now().isoformat()
    }

@app.post("/auth/video", response_model=AuthResponse)
async def authenticate_video(file: UploadFile = File(..., description="Video file (mp4, avi, mov)")):
    """
    Complete authentication pipeline:
    1. Run LivenessDetectorBytetrack.analyze_video (ByteTrack + smile + blink)
    2. If liveness_passed, save frames + JSONs in liveness_frames/user_<session_id>
    3. Run face_recog_system.process_face_jsons(...) on newly saved JSONs
    4. Return recognized name or "NEW_USER"
    """
    if not face_recog_system:
        init_system()

    # Validate file
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")

    session_id = str(uuid.uuid4())
    temp_video_path = TEMP_DIR / f"{session_id}_{file.filename}"
    os.makedirs(TEMP_DIR, exist_ok=True)

    try:
        # 1. Save uploaded video
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📹 Processing video: {temp_video_path}")

        # 2. 👁️ Liveness: dùng LivenessDetectorBytetrack.analyze_video (thay cho for loop)
        # Cấu hình: userID theo session ID, lưu vào liveness_frames/user_<session_id>
        if liveness_detector is None:
            init_system()
        with liveness_lock:
            # Ensure request-local state does not leak across concurrent requests.
            liveness_detector.reset()
            liveness_result = liveness_detector.analyze_video(str(temp_video_path), session_id=f"user_{session_id}")

        liveness_passed = liveness_result["is_live"]

        if not liveness_passed:
            return AuthResponse(
                success=False,
                message="Liveness test failed",
                liveness_passed=False,
                reason="liveness_failed",
                details={
                    "smile": liveness_result["smile"],
                    "blink": liveness_result["blink"]
                }
            )

        print("✅ Liveness PASSED!")

        #Face Recog
        json_pattern = f"output/user_{session_id}/tracking_jsons/frame_*.json"

        final_name, reason = face_recog_system.process_face_jsons(
            json_pattern=json_pattern,
            vote_min=2,
            max_dist_threshold=0.6
        )

        # 4. 🏁 Tạo AuthResponse
        if final_name != "Unknown":

            return AuthResponse(
                success=True,
                message="Authentication successful",
                name=final_name,
                liveness_passed=True,
                reason="liveness_and_face_recognition"
            )
        else:
            # Liveness pass nhưng không nhận diện được khuôn mặt đã biết
            return AuthResponse(
                success=True,
                message="New face detected - enrollment required",
                name=final_name,
                confidence=0.0,
                liveness_passed=True,
                reason="new_face",
            )

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        # Dọn file tạm
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        print(f"🧹 Cleaned up session {session_id}")


@app.post("/enroll_single/{user_name}/{user_id}")
async def enroll_user_single(
    user_id: UUID,
    user_name: str,
    file: UploadFile = File(..., description="Single face image (jpg, png, jpeg)")
):
    """
    Enroll new user with SINGLE image - fast enrollment
    """
    if not face_recog_system:
        init_system()

    # Validate username
    if not user_name or len(user_name.strip()) < 2:
        raise HTTPException(status_code=400, detail="Username must be 2+ characters")

    user_name = user_name.strip().replace(" ", "_")
    user_id_str = str(user_id)
    session_id = str(uuid.uuid4())

    # Create temp directories
    frames_dir = TEMP_DIR / f"enroll_single_{session_id}"
    person_dir = frames_dir / user_name
    person_dir.mkdir(parents=True, exist_ok=True)

    temp_image_path = TEMP_DIR / f"enroll_single_{session_id}.jpg"

    try:
        # === STEP 1: Validate and save image ===
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image (jpg/png)")

        # Convert to JPG if needed
        if file.content_type == 'image/png':
            image = Image.open(file.file)
            image = image.convert('RGB')
            image.save(temp_image_path, 'JPEG', quality=95)
        else:
            with open(temp_image_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # === STEP 2: Validate face detection ===
        bgr = cv2.imread(str(temp_image_path))
        if bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Resize for consistency
        bgr = cv2.resize(bgr, (640, 480))

        # Save the single frame
        frame_path = person_dir / "single_00.jpg"
        cv2.imwrite(str(frame_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        actual_frames = 1

        # === STEP 3: Build database using add_identity_from_images ===
        img_paths = [str(frame_path)]
        success = face_recog_system.add_identity_from_images(user_name, user_id_str, img_paths)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to create face embedding from image"
            )

        print(f"✅ Database added: {user_name}, {user_id_str}")

        return {
            "success": True,
            "message": f"✅ Enrolled {user_id_str} with single image!",
            "user_name": user_name,
            "total_frames_extracted": actual_frames,
            "frames_used": 1,
            "embedding_quality": 1.0,  # Single image baseline
            "face_detected": True,
            "total_identities": len(face_recog_system.database)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Single image enrollment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        # Cleanup
        if temp_image_path.exists():
            temp_image_path.unlink()
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        print(f"✅ Single image enrollment complete for {user_name}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)