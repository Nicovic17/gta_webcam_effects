"""Face tracking local to GTA Webcam; no imports from Gojo/Sukuna or other projects."""
from pathlib import Path
import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).resolve().parent
FACE_MODEL = BASE_DIR/'face_landmarker.task'
SEGMENTER_MODEL = BASE_DIR/'models/selfie_segmenter_landscape.tflite'

def create_face_tracker():
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,num_faces=1,
        min_face_detection_confidence=.45,min_face_presence_confidence=.45,
        min_tracking_confidence=.45,
    ))

def person_mask(segmenter,image,timestamp_ms,width,height):
    result = segmenter.segment_for_video(image,timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height,width),np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    resized = cv2.resize(raw,(width,height),interpolation=cv2.INTER_LINEAR)
    resized = cv2.GaussianBlur(resized,(0,0),sigmaX=2.2,sigmaY=2.2)
    return np.clip((resized-.12)/.68,0,1)
