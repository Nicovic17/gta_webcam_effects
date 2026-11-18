"""Standalone Windows camera acquisition with the original MJPEG/latest-frame settings."""
import time
import cv2
from video_pipeline import LatestFrameCapture

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
VIRTUAL_CAMERA_FPS = 30

def open_camera(index=0):
    """Apre la webcam provando i backend Windows e attende il primo frame."""
    backends = (
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("automatico", cv2.CAP_ANY),
    )
    attempts = []

    for backend_name, backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            attempts.append(f"{backend_name}: non disponibile")
            cap.release()
            continue

        # 1280x720 e una modalita molto piu comune di 960x540. Il frame viene
        # ridimensionato solo dopo l'acquisizione, per la visualizzazione.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, VIRTUAL_CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # DirectShow can reset the codec when size/FPS changes. Set MJPEG LAST:
        # requesting it before the resolution silently falls back to slow YUY2.
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

        for _ in range(30):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                code=int(cap.get(cv2.CAP_PROP_FOURCC))
                codec=''.join(chr((code>>8*i)&255) for i in range(4)).strip('\x00')
                print(f"[WEBCAM] {backend_name}: {frame.shape[1]}x{frame.shape[0]}, formato {codec or 'automatico'}")
                return LatestFrameCapture(cap,frame), None
            time.sleep(0.05)

        attempts.append(f"{backend_name}: aperta, ma nessun frame ricevuto")
        cap.release()

    details = "; ".join(attempts)
    raise RuntimeError(
        "Non riesco a ricevere immagini dalla webcam. "
        "Chiudi eventuali programmi che la stanno usando e controlla i permessi "
        f"di Windows. Tentativi: {details}"
    )
