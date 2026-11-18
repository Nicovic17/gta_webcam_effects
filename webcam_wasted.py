"""WASTED mode: async face anchor, local replay, optional sound and clean OBS feed."""
from contextlib import ExitStack
from pathlib import Path
import time

import cv2
import pyvirtualcam

from preview_controls import MirrorControl
from video_pipeline import LatestAnalysis, FramePacer, VideoStats, tracking_image
from vanish_sound import VanishSound
from wasted_effect import WastedEffect

WASTED_SOUND = Path(__file__).resolve().parent/'assets'/'wasted.mp3'


class LanguageControl:
    """Switch caption language in the preview without affecting the OBS feed."""
    def __init__(self, effect, mirror, width, height):
        self.effect, self.mirror = effect, mirror
        scale = min(1., width/720, height/405)
        self.font = max(.24, .5*scale)
        w = min(width-8, max(90, round(180*scale)))
        h = min(height-4, max(18, round(34*scale)))
        margin = max(2, round(12*scale))
        self.bounds = (width-margin-w, margin, width-margin, margin+h)

    def toggle(self):
        self.effect.set_language('eng' if self.effect.language == 'it' else 'it')

    def on_mouse(self, event, x, y, flags, userdata):
        x1, y1, x2, y2 = self.bounds
        if event == cv2.EVENT_LBUTTONUP and x1 <= x < x2 and y1 <= y < y2:
            self.toggle()
        else:
            self.mirror.on_mouse(event, x, y, flags, userdata)

    def draw(self, preview):
        x1, y1, x2, y2 = self.bounds
        cv2.rectangle(preview, (x1, y1), (x2-1, y2-1), (34, 43, 48), -1)
        cv2.rectangle(preview, (x1, y1), (x2-1, y2-1), (125, 220, 175), 1)
        text = f'{self.effect.language.upper()}  [L] IT / ENG'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font, 1)
        cv2.putText(preview, text, (x1+(x2-x1-tw)//2, y1+(y2-y1+th)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font, (235, 242, 246), 1, cv2.LINE_AA)


def create_face_analyzer():
    from gta_tracking import create_face_tracker
    detector = create_face_tracker()

    def analyze(packet):
        frame, stamp, now, generation = packet
        faces = detector.detect_for_video(tracking_image(frame, max_width=480), stamp).face_landmarks
        box = None
        if faces:
            xs, ys = [p.x for p in faces[0]], [p.y for p in faces[0]]
            box = (min(xs), min(ys), max(xs), max(ys))
        return box, now, generation

    try:
        return LatestAnalysis(analyze, cleanup=detector.close)
    except Exception:
        detector.close()
        raise


def run_wasted_webcam(camera_opener, *, width=960, height=540, fps=30):
    title = 'WASTED Webcam'
    effect = WastedEffect(width, height)
    with ExitStack() as resources:
        analysis = create_face_analyzer()
        resources.callback(analysis.close)
        sound = None
        try:
            sound = VanishSound(WASTED_SOUND)
        except RuntimeError as error:
            print(f'[WASTED] Audio non disponibile: {error}')
        if sound is not None:
            resources.callback(sound.close)
        capture, pending = camera_opener(0)
        resources.callback(capture.release)
        try:
            virtual = pyvirtualcam.Camera(width=width, height=height, fps=fps,
                                          fmt=pyvirtualcam.PixelFormat.BGR, backend='obs')
        except Exception as error:
            raise RuntimeError('Non riesco ad avviare OBS Virtual Camera. Disattiva la sua '
                               'Virtual Camera in OBS e chiudi altre istanze del programma.') from error
        resources.callback(virtual.close)
        resources.callback(cv2.destroyAllWindows)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, width, height)
        mirror = MirrorControl(width, height)
        language = LanguageControl(effect, mirror, width, height)
        generation = mirror.generation
        cv2.setMouseCallback(title, language.on_mouse)
        pacer, stats = FramePacer(fps), VideoStats(title)
        started = time.monotonic()
        last_scan, last_stamp = float('-inf'), -1
        face_box, failures, fullscreen = None, 0, False
        print('[WASTED] SPAZIO: avvia, L: lingua IT/ENG, R: annulla, M: flip, F: fullscreen, Q/ESC: esci. '
              'Respawn automatico dopo circa 5 secondi. Clicca l\'anteprima per usare i tasti.')
        print(f'[WEBCAM VIRTUALE] Attiva: {virtual.device}')
        while True:
            frame_started = time.perf_counter()
            if pending is not None:
                frame, pending, ok = pending, None, True
            else:
                ok, frame = capture.read()
            if not ok or frame is None or not frame.size:
                failures += 1
                if failures >= 30:
                    raise RuntimeError('La webcam ha smesso di inviare fotogrammi per circa un secondo.')
                time.sleep(.03)
                continue
            failures = 0
            if generation != mirror.generation:
                generation = mirror.generation
                effect.reset()
                face_box, last_scan = None, float('-inf')
                if sound is not None:
                    sound.stop()
            frame = cv2.resize(frame, (width, height))
            now = time.monotonic()
            if not effect.running and now-last_scan >= .18:
                last_stamp = max(last_stamp+1, int((now-started)*1000))
                analysis.submit((frame.copy(), last_stamp, now, generation))
                last_scan = now
            tracked = analysis.latest()
            if tracked is not None:
                box, detected_at, detected_generation = tracked.data
                if detected_generation == generation:
                    face_box = box if now-detected_at < .75 else None
            display = effect.render(frame, now)
            # Mirror the completed scene so WASTED, captions and RESPAWN follow the camera.
            if mirror.mirrored:
                display = cv2.flip(display, 1)
            virtual.send(display)
            preview = mirror.draw(display)
            label = 'R: annulla / respawn automatico' if effect.running else 'SPAZIO: WASTED'
            cv2.putText(preview, label, (max(3, width//60), max(13, height//18)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(.25, min(.5, width/1400)),
                        (230, 235, 245), 1, cv2.LINE_AA)
            language.draw(preview)
            cv2.imshow(title, preview)
            key = cv2.waitKey(1)&0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(' ') and effect.trigger(time.monotonic(), face_box):
                if sound is not None:
                    try:
                        sound.play()
                    except RuntimeError as error:
                        print(f'[WASTED] Audio non disponibile: {error}')
                        sound = None
            elif key == ord('r'):
                effect.reset()
                if sound is not None:
                    sound.stop()
            elif key == ord('m'):
                mirror.toggle()
            elif key == ord('l'):
                language.toggle()
            elif key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            stats.tick(tracked)
            pacer.wait(frame_started)
    return 0
