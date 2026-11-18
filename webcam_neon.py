"""Shared WANTED/VICE runtime with async local masks and optional local audio."""
from contextlib import ExitStack
from pathlib import Path
import time

import cv2
import pyvirtualcam
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gta_webcam_effects import WantedEffect, ViceEffect
from preview_controls import MirrorControl
from vanish_sound import VanishSound
from video_pipeline import LatestAnalysis, FramePacer, VideoStats, tracking_image

BASE_DIR = Path(__file__).resolve().parent
SOUNDS = {'wanted': BASE_DIR/'assets'/'gtapolice.mp3',
          'vice': BASE_DIR/'assets'/'vice_intro.wav'}


def create_neon_analyzer(width, height, *, mode='vice'):
    from gta_tracking import create_face_tracker, person_mask, SEGMENTER_MODEL
    resources = ExitStack()
    try:
        detector = create_face_tracker()
        resources.callback(detector.close)
        segmenter = None
        if mode == 'vice':
            segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
                running_mode=vision.RunningMode.VIDEO,
                output_confidence_masks=True, output_category_mask=False))
            resources.callback(segmenter.close)
        state = dict(mask=None, face_box=None, face_at=float('-inf'), generation=0)

        def analyze(packet):
            frame, stamp, now, active, generation = packet
            if generation != state['generation']:
                state.update(mask=None, face_box=None, face_at=float('-inf'), generation=generation)
            image = tracking_image(frame, max_width=480)
            if active and segmenter is not None:
                state['mask'] = person_mask(segmenter, image, stamp, width, height)
            else:
                faces = detector.detect_for_video(image, stamp).face_landmarks
                state['face_box'] = None
                if faces:
                    xs, ys = [p.x for p in faces[0]], [p.y for p in faces[0]]
                    state['face_box'] = (min(xs), min(ys), max(xs), max(ys))
                state['face_at'] = now
                state['mask'] = None
            return state.copy()

        return LatestAnalysis(analyze, cleanup=resources.close)
    except Exception:
        resources.close()
        raise


def run_neon_webcam(mode, camera_opener, *, width=960, height=540, fps=30):
    if mode == 'vice':
        from webcam_vice_ai import run_vice_webcam
        return run_vice_webcam(camera_opener, width=width, height=height, fps=fps)
    if mode not in ('wanted', 'vice'):
        raise ValueError(f'Modalita non riconosciuta: {mode}')
    title = 'WANTED Webcam' if mode == 'wanted' else 'VICE MODE Webcam'
    effect = WantedEffect(width, height)
    with ExitStack() as resources:
        analysis = create_neon_analyzer(width, height, mode=mode)
        resources.callback(analysis.close)
        sound = None
        if mode in SOUNDS:
            try:
                sound = VanishSound(SOUNDS[mode])
            except RuntimeError as error:
                print(f'[{title}] Audio non disponibile: {error}')
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
        generation = mirror.generation
        cv2.setMouseCallback(title, mirror.on_mouse)
        pacer, stats = FramePacer(fps), VideoStats(title)
        started = time.monotonic()
        last_submit, last_stamp = float('-inf'), -1
        mask, face_box, fullscreen, failures = None, None, False, 0
        wanted_audio_started = False
        print(f'[{title}] SPAZIO: attiva/disattiva, R: reset, M: flip, F: fullscreen, Q/ESC: esci. '
              'Clicca l\'anteprima per usare i tasti.')
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
                mask = face_box = None
                last_submit = float('-inf')
            frame = cv2.resize(frame, (width, height))
            now = time.monotonic()
            if (mode == 'vice' and effect.running) or now-last_submit >= (.10 if mode == 'wanted' else .18):
                last_stamp = max(last_stamp+1, int((now-started)*1000))
                analysis.submit((frame.copy(), last_stamp, now, effect.running, generation))
                last_submit = now
            tracked = analysis.latest()
            if tracked is not None and tracked.data['generation'] == generation:
                state = tracked.data
                mask = state['mask']
                face_box = state['face_box'] if now-state['face_at'] < .75 else None
            display = (effect.render(frame, None, now, face_box=face_box) if mode == 'wanted'
                       else effect.render(frame, mask, now))
            # Mirror the completed scene, including the HUD and WANTED photo card.
            if mirror.mirrored:
                display = cv2.flip(display, 1)
            if (mode == 'wanted' and effect.running and not wanted_audio_started
                    and effect.star_count(now-effect.started) == 5):
                wanted_audio_started = True
                if sound is not None:
                    try:
                        sound.play(loop=True)
                    except RuntimeError as error:
                        print(f'[{title}] Audio non disponibile: {error}')
                        sound = None
            virtual.send(display)
            preview = mirror.draw(display)
            hint = 'SPAZIO: disattiva / R: reset' if effect.running else f'SPAZIO: {"WANTED" if mode == "wanted" else "VICE MODE"}'
            cv2.putText(preview, hint, (max(3, width//60), max(13, height//18)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(.25, min(.5, width/1400)), (235, 235, 245), 1, cv2.LINE_AA)
            cv2.imshow(title, preview)
            key = cv2.waitKey(1)&0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(' '):
                if effect.running:
                    effect.reset()
                    wanted_audio_started = False
                    mask = None
                    if sound is not None:
                        sound.stop()
                elif effect.trigger(time.monotonic(), face_box):
                    wanted_audio_started = False
                    if mode == 'vice' and sound is not None:
                        try:
                            sound.play()
                        except RuntimeError as error:
                            print(f'[{title}] Audio non disponibile: {error}')
                            sound = None
            elif key == ord('r'):
                effect.reset()
                wanted_audio_started = False
                mask = None
                if sound is not None:
                    sound.stop()
            elif key == ord('m'):
                mirror.toggle()
            elif key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            stats.tick(tracked)
            pacer.wait(frame_started)
    return 0
