"""VICE AI runtime: network generation never runs on the webcam thread."""
from contextlib import ExitStack
import time

import cv2
import pyvirtualcam

from preview_controls import MirrorControl
from vice_ai_effect import ViceEffect
from vice_openai import ImageConfig, ImageEditJob, OpenAIImageEditor, ViceAPIError
from video_pipeline import FramePacer, VideoStats


def load_config():
    try:
        return ImageConfig.load()
    except ViceAPIError as error:
        if 'OPENAI_API_KEY' not in str(error):
            raise
    # Only used when no env/local key is configured. The value stays in RAM.
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk()
    root.withdraw()
    try:
        key = simpledialog.askstring('VICE MODE - OpenAI',
            'Inserisci la tua API key OpenAI. Lo scatto verra inviato premendo SPAZIO.',
            show='*', parent=root)
    finally:
        root.destroy()
    if not key or not key.strip():
        return None
    # Reuse the same validation and optional local model settings.
    import os
    return ImageConfig.load(environ={**os.environ, 'OPENAI_API_KEY': key.strip()})


def run_vice_webcam(camera_opener, *, width=960, height=540, fps=30):
    config = load_config()
    if config is None:
        return 0
    title = 'VICE MODE AI Webcam'
    with ExitStack() as resources:
        job = ImageEditJob(OpenAIImageEditor(config))
        resources.callback(job.close)
        effect = ViceEffect(width, height, job)
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
        fullscreen, failures = False, 0
        print('[VICE AI] SPAZIO: scatta e invia a OpenAI; R: webcam/annulla; M: flip; F: fullscreen; Q/ESC: esci.')
        while True:
            before = time.perf_counter()
            if pending is not None:
                frame, pending, ok = pending, None, True
            else:
                ok, frame = capture.read()
            if not ok or frame is None or not frame.size:
                failures += 1
                if failures >= 30:
                    raise RuntimeError('La webcam ha smesso di inviare fotogrammi.')
                time.sleep(.03)
                continue
            failures = 0
            if generation != mirror.generation:
                generation = mirror.generation
                effect.mirror()
            if mirror.mirrored:
                frame = cv2.flip(frame, 1)
            full_resolution = frame
            frame = cv2.resize(frame, (width, height))
            now = time.monotonic()
            display = effect.render(frame, None, now)
            virtual.send(display)
            preview = mirror.draw(display)
            text = effect.status(now)
            font = max(.23, min(.48, width/1400))
            # Keep status readable within the preview, including API failures.
            words, line, lines = text.split(), '', []
            for word in words:
                candidate = (line+' '+word).strip()
                if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font, 1)[0][0] > width-16 and line:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
            for index, line in enumerate(lines[:3]):
                y = max(13, height//18)+index*max(13, round(height*.035))
                cv2.putText(preview, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, font,
                            (235, 235, 245), 1, cv2.LINE_AA)
            cv2.imshow(title, preview)
            key = cv2.waitKey(1)&0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(' '):
                effect.capture(full_resolution, time.monotonic())
            elif key == ord('r'):
                effect.reset()
            elif key == ord('m'):
                mirror.toggle()
            elif key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            stats.tick()
            pacer.wait(before)
    return 0
