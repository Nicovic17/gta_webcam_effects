"""Live local img2img mode, with independent camera/display and GPU workers."""
from contextlib import ExitStack
import time

import cv2
import pyvirtualcam

from neon_scene import ease
from preview_controls import MirrorControl
from vice_local_client import LocalStreamClient
from vice_local_settings import DEFAULT_PROMPT
from vice_prompt_panel import PromptPanel
from video_pipeline import FramePacer, VideoStats


def run_local_webcam(camera_opener, *, width=960, height=540, fps=30):
    title = 'VICE Locale - StreamDiffusion'
    with ExitStack() as resources:
        panel = PromptPanel()
        resources.callback(panel.close)
        client = LocalStreamClient()
        resources.callback(client.close)
        capture, pending = camera_opener(0)
        resources.callback(capture.release)
        try:
            virtual = pyvirtualcam.Camera(width=width, height=height, fps=fps,
                fmt=pyvirtualcam.PixelFormat.BGR, backend='obs')
        except Exception as error:
            raise RuntimeError('OBS Virtual Camera non disponibile. Chiudi le altre istanze.') from error
        resources.callback(virtual.close)
        resources.callback(cv2.destroyAllWindows)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, width, height)
        mirror = MirrorControl(width, height)
        cv2.setMouseCallback(title, mirror.on_mouse)
        mirror_generation = mirror.generation
        generation, strength, failures = 0, 2, 0
        prompt = DEFAULT_PROMPT
        enabled = fullscreen = False
        fade_at = None
        pacer, stats = FramePacer(fps), VideoStats(title)
        print('[VICE LOCALE] Caricamento GPU. SPAZIO: attiva/disattiva; 1/2/3: intensita; '
              'R: reset; M: flip; F: fullscreen; P: prompt; Q/ESC: esci. Nessuna API OpenAI.')
        while True:
            before = time.perf_counter()
            change = panel.poll()
            if change.prompt is not None or change.toggle or change.strength is not None:
                if change.prompt is not None:
                    prompt = change.prompt
                if change.toggle:
                    enabled = not enabled
                if change.strength is not None:
                    strength = change.strength
                generation += 1
                client.clear(generation)
                fade_at = None
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
            if mirror.generation != mirror_generation:
                mirror_generation = mirror.generation
                generation += 1
                client.clear(generation)
                fade_at = None
            if mirror.mirrored:
                frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (width, height))
            now = time.monotonic()
            if enabled and not client.error:
                client.submit(frame, generation, strength, prompt)
            tracked = client.latest()
            display = frame
            if enabled and not client.error and tracked is not None and tracked.generation == generation:
                if fade_at is None:
                    fade_at = now
                amount = ease((now-fade_at)/.45)
                generated = cv2.resize(tracked.image, (width, height), interpolation=cv2.INTER_LINEAR)
                display = cv2.addWeighted(frame, 1-amount, generated, amount, 0)
            virtual.send(display)
            preview = mirror.draw(display)
            if client.error:
                hint = 'Errore GPU. Controlla .vice-local/worker.log e riavvia.'
            elif not client.ready:
                hint = 'Caricamento StreamDiffusion sulla GPU...'
            elif enabled and tracked is not None:
                hint = f'Locale ON | livello {strength} | AI {tracked.inference_ms:.0f} ms | ritardo {tracked.elapsed_ms:.0f} ms'
            elif enabled:
                hint = 'Generazione locale del primo frame...'
            else:
                hint = 'SPAZIO: VICE locale | 1/2/3: intensita | P: prompt'
            font = max(.23, min(.48, width/1400))
            cv2.putText(preview, hint, (8, max(13, height//18)), cv2.FONT_HERSHEY_SIMPLEX,
                        font, (235, 235, 245), 1, cv2.LINE_AA)
            cv2.imshow(title, preview)
            key = cv2.waitKey(1)&0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord(' '), ord('r')):
                enabled = not enabled if key == ord(' ') else False
                generation += 1
                client.clear(generation)
                fade_at = None
            elif key in (ord('1'), ord('2'), ord('3')):
                strength = key-ord('0')
                generation += 1
                client.clear(generation)
                fade_at = None
            elif key == ord('m'):
                mirror.toggle()
            elif key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == ord('p'):
                panel.show()
            panel.set_state(enabled, strength)
            stats.tick()
            pacer.wait(before)
    return 0
