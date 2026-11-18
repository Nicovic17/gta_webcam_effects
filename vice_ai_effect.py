"""Live capture -> asynchronous AI edit -> smooth photograph replacement."""
import cv2
import numpy as np

from neon_scene import ease

TRANSITION_SECONDS = 2.0


class ViceEffect:
    def __init__(self, width, height, job):
        self.width, self.height, self.job = width, height, job
        self.phase = 'idle'
        self.photo = self.artwork = self.origin = None
        self.started = self.transition_at = 0.
        self.message = ''

    @property
    def running(self):
        return self.phase in ('generating', 'transition', 'ready')

    def capture(self, frame, now):
        if self.phase == 'generating':
            return False
        if not self.job.start(frame):
            self.message = 'Attendi la conclusione della richiesta precedente.'
            return False
        self.photo = self._fit(frame)
        self.artwork = self.origin = None
        self.started = now
        self.message = ''
        self.phase = 'generating'
        return True

    def reset(self):
        self.job.cancel()
        self.photo = self.artwork = self.origin = None
        self.phase = 'idle'
        self.message = ''

    def mirror(self):
        # Keep photo, request and returned identity in one orientation.
        self.reset()

    def _fit(self, image):
        h, w = image.shape[:2]
        scale = max(self.width/w, self.height/h)
        resized = cv2.resize(image, (max(self.width, round(w*scale)), max(self.height, round(h*scale))))
        x, y = (resized.shape[1]-self.width)//2, (resized.shape[0]-self.height)//2
        return np.ascontiguousarray(resized[y:y+self.height, x:x+self.width])

    def render(self, frame, mask, now):
        if self.phase == 'generating':
            result = self.job.take_result()
            if result is not None:
                if result.error:
                    self.phase, self.message = 'error', result.error
                    self.photo = None
                else:
                    self.artwork = self._fit(result.image)
                    self.origin = frame.copy()
                    self.transition_at = now
                    self.phase = 'transition'
        if self.phase == 'transition':
            progress = max(0., (now-self.transition_at)/TRANSITION_SECONDS)
            if progress >= 1:
                self.phase = 'ready'
                self.photo = self.origin = None
            else:
                # First settle onto the captured pose, then reveal its artwork.
                settle = ease(progress/.23)
                original = cv2.addWeighted(self.origin, 1-settle, self.photo, settle, 0)
                amount = ease((progress-.08)/.92)
                return cv2.addWeighted(original, 1-amount, self.artwork, amount, 0)
        if self.phase == 'ready':
            return self.artwork.copy()
        return frame.copy()

    def status(self, now):
        if self.phase == 'generating':
            return f'Generazione AI in corso... {max(0, int(now-self.started))} s | R: annulla'
        if self.phase == 'transition':
            return 'Trasformazione GTA... | R: webcam'
        if self.phase == 'ready':
            return 'VICE AI pronta | SPAZIO: nuova foto | R: webcam'
        if self.message:
            return self.message+' | SPAZIO: riprova'
        return 'SPAZIO: foto e invio a OpenAI | R: reset'
