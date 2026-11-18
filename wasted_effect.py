"""A short cinematic webcam knockout and respawn, driven by elapsed time."""
from collections import deque
from pathlib import Path
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEATH_SECONDS = 4.15
RESPAWN_SECONDS = 1.10
TOTAL_SECONDS = DEATH_SECONDS+RESPAWN_SECONDS
TITLE_AT = .85
REASONS = (
    'Hai detto "mi sentite?" per la quinta volta.',
    'La riunione poteva essere una mail.',
    'Hai condiviso lo schermo sbagliato.',
    'Hai risposto "anche a te" al momento sbagliato.',
    'Ti hanno chiesto un ultimo piccolo favore.',
    'Hai aperto Teams prima del caffe.',
    'Il microfono era acceso. Da venti minuti.',
    'Hai promesso: "ci metto solo cinque minuti".',
    'Il venerdi e arrivata una nuova deadline.',
    'Hai scritto nella chat sbagliata.',
)
REASONS_EN = (
    'You said "Can you hear me?" for the fifth time.',
    'This meeting could have been an email.',
    'You shared the wrong screen.',
    'You said "You too" at the wrong moment.',
    'They asked for one last tiny favor.',
    'You opened Teams before your coffee.',
    'Your mic was on. For twenty minutes.',
    'You promised: "It will only take five minutes."',
    'A new deadline just dropped. On a Friday.',
    'You posted in the wrong chat.',
)
REASONS_BY_LANGUAGE = {'it': REASONS, 'eng': REASONS_EN}


def _ease(value):
    value = max(0.0, min(1.0, value))
    return value*value*(3-2*value)


def _text_sprite(text, width, size, *, headline=False, color=(245, 245, 245)):
    candidates = (Path('C:/Windows/Fonts')/('impact.ttf' if headline else 'segoeuib.ttf'),
                  Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    path = next((p for p in candidates if p.is_file()), None)
    # Rasterise once at double resolution; all playback uses cached sprites.
    font = ImageFont.truetype(str(path), max(8, size*2)) if path else ImageFont.load_default(size=max(8, size*2))
    stroke = max(1, round(size*.035))*2
    box = font.getbbox(text, stroke_width=stroke)
    image = Image.new('RGBA', (box[2]-box[0]+16, box[3]-box[1]+16))
    ImageDraw.Draw(image).text((8-box[0], 8-box[1]), text, font=font,
                              fill=(*color, 255), stroke_width=stroke,
                              stroke_fill=(0, 0, 0, 235))
    ratio = min(.5, width/image.width)
    image = image.resize((max(1, round(image.width*ratio)), max(1, round(image.height*ratio))),
                         Image.Resampling.LANCZOS)
    rgba = np.asarray(image)
    return (np.ascontiguousarray(rgba[:, :, 2::-1]),
            np.ascontiguousarray(rgba[:, :, 3].astype(np.float32)/255))


class WastedEffect:
    def __init__(self, width, height, seed=610, language='it'):
        self.width, self.height = width, height
        self.history = deque(maxlen=20)
        self.reason_index = -1
        order = np.random.default_rng(seed).permutation(len(REASONS))
        self._reasons = {lang: [reasons[int(i)] for i in order]
                         for lang, reasons in REASONS_BY_LANGUAGE.items()}
        self.title = _text_sprite('WASTED', width*.66, round(width*.16),
                                  headline=True, color=(210, 58, 62))
        self._captions = {
            lang: [_text_sprite(text, width*.86, max(10, round(width*.022)))
                   for text in reasons]
            for lang, reasons in self._reasons.items()
        }
        self.set_language(language)
        self.respawn = _text_sprite('RESPAWN', width*.38, max(16, round(width*.055)),
                                   headline=True, color=(164, 237, 226))
        y, x = np.mgrid[:height, :width].astype(np.float32)
        vignette = np.clip(1-.62*((x/width-.5)**2+(y/height-.45)**2), .52, 1)
        self.vignette = cv2.merge([vignette]*3)
        self.reset()

    def set_language(self, language):
        if language not in REASONS_BY_LANGUAGE:
            raise ValueError('WASTED language must be it or eng')
        self.language = language
        self.reasons = self._reasons[language]
        self.captions = self._captions[language]

    @property
    def running(self):
        return self.started is not None

    def reset(self):
        self.started = None
        self.history.clear()
        self.replay = []
        self.last_recorded = float('-inf')
        self.anchor = (.5, .4)

    def trigger(self, now, face_box=None):
        if self.running:
            return False
        self.started = now
        self.replay = [(stamp, image) for stamp, image in self.history if now-stamp <= .8]
        self.reason_index = (self.reason_index+1) % len(self.reasons)
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            self.anchor = (float(np.clip((x1+x2)/2, .15, .85)),
                           float(np.clip((y1+y2)/2, .15, .85)))
        return True

    def _record(self, frame, now):
        if now-self.last_recorded < .055:
            return
        scale = min(1., 640/frame.shape[1])
        small = cv2.resize(frame, (max(1, round(frame.shape[1]*scale)),
                                   max(1, round(frame.shape[0]*scale))), interpolation=cv2.INTER_AREA)
        self.history.append((now, small))
        while self.history and now-self.history[0][0] > .8:
            self.history.popleft()
        self.last_recorded = now

    def _source(self, frame, age):
        if not self.replay:
            self.replay = [(self.started, frame.copy())]
        # A real pre-trigger replay at 22% speed, then a held final frame.
        target = self.replay[0][0]+age*.22
        chosen = self.replay[0][1]
        for stamp, image in self.replay:
            if stamp > target:
                break
            chosen = image
        return chosen

    def _draw_sprite(self, scene, sprite, center_y, opacity=1., scale=1.):
        color, alpha = sprite
        if opacity <= 0:
            return
        if scale != 1:
            size = (max(1, round(color.shape[1]*scale)), max(1, round(color.shape[0]*scale)))
            color, alpha = cv2.resize(color, size), cv2.resize(alpha, size)
        height, width = color.shape[:2]
        x, y = (self.width-width)//2, round(center_y*self.height-height/2)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x+width), min(self.height, y+height)
        if x1 >= x2 or y1 >= y2:
            return
        color = color[y1-y:y2-y, x1-x:x2-x]
        weight = np.ascontiguousarray(alpha[y1-y:y2-y, x1-x:x2-x]*np.float32(opacity))
        roi = scene[y1:y2, x1:x2]
        cv2.blendLinear(roi, color, 1-weight, weight, dst=roi)

    def _death_scene(self, frame, age):
        source = self._source(frame, age)
        progress = _ease(age/3.5)
        zoom = 1+.16*_ease(age/1.2)+.035*progress
        cx, cy = self.anchor[0]*self.width, self.anchor[1]*self.height
        transform = cv2.getRotationMatrix2D((cx, cy), -1.4*_ease(age/.8), zoom)
        transform[0, 2] += math.sin(age*61)*self.width*.008*math.exp(-age*10)
        transform[1, 2] += math.cos(age*43)*self.height*.007*math.exp(-age*10)
        transform[0, 2] += (.5-self.anchor[0])*self.width*.3*progress
        transform[1, 2] += (.4-self.anchor[1])*self.height*.3*progress
        transform[:, 0] *= self.width/source.shape[1]
        transform[:, 1] *= self.height/source.shape[0]
        scene = cv2.warpAffine(source, transform, (self.width, self.height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        gray = cv2.cvtColor(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        desaturation = _ease(age/.62)
        scene = cv2.addWeighted(scene, 1-desaturation, gray, desaturation, 0)
        scene = cv2.multiply(scene, self.vignette, dtype=cv2.CV_8U)
        scene = cv2.convertScaleAbs(scene, alpha=1-.24*_ease(age/.8))
        # One subdued impact tint, followed by monochrome cinematography.
        tint = .12*math.exp(-age*14)
        if tint > .002:
            overlay = np.full_like(scene, (18, 20, 110))
            cv2.addWeighted(scene, 1-tint, overlay, tint, 0, dst=scene)
        bar = round(self.height*.065*_ease(age/.5))
        if bar:
            scene[:bar] = 0
            scene[-bar:] = 0
        if age >= TITLE_AT:
            entrance = _ease((age-TITLE_AT)/.18)
            top, bottom = round(self.height*.35), round(self.height*.67)
            cv2.convertScaleAbs(scene[top:bottom], alpha=1-.34*entrance, dst=scene[top:bottom])
            self._draw_sprite(scene, self.title, .48, entrance, 1+.09*(1-_ease((age-TITLE_AT)/.28)))
            self._draw_sprite(scene, self.captions[self.reason_index], .625,
                              _ease((age-TITLE_AT-.16)/.3))
        return scene

    def render(self, frame, now):
        if not self.running:
            self._record(frame, now)
            return frame.copy()
        age = max(0., now-self.started)
        if age >= TOTAL_SECONDS:
            self.reset()
            self._record(frame, now)
            return frame.copy()
        if age < DEATH_SECONDS:
            return self._death_scene(frame, age)
        progress = (age-DEATH_SECONDS)/RESPAWN_SECONDS
        if progress < .32:
            scene = self._death_scene(frame, DEATH_SECONDS)
            scene = cv2.convertScaleAbs(scene, alpha=1-_ease(progress/.32))
        else:
            reveal = _ease((progress-.32)/.52)
            scene = cv2.convertScaleAbs(frame, alpha=reveal)
            strength = max(0., 1-(progress-.32)/.45)
            if strength > 0:
                for index in range(5):
                    y = round(((progress*1.3+index*.193)%1)*self.height)
                    band = scene[y:min(self.height, y+max(1, self.height//48))]
                    if band.size:
                        band[:] = np.roll(band, round(math.sin(index+progress*32)*self.width*.035*strength), axis=1)
                y = min(self.height-1, round((progress-.32)/.68*self.height))
                cv2.line(scene, (0, y), (self.width-1, y),
                         (round(130*strength), round(180*strength), round(100*strength)), 1)
        opacity = _ease((progress-.30)/.12)*_ease((1-progress)/.28)
        self._draw_sprite(scene, self.respawn, .50, opacity)
        return np.ascontiguousarray(scene)
