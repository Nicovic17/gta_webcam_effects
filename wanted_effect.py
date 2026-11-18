"""Asset-based wanted HUD over the real webcam, with an in-memory mugshot."""
from pathlib import Path
import math

import cv2
import numpy as np

from neon_scene import ease, label, over

ASSET_DIR = Path(__file__).resolve().parent/'assets'
STAR_SECONDS = 1.6
FULL_STARS_SECONDS = 5*STAR_SECONDS
STAR_BLINK_SECONDS = 2.0
FULL_BLINK_SECONDS = 3.0
BLINK_PERIOD = .55
SCAN_SECONDS = .95
CARD_ZOOM_SECONDS = 1.8
CARD_ZOOM_AMOUNT = .08


def load_sprite(path, width, height):
    source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if source is None or source.ndim != 3 or source.shape[2] != 4:
        raise RuntimeError(f'Asset WANTED PNG trasparente mancante: {path}')
    visible = (source[:, :, 3] > 16).astype(np.uint8)
    if not visible.any():
        raise RuntimeError(f'Asset WANTED vuoto: {path}')
    x, y, w, h = cv2.boundingRect(visible)
    source = source[y:y+h, x:x+w]
    alpha = source[:, :, 3:4].astype(np.float32)/255
    premultiplied = np.concatenate((source[:, :, :3].astype(np.float32)*alpha, alpha), axis=2)
    ratio = min(width/w, height/h)
    size = (max(1, round(w*ratio)), max(1, round(h*ratio)))
    resized = cv2.resize(premultiplied, size, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 4), np.float32)
    left, top = (width-size[0])//2, (height-size[1])//2
    canvas[top:top+size[1], left:left+size[0]] = resized
    alpha = canvas[:, :, 3]
    color = np.clip(canvas[:, :, :3]/np.maximum(alpha[:, :, None], 1e-6), 0, 255).astype(np.uint8)
    return color, np.ascontiguousarray(alpha)


class WantedEffect:
    def __init__(self, width, height, asset_dir=ASSET_DIR):
        self.width, self.height = width, height
        self.margin = max(3, round(width*.028))
        star_size = max(8, round(min(width*.037, height*.070)))
        gap = max(2, round(star_size*.22))
        self.star_size, self.star_gap = star_size, gap
        self.stars_x = width-self.margin-5*star_size-4*gap
        self.stars_y = max(3, round(height*.035))
        white = load_sprite(Path(asset_dir)/'whitestar.png', star_size, star_size)
        yellow = load_sprite(Path(asset_dir)/'yellowstar.png', star_size, star_size)
        # Switch directly between the supplied assets, with no colour blending.
        self.star_frames = (white, yellow)
        map_size = max(18, round(min(width*.23, height*.30)))
        self.minimap = load_sprite(Path(asset_dir)/'map.png', map_size, map_size)
        self.map_x, self.map_y = self.margin, height-max(3, round(height*.035))-map_size
        self.hud_bounds = ((self.stars_x, self.stars_y, width-self.margin, self.stars_y+star_size),
                           (self.map_x, self.map_y, self.map_x+map_size, self.map_y+map_size))
        self.panel_size = max(16, round(min(width*.33, height*.52)))
        self.title = label('WANTED', self.panel_size*.96, width*.065, (244, 75, 78), True)
        self.scanning = label('SCANSIONE IN CORSO', width*.44, width*.030, (139, 247, 192))
        self._prepare_lights()
        self.reset()

    @property
    def running(self):
        return self.started is not None

    def reset(self):
        self.started = None
        self.scan_snapshot = None
        self.portrait = None
        self.face_box = (.35, .17, .65, .65)

    def trigger(self, now, face_box=None):
        if self.running:
            return False
        self.started = now
        if face_box is not None:
            self.face_box = face_box
        return True

    def mirror(self):
        x1, y1, x2, y2 = self.face_box
        self.face_box = (1-x2, y1, 1-x1, y2)
        if self.scan_snapshot is not None:
            self.scan_snapshot = cv2.flip(self.scan_snapshot, 1)
        if self.portrait is not None:
            self.portrait = cv2.flip(self.portrait, 1)

    @staticmethod
    def star_fill(age, index):
        return float(age+1e-9 >= (index+1)*STAR_SECONDS)

    @staticmethod
    def star_opacity(age, index):
        if age+1e-9 >= FULL_STARS_SECONDS:
            # At level five, all stars share one phase for a bounded burst.
            elapsed = max(0., age-FULL_STARS_SECONDS)
            if elapsed+1e-9 >= FULL_BLINK_SECONDS:
                return 1.
        else:
            elapsed = age-(index+1)*STAR_SECONDS
            if elapsed < 0 or elapsed+1e-9 >= STAR_BLINK_SECONDS:
                return 1.
        return 1. if elapsed%BLINK_PERIOD < BLINK_PERIOD*.55 else .20

    def star_count(self, age):
        return min(5, max(0, int((age+1e-9)/STAR_SECONDS)))

    def _hud(self, scene, age=None):
        over(scene, self.minimap, self.map_x, self.map_y)
        for index in range(5):
            amount = self.star_fill(age, index) if age is not None else 0
            sprite = self.star_frames[int(amount)]
            blink = self.star_opacity(age, index) if age is not None else 1.
            over(scene, sprite, self.stars_x+index*(self.star_size+self.star_gap), self.stars_y, blink)

    def _capture(self, frame):
        self.scan_snapshot = frame.copy()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.face_box
        if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = (.35, .17, .65, .65)
        side = min(min(w, h), max(8, round(max((x2-x1)*w, (y2-y1)*h)*1.12)))
        left = int(np.clip(round((x1+x2)*w/2-side/2), 0, w-side))
        top = int(np.clip(round((y1+y2)*h/2-side/2), 0, h-side))
        self.portrait = frame[top:top+side, left:left+side].copy()

    def _scan(self, scene, progress):
        x1, y1, x2, y2 = self.face_box
        fw = max(.12*self.width, (x2-x1)*self.width)
        cx = (x1+x2)*self.width/2
        left = int(np.clip(cx-fw*.95, 0, self.width-1))
        right = int(np.clip(cx+fw*.95, left+1, self.width))
        top = int(np.clip(y1*self.height-self.height*.04, 0, self.height-2))
        bottom = max(top+1, round(self.height*.95))
        color = (125, 238, 169)
        corner = max(5, self.width//50)
        for x, y, dx, dy in ((left, top, 1, 1), (right-1, top, -1, 1),
                              (left, bottom, 1, -1), (right-1, bottom, -1, -1)):
            cv2.line(scene, (x, y), (x+dx*corner, y), color, 1, cv2.LINE_AA)
            cv2.line(scene, (x, y), (x, y+dy*corner), color, 1, cv2.LINE_AA)
        scan_y = round(top+(bottom-top)*ease(progress))
        band = scene[max(top, scan_y-3):min(bottom, scan_y+4), left:right]
        if band.size:
            cv2.add(band, (9, 35, 12, 0), dst=band)
        cv2.line(scene, (left, scan_y), (right-1, scan_y), color, 2, cv2.LINE_AA)
        over(scene, self.scanning, (self.width-self.scanning[0].shape[1])/2,
             self.height*.13, .5+.5*math.sin(progress*math.pi))

    def _mugshot(self, scene, age):
        if self.portrait is None:
            return
        entrance = ease(age/.30)
        if entrance <= 0:
            return
        size = self.panel_size
        padding = max(3, round(self.width*.007))
        title_h = self.title[0].shape[0]
        card_w, card_h = size+padding*2, size+padding*4+title_h
        base_x = self.width-self.margin-size-padding
        base_y = round(self.height*.22)-padding
        # Assemble the entire card once per frame before applying one shared
        # transform: photo, border, backing and caption breathe as one object.
        card = np.zeros((card_h, card_w, 3), np.uint8)
        x1, y1 = max(0, base_x), max(0, base_y)
        x2, y2 = min(self.width, base_x+card_w), min(self.height, base_y+card_h)
        if x1 < x2 and y1 < y2:
            card[y1-base_y:y2-base_y, x1-base_x:x2-base_x] = cv2.convertScaleAbs(
                scene[y1:y2, x1:x2], alpha=.14)
        source = self.portrait
        side = max(1, round(source.shape[0]/1.16))
        start = (source.shape[0]-side)//2
        photo = cv2.resize(source[start:start+side, start:start+side], (size, size), interpolation=cv2.INTER_LINEAR)
        card[padding:padding+size, padding:padding+size] = photo
        blink = .12+.88*(.5+.5*math.sin(age*2*math.pi))
        edge = (round(100+130*blink), 75, round(230-130*blink))
        cv2.rectangle(card, (padding-1, padding-1), (padding+size, padding+size),
                      edge, max(1, self.width//400), cv2.LINE_AA)
        over(card, self.title, (card_w-self.title[0].shape[1])/2,
             padding+size+padding*1.5, blink)
        pulse = 1-CARD_ZOOM_AMOUNT*math.cos(age*2*math.pi/CARD_ZOOM_SECONDS)
        scale = (.88+.12*entrance)*pulse
        # Keep the card centred on the same point through the whole cycle.
        x = base_x+(card_w-card_w*scale)/2
        y = base_y+(card_h-card_h*scale)/2+(1-entrance)*self.height*.025
        over(scene, (card, np.ones((card_h, card_w), np.float32)), x, y, entrance, scale)

    def _prepare_lights(self):
        self.lw = min(320, self.width)
        self.lh = max(1, round(self.height*self.lw/self.width))
        y, x = np.mgrid[:self.lh, :self.lw].astype(np.float32)
        x, y = x/self.lw, y/self.lh
        self.light_fields = [np.exp(-((x-cx)/sx)**2-((y-cy)/sy)**2).astype(np.float32)
                             for cx, cy, sx, sy in ((0, .16, .20, .30), (1, .30, .20, .34),
                                 (.20, 0, .28, .16), (.78, 1, .30, .18),
                                 (0, .82, .20, .30), (1, .84, .20, .30))]

    def _lights(self, scene, age):
        phase = age/1.15
        index = int(phase)%len(self.light_fields)
        mix = ease(phase-math.floor(phase))
        red = self.light_fields[index]*(1-mix)+self.light_fields[(index+1)%6]*mix
        blue = self.light_fields[(index+3)%6]*(1-mix)+self.light_fields[(index+4)%6]*mix
        strength = (.55+.45*math.sin(age*math.pi)**2)*ease(age/.6)
        glow = (red[:, :, None]*np.array((4, 8, 100), np.float32)
                +blue[:, :, None]*np.array((105, 20, 4), np.float32))*strength
        glow = cv2.resize(np.clip(glow, 0, 255).astype(np.uint8), (self.width, self.height))
        cv2.add(scene, glow, dst=scene)

    def render(self, frame, mask, now, face_box=None):
        # No synthetic background: the camera remains live throughout.
        scene = frame.copy()
        if face_box is not None:
            self.face_box = face_box
        if not self.running:
            self._hud(scene)
            return scene
        age = max(0., now-self.started)
        if age >= FULL_STARS_SECONDS:
            if self.portrait is None:
                self._capture(frame)
            scan_age = age-FULL_STARS_SECONDS
            if scan_age < SCAN_SECONDS:
                self._scan(scene, scan_age/SCAN_SECONDS)
            else:
                alert_age = scan_age-SCAN_SECONDS
                self._lights(scene, alert_age)
                self._mugshot(scene, alert_age)
        self._hud(scene, age)
        return np.ascontiguousarray(scene)
