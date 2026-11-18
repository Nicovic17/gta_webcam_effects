"""Code-drawn tropical city, moving police pursuit and cached text sprites."""
import math

import cv2
import numpy as np

from wasted_effect import _text_sprite


def ease(value):
    value = max(0., min(1., value))
    return value*value*(3-2*value)


def label(text, width, size, color=(245, 245, 245), headline=False):
    return _text_sprite(text, width, max(9, round(size)), color=color, headline=headline)


def over(scene, sprite, x, y, opacity=1., scale=1.):
    if opacity <= .001:
        return
    color, alpha = sprite
    if scale != 1:
        size = (max(1, round(color.shape[1]*scale)), max(1, round(color.shape[0]*scale)))
        color, alpha = cv2.resize(color, size), cv2.resize(alpha, size)
    x, y = round(x), round(y)
    h, w = color.shape[:2]
    sh, sw = scene.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(sw, x+w), min(sh, y+h)
    if x1 >= x2 or y1 >= y2:
        return
    weight = np.ascontiguousarray(alpha[y1-y:y2-y, x1-x:x2-x]*np.float32(opacity))
    roi = scene[y1:y2, x1:x2]
    cv2.blendLinear(roi, color[y1-y:y2-y, x1-x:x2-x], 1-weight, weight, dst=roi)


class TropicalCity:
    """Prepaint architecture once; animate only lights, vehicles and scenery."""
    def __init__(self, width, height, night=False):
        self.width, self.height, self.night = width, height, night
        self.sw = min(480, width)
        self.sh = max(1, round(height*self.sw/width))
        w, h = self.sw, self.sh
        self.base = np.zeros((h, w, 3), np.uint8)
        horizon = round(h*.64)
        top, middle, bottom = ((42, 12, 21), (78, 26, 70), (22, 18, 35)) if night else (
            (133, 37, 90), (169, 91, 245), (129, 153, 250))
        for y in range(h):
            t = y/max(horizon, 1)
            color = np.array(top)*(1-min(t, 1))+np.array(middle)*min(t, 1)
            if y > horizon:
                color = np.array(bottom)
            self.base[y] = color.astype(np.uint8)
        if not night:
            sun = np.zeros_like(self.base)
            center = (round(w*.72), round(h*.35))
            cv2.circle(sun, center, round(h*.22), (93, 204, 255), -1, cv2.LINE_AA)
            for y in range(center[1], min(horizon, center[1]+round(h*.23)), max(2, h//45)):
                sun[y:y+max(1, h//130)] = 0
            visible = sun.any(axis=2)
            self.base[visible] = sun[visible]
        rng = np.random.default_rng(606)
        # Distant skyline, then art-deco buildings framing a central avenue.
        for x in range(0, w, max(8, w//42)):
            bw = int(rng.integers(max(4, w//70), max(5, w//24)))
            bh = int(rng.integers(max(8, h//13), max(9, h//4)))
            cv2.rectangle(self.base, (x, horizon-bh), (x+bw, horizon), (52, 25, 58), -1)
        road = np.int32(((w*.43, h*.63), (w*.57, h*.63), (w*.92, h), (w*.08, h)))
        cv2.fillConvexPoly(self.base, road, (32, 24, 42))
        for side in (0, 1):
            for i in range(4):
                bw = round(w*(.13-i*.018))
                bh = round(h*(.60-i*.10))
                x = round(w*(.00+i*.095)) if side == 0 else round(w*(.87-i*.095))
                y = horizon-bh
                body = (55+i*6, 39+i*4, 65+i*5) if night else (126+i*10, 98+i*8, 176+i*9)
                cv2.rectangle(self.base, (x, y), (x+bw, horizon), body, -1)
                neon = (220, 190, 65) if (side+i)%2 else (170, 64, 243)
                cv2.rectangle(self.base, (x, y), (x+bw, horizon), neon, 1, cv2.LINE_AA)
                for floor in range(y+9, horizon, max(7, h//28)):
                    cv2.line(self.base, (x+2, floor), (x+bw-2, floor), neon, 1)
                    for wx in range(x+4, x+bw-3, max(4, w//70)):
                        cv2.rectangle(self.base, (wx, floor+2), (wx+2, floor+5), (102, 166, 198), -1)
        self.glow = cv2.GaussianBlur(self.base, (0, 0), 3)
        self.y, self.x = np.mgrid[:h, :w].astype(np.float32)

    def palm(self, scene, x, base_y, length, sway, color):
        top = (round(x+length*.13*math.sin(sway)), round(base_y-length))
        cv2.line(scene, (round(x), round(base_y)), top, color, max(2, round(length*.04)), cv2.LINE_AA)
        for direction in (-1, 1):
            for slope in (-.4, .05, .45):
                points = []
                for t in np.linspace(0, 1, 9):
                    px = top[0]+direction*length*.44*t
                    py = top[1]+length*(slope*t+.33*t*t)
                    points.append((round(px), round(py)))
                cv2.polylines(scene, [np.int32(points)], False, color, max(1, round(length*.024)), cv2.LINE_AA)

    def render(self, age, level=1):
        w, h = self.sw, self.sh
        scene = cv2.addWeighted(self.base, .94, self.glow, .10, 0)
        # Depth-scaled dashed lane markers rush towards the camera.
        for i in range(9):
            depth = (i/9+age*(.12 if self.night else .02))%1
            py = h*(.65+.35*depth*depth)
            half = 1+depth*2
            cv2.line(scene, (round(w*.5-half), round(py)),
                     (round(w*.5+half), min(h-1, round(py+depth*depth*h*.08))),
                     (95, 144, 169), max(1, round(depth*3)), cv2.LINE_AA)
        if self.night:
            # Helicopter and a soft searchlight sweep behind the subject.
            hx = w*(.5+.38*math.sin(age*.24))
            hy = h*(.16+.035*math.cos(age*.34))
            aim = w*(.5+.38*math.sin(age*.61))
            light = scene.copy()
            cone = np.int32(((hx, hy+3), (aim-w*.15, h*.89), (aim+w*.15, h*.89)))
            cv2.fillConvexPoly(light, cone, (171, 154, 120), cv2.LINE_AA)
            cv2.addWeighted(scene, .87, light, .13, 0, dst=scene)
            cv2.ellipse(scene, (round(hx), round(hy)), (max(3, w//55), max(2, h//80)), 0, 0, 360, (14, 12, 20), -1)
            cv2.line(scene, (round(hx-w*.027), round(hy-5)), (round(hx+w*.027), round(hy-5)), (100, 95, 110), 1)
            for i in range(2+level):
                depth = (i/(2+level)+age*(.07+level*.008))%1
                x = w*(.5+(-1 if i%2 else 1)*(.065+.27*depth))
                y = h*(.64+.34*depth*depth)
                size = max(3, round(w*(.012+.050*depth)))
                cv2.rectangle(scene, (round(x-size), round(y-size*.45)),
                              (round(x+size), round(y+size*.5)), (26, 22, 30), -1)
                cv2.rectangle(scene, (round(x-size*.7), round(y-size*.7)),
                              (round(x+size*.7), round(y-size*.15)), (70, 74, 93), -1)
                pulse = .55+.45*math.sin(age*5+i)**2
                cv2.circle(scene, (round(x-size*.35), round(y-size*.55)), max(1, size//6), (30, 36, round(250*pulse)), -1)
                cv2.circle(scene, (round(x+size*.35), round(y-size*.55)), max(1, size//6), (round(250*pulse), 76, 25), -1)
                for offset in (-.75, .75):
                    cv2.line(scene, (round(x+size*offset), round(y)),
                             (round(x+size*offset*1.6), min(h-1, round(y+size*1.4))), (90, 117, 140), 1)
        for x, length in ((w*.10, h*.72), (w*.90, h*.82), (w*.25, h*.48), (w*.77, h*.45)):
            self.palm(scene, x, h*.79, length, age*.5+x, (28, 14, 35))
        return cv2.resize(scene, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
