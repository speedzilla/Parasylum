"""
Watching Eye
------------
Fullscreen realistic eye that tracks the nearest red LED seen by the
built-in webcam in a dark room. Nearest = largest red blob.

Runs offline. ESC or Q quits. No config needed; tunables below.
"""

import sys
import os
import time
import math
import random
import threading

import numpy as np
import cv2
import pygame

# ----------------------------- Tunables -------------------------------------

CAM_INDEX = 0            # built-in webcam is almost always 0
CAM_W, CAM_H = 640, 480

# Red LED detection. Two hue bands because red wraps around 0 in HSV.
HSV_LOWER_1 = (0,   90, 160)
HSV_UPPER_1 = (12, 255, 255)
HSV_LOWER_2 = (168, 90, 160)
HSV_UPPER_2 = (180, 255, 255)
MIN_BLOB_AREA = 8        # px; below this = noise
GAZE_SMOOTHING = 0.055   # 0..1, lower = smoother/slower eye
LOST_TIMEOUT = 1.5       # s without a blob before eye wanders idly

# Eye look
IRIS_COLOR = (70, 110, 140)     # steel blue; try (110, 80, 50) for brown
BLINK_MIN_GAP, BLINK_MAX_GAP = 5.0, 14.0  # s between random blinks; long stares
BLINK_DURATION = 0.30            # s for a full blink, slightly languid
DILATION_PERIOD = 11.0           # s, slow random pupil size drift
SACCADE_MIN_GAP, SACCADE_MAX_GAP = 0.6, 3.5  # s between micro-twitches
SACCADE_SIZE = 0.11              # twitch amplitude in gaze units

NUM_EYES = 22                    # eyes on screen, varied sizes, non-overlapping
EYE_SIZE_RANGE = (0.04, 0.19)    # radius as fraction of screen's smaller dimension

# ----------------------------- Tracker --------------------------------------

class Tracker(threading.Thread):
    """Reads webcam frames, finds the largest red blob, publishes its
    normalized position (-1..1 both axes, 0,0 = center) and a seen-timestamp.
    The VideoCapture must be created on the main thread (macOS requires the
    camera permission request to happen there) and passed in."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.pos = (0.0, 0.0)
        self.last_seen = 0.0
        self.ok = cap is not None and cap.isOpened()
        self.debug_frame = None   # BGR frame with blob annotation, for debug view
        self._halt = threading.Event()

    def stop(self):
        self._halt.set()

    def run(self):
        cap = self.cap
        if cap is None:
            return
        while not self._halt.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, HSV_LOWER_1, HSV_UPPER_1) | \
                   cv2.inRange(hsv, HSV_LOWER_2, HSV_UPPER_2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            found = None
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) >= MIN_BLOB_AREA:
                    m = cv2.moments(c)
                    cx = m["m10"] / m["m00"]
                    cy = m["m01"] / m["m00"]
                    h, w = mask.shape
                    # Mirror x so the eye looks toward the person, not away.
                    nx = -((cx / w) * 2 - 1)
                    ny = (cy / h) * 2 - 1
                    self.pos = (nx, ny)
                    self.last_seen = time.time()
                    found = (int(cx), int(cy), int(math.sqrt(cv2.contourArea(c) / math.pi)) + 6)
            dbg = frame.copy()
            if found:
                cv2.circle(dbg, (found[0], found[1]), found[2], (0, 255, 0), 2)
                cv2.putText(dbg, "TRACKING", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(dbg, "NO LED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.debug_frame = dbg

# ----------------------------- Eye renderer ---------------------------------

def lens_points(w, h, p, steps=90):
    """Almond (lens) outline, pointed left/right. p controls pointiness."""
    cx, cy = w / 2, h / 2
    hw, hh = w / 2 - 2, h / 2 - 2
    top, bot = [], []
    for i in range(steps + 1):
        x = -hw + 2 * hw * i / steps
        y = hh * (max(0.0, 1 - (abs(x) / hw) ** 2)) ** p
        top.append((cx + x, cy - y))
        bot.append((cx + x, cy + y))
    return top + bot[::-1], hw, hh


class Eye:
    """Monochrome almond eye. Body texture pre-rendered once; pupil, blink
    lids, and rotation composited per frame."""

    def __init__(self, screen, cx, cy, width, seed=0):
        self.screen = screen
        self.cx, self.cy = cx, cy
        self.rng = random.Random(seed)
        self.w = width
        self.h = int(width * self.rng.uniform(0.52, 0.62))
        self.angle = self.rng.uniform(-16, 16)
        self.p = self.rng.uniform(0.55, 0.8)
        self.pupil_frac = self.rng.uniform(0.38, 0.52)

        self.gaze = [0.0, 0.0]
        self.idle_target = [0.0, 0.0]
        self.next_idle = 0.0

        self.next_blink = time.time() + self.rng.uniform(BLINK_MIN_GAP, BLINK_MAX_GAP)
        self.blink_start = None

        self.saccade = [0.0, 0.0]
        self.next_saccade = time.time() + self.rng.uniform(SACCADE_MIN_GAP, SACCADE_MAX_GAP)

        self.dil_seed = self.rng.random() * 100

        self.body = self._make_body()
        # cached lens mask at display resolution, for clipping pupil and lids
        self.mask = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        outline, _, _ = lens_points(self.w, self.h, self.p)
        pygame.draw.polygon(self.mask, (255, 255, 255, 255), outline)

    def _make_body(self):
        S = 2  # supersample, downscale = cheap blur
        W, H = self.w * S, self.h * S
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        outline, hw, hh = lens_points(W, H, self.p)
        cx, cy = W / 2, H / 2
        n = 16
        for k in range(n):  # dark rim first, lighter and smaller toward center
            t = k / (n - 1)
            g = int(55 + 155 * (t ** 0.8))
            scale = 1 - 0.055 * k
            ring = [(cx + (px - cx) * scale, cy + (py - cy) * scale) for px, py in outline]
            pygame.draw.polygon(surf, (g, g, g), ring)
        grime = pygame.Surface((W, H), pygame.SRCALPHA)
        for _ in range(220):
            x = self.rng.randint(0, W - 1)
            y = self.rng.randint(0, H - 1)
            dark = self.rng.random() < 0.6
            g = self.rng.randint(30, 90) if dark else self.rng.randint(180, 230)
            pygame.draw.circle(grime, (g, g, g, self.rng.randint(14, 42)),
                               (x, y), self.rng.randint(2, 12))
        surf.blit(grime, (0, 0))

        # bloodshot veins: jagged dark branches creeping in from the corners
        # and rim toward the pupil zone; monochrome-dark red so they read as
        # veins without breaking the grayscale look
        veins = pygame.Surface((W, H), pygame.SRCALPHA)
        for _ in range(18):
            # bias start points toward the pointed corners like real bloodshot
            corner = self.rng.random() < 0.55
            if corner:
                sx = 0 if self.rng.random() < 0.5 else W
                a = 0.0 if sx == 0 else math.pi
                a += self.rng.uniform(-0.5, 0.5)
                px, py = sx, cy + self.rng.uniform(-0.25, 0.25) * H
            else:
                a = self.rng.uniform(0, 2 * math.pi)
                px = cx + math.cos(a) * hw * 0.97
                py = cy + math.sin(a) * hh * 0.97
                a += math.pi  # head inward
            shade = self.rng.randint(70, 120)
            col = (shade, int(shade * 0.55), int(shade * 0.55),
                   self.rng.randint(90, 150))
            width = max(2, int(H * 0.012))
            for _seg in range(self.rng.randint(4, 8)):
                a += self.rng.uniform(-0.55, 0.55)
                step = self.rng.uniform(0.05, 0.11) * W
                nx, ny = px + math.cos(a) * step, py + math.sin(a) * step
                # stop before invading the pupil zone
                if abs(nx - cx) < W * 0.16 and abs(ny - cy) < H * 0.2:
                    break
                pygame.draw.line(veins, col, (px, py), (nx, ny), width)
                # occasional fork
                if self.rng.random() < 0.3:
                    fa = a + self.rng.uniform(-1.0, 1.0)
                    fx, fy = nx + math.cos(fa) * step * 0.6, ny + math.sin(fa) * step * 0.6
                    pygame.draw.line(veins, col, (nx, ny), (fx, fy), max(1, width - 1))
                px, py = nx, ny
                width = max(1, width - (1 if self.rng.random() < 0.4 else 0))
        surf.blit(veins, (0, 0))
        mask = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.polygon(mask, (255, 255, 255, 255), outline)
        surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        return pygame.transform.smoothscale(surf, (self.w, self.h))

    def _blink_amount(self, now):
        if self.blink_start is None:
            if now >= self.next_blink:
                self.blink_start = now
            return 0.0
        t = (now - self.blink_start) / BLINK_DURATION
        if t >= 1.0:
            self.blink_start = None
            self.next_blink = now + self.rng.uniform(BLINK_MIN_GAP, BLINK_MAX_GAP)
            return 0.0
        return math.sin(t * math.pi)

    def update(self, target, tracking, now):
        if not tracking:
            if now >= self.next_idle:
                self.idle_target = [self.rng.uniform(-0.5, 0.5), self.rng.uniform(-0.3, 0.3)]
                self.next_idle = now + self.rng.uniform(2.0, 6.0)
            target = self.idle_target
        if now >= self.next_saccade:
            self.saccade = [self.rng.uniform(-SACCADE_SIZE, SACCADE_SIZE),
                            self.rng.uniform(-SACCADE_SIZE, SACCADE_SIZE)]
            self.next_saccade = now + self.rng.uniform(SACCADE_MIN_GAP, SACCADE_MAX_GAP)
        for i in (0, 1):
            self.gaze[i] += (target[i] + self.saccade[i] - self.gaze[i]) * GAZE_SMOOTHING
            self.saccade[i] *= 0.94

    def draw(self, now):
        frame = self.body.copy()
        hw, hh = self.w / 2 - 2, self.h / 2 - 2
        cx, cy = self.w / 2, self.h / 2

        # pupil with slow dilation drift
        d = 0.5 + 0.5 * math.sin(now * 2 * math.pi / DILATION_PERIOD + self.dil_seed)
        pr = int(hh * self.pupil_frac * (0.86 + 0.14 * d))
        px = cx + self.gaze[0] * (hw - pr) * 0.5
        py = cy + self.gaze[1] * (hh - pr) * 0.5
        layer = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        pygame.draw.circle(layer, (120, 120, 120), (px, py), int(pr * 1.16))
        pygame.draw.circle(layer, (60, 60, 60), (px, py), int(pr * 1.16), max(2, pr // 9))
        pygame.draw.circle(layer, (10, 10, 12), (px, py), pr)
        pygame.draw.circle(layer, (240, 240, 240),
                           (px - pr * 0.38, py - pr * 0.38), max(2, pr // 6))
        pygame.draw.circle(layer, (170, 170, 170),
                           (px + pr * 0.28, py + pr * 0.22), max(1, pr // 11))

        # blink lids: black covers from top and bottom
        b = self._blink_amount(now)
        if b > 0:
            lid = int(self.h / 2 * b) + 1
            pygame.draw.rect(layer, (0, 0, 0), (0, 0, self.w, lid))
            pygame.draw.rect(layer, (0, 0, 0), (0, self.h - lid, self.w, lid))

        layer.blit(self.mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        frame.blit(layer, (0, 0))

        rot = pygame.transform.rotozoom(frame, self.angle, 1.0)
        self.screen.blit(rot, rot.get_rect(center=(self.cx, self.cy)))


# ----------------------------- Layout ---------------------------------------

def layout_eyes(screen):
    """Place NUM_EYES non-overlapping almond eyes of varied width via
    rejection sampling; shrink and retry if the screen is too crowded."""
    w, h = screen.get_size()
    base = min(w, h)
    rng = random.Random()   # different arrangement every launch
    placed = []
    attempts = 0
    scale = 1.0
    while len(placed) < NUM_EYES:
        ew = int(base * rng.uniform(*EYE_SIZE_RANGE) * 2 * scale)  # width
        eh = int(ew * 0.62)
        x = rng.randint(ew // 2, max(ew // 2 + 1, w - ew // 2))
        y = rng.randint(eh // 2, max(eh // 2 + 1, h - eh // 2))
        pad = int(base * 0.02)
        if all((x - px) ** 2 + (y - py) ** 2 >= ((ew + pw) / 2 * 0.85 + pad) ** 2
               for px, py, pw in placed):
            placed.append((x, y, ew))
        attempts += 1
        if attempts > 4000:
            scale *= 0.85
            placed = []
            attempts = 0
    return [Eye(screen, x, y, ew, seed=i * 977 + ew) for i, (x, y, ew) in enumerate(placed)]


# ----------------------------- Main -----------------------------------------

def make_grunge(w, h, seed=11):
    """Static film-grime overlay: vignette, dust, scratches, smudges.
    Pre-rendered once, blitted every frame."""
    rng = random.Random(seed)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)

    # vignette: smooth per-pixel radial darkening (numpy, no banding)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w / 2) / (w / 2)
    ny = (yy - h / 2) / (h / 2)
    dist = np.sqrt(nx * nx + ny * ny) / math.sqrt(2)
    alpha = np.clip((dist - 0.55) / 0.45, 0, 1) ** 1.8 * 150
    vig = pygame.Surface((w, h), pygame.SRCALPHA)
    px = pygame.surfarray.pixels_alpha(vig)
    px[:, :] = alpha.T.astype(np.uint8)
    del px
    surf.blit(vig, (0, 0))

    # dust specks
    for _ in range(w * h // 4500):
        x = rng.randint(0, w - 1); y = rng.randint(0, h - 1)
        g = rng.randint(120, 220)
        surf.set_at((x, y), (g, g, g, rng.randint(20, 70)))

    # scratches: long faint vertical-ish lines
    for _ in range(10):
        x = rng.randint(0, w - 1)
        drift = rng.uniform(-0.06, 0.06)
        g = rng.randint(140, 210)
        a = rng.randint(10, 26)
        y0 = rng.randint(0, h // 3)
        y1 = rng.randint(2 * h // 3, h - 1)
        pts = [(x + (y - y0) * drift + rng.uniform(-1, 1), y)
               for y in range(y0, y1, 6)]
        if len(pts) > 1:
            pygame.draw.lines(surf, (g, g, g, a), False, pts, 1)

    # smudges: big soft dark blotches
    for _ in range(18):
        x = rng.randint(0, w - 1); y = rng.randint(0, h - 1)
        rad = rng.randint(int(h * 0.03), int(h * 0.12))
        blot = pygame.Surface((rad * 2, rad * 2), pygame.SRCALPHA)
        for rr in range(rad, 0, -2):
            a = int(14 * (1 - rr / rad) + 2)
            pygame.draw.circle(blot, (10, 10, 10, a), (rad, rad), rr)
        surf.blit(blot, (x - rad, y - rad))

    return surf


def open_camera(index=CAM_INDEX):
    """Must run on the main thread: macOS ties the camera permission prompt
    to the main run loop."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    # Fight auto-exposure: in a dark room we want the LED as a tight dot,
    # not a blown-out frame. Not all drivers honor these; harmless if ignored.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    # Force one read on the main thread so the permission dialog fires here.
    cap.read()
    return cap


def main():
    cap = open_camera()

    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    pygame.init()
    pygame.mouse.set_visible(False)
    # Borderless window at screen size instead of exclusive fullscreen:
    # looks identical, but Cmd+Tab and system dialogs keep working on macOS.
    # Press F for true fullscreen if the window doesn't cover the screen.
    info = pygame.display.Info()
    screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
    clock = pygame.time.Clock()

    tracker = Tracker(cap)
    tracker.start()

    eyes = layout_eyes(screen)
    sw, sh = screen.get_size()
    grunge = make_grunge(sw, sh)
    cam_index = CAM_INDEX
    fullscreen = False
    debug = False
    follow = True    # M toggles: True = track the LED, False = static stare
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                if ev.key == pygame.K_d:
                    debug = not debug
                if ev.key == pygame.K_m:
                    follow = not follow
                if ev.key == pygame.K_f:
                    fullscreen = not fullscreen
                    flags = pygame.NOFRAME if fullscreen else pygame.FULLSCREEN
                    screen = pygame.display.set_mode((sw, sh), flags)
                    for e in eyes:
                        e.screen = screen
                if ev.key == pygame.K_c:
                    # cycle to next camera (phone vs built-in); wraps at 4
                    tracker.stop()
                    tracker.join(timeout=1.0)
                    cap.release()
                    for _try in range(4):
                        cam_index = (cam_index + 1) % 4
                        cap = open_camera(cam_index)
                        if cap.isOpened() and cap.read()[0]:
                            break
                        cap.release()
                    tracker = Tracker(cap)
                    tracker.start()

        now = time.time()
        tracking = follow and (now - tracker.last_seen) < LOST_TIMEOUT
        # Project the LED onto a screen point, then aim each eye from its own
        # position toward that point so the eyes converge instead of moving
        # in lockstep.
        tx = (tracker.pos[0] * 0.5 + 0.5) * sw
        ty = (tracker.pos[1] * 0.5 + 0.5) * sh

        screen.fill((0, 0, 0))
        for eye in eyes:
            if follow:
                dx = (tx - eye.cx) / (sw * 0.5)
                dy = (ty - eye.cy) / (sh * 0.5)
                mag = math.hypot(dx, dy)
                if mag > 1.0:
                    dx, dy = dx / mag, dy / mag
                eye.update([dx, dy], tracking, now)
            else:
                # static stare: dead ahead, twitches and blinks continue
                eye.update([0.0, 0.0], True, now)
            eye.draw(now)

        screen.blit(grunge, (0, 0))

        if debug and tracker.debug_frame is not None:
            f = cv2.cvtColor(tracker.debug_frame, cv2.COLOR_BGR2RGB)
            f = np.rot90(np.fliplr(f))
            surf = pygame.surfarray.make_surface(f)
            surf = pygame.transform.smoothscale(surf, (320, 240))
            screen.blit(surf, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    tracker.stop()
    tracker.join(timeout=1.0)
    if cap is not None:
        cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
