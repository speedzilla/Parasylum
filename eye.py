"""
Watching Eye
------------
Fullscreen realistic eye that tracks the nearest red LED seen by the
built-in webcam in a dark room. Nearest = largest red blob.

Runs offline. ESC or Q quits. No config needed; tunables below.
"""

import sys
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
SACCADE_MIN_GAP, SACCADE_MAX_GAP = 3.0, 9.0  # s between micro-twitches
SACCADE_SIZE = 0.045             # twitch amplitude in gaze units

NUM_EYES = 7                     # eyes on screen, varied sizes, non-overlapping
EYE_SIZE_RANGE = (0.10, 0.30)    # radius as fraction of screen's smaller dimension

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

def radial_gradient_surface(radius, inner, outer):
    """Circle surface with a radial gradient, used for sclera shading."""
    size = radius * 2
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    for r in range(radius, 0, -1):
        t = r / radius
        col = [int(inner[i] * (1 - t) + outer[i] * t) for i in range(3)]
        pygame.draw.circle(surf, col, (radius, radius), r)
    return surf


class Eye:
    def __init__(self, screen, cx, cy, radius, seed=0):
        self.screen = screen
        self.cx, self.cy = cx, cy
        self.eye_rx = radius
        self.iris_r = int(self.eye_rx * 0.42)
        self.max_travel = self.eye_rx - self.iris_r - max(2, int(self.eye_rx * 0.06))
        self.rng = random.Random(seed)

        self.gaze = [0.0, 0.0]           # smoothed, -1..1
        self.idle_target = [0.0, 0.0]
        self.next_idle = 0.0

        self.next_blink = time.time() + self.rng.uniform(BLINK_MIN_GAP, BLINK_MAX_GAP)
        self.blink_start = None

        self.saccade = [0.0, 0.0]
        self.next_saccade = time.time() + self.rng.uniform(SACCADE_MIN_GAP, SACCADE_MAX_GAP)

        self.dil_seed = self.rng.random() * 100
        # Slightly grey, aged sclera reads creepier than bright white.
        self.sclera = radial_gradient_surface(self.eye_rx, (238, 232, 222), (120, 105, 100))

        # Pre-render iris fibres (random radial lines) once.
        self.iris_surf = self._make_iris()

        # Pre-render bloodshot veins to a surface (drawn per-eye each frame
        # otherwise; static per eye so bake them once).
        self.veins = self._make_veins()

    def _make_veins(self):
        size = self.eye_rx * 2
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        rng = random.Random(self.rng.randint(0, 9999))
        c = self.eye_rx
        for _ in range(26):
            a = rng.uniform(0, 2 * math.pi)
            r = self.eye_rx * rng.uniform(0.93, 0.99)
            px, py = c + r * math.cos(a), c + r * math.sin(a)
            col = (rng.randint(150, 195), rng.randint(45, 75), rng.randint(45, 70))
            for _seg in range(rng.randint(3, 6)):
                a += rng.uniform(-0.5, 0.5)
                r -= self.eye_rx * rng.uniform(0.03, 0.09)
                if r < self.eye_rx * 0.5:
                    break
                nx, ny = c + r * math.cos(a), c + r * math.sin(a)
                pygame.draw.aaline(surf, col, (px, py), (nx, ny))
                px, py = nx, ny
        return surf

    def _make_iris(self):
        r = self.iris_r
        surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        base = IRIS_COLOR
        for rr in range(r, 0, -1):
            t = rr / r
            col = [int(base[i] * (0.55 + 0.45 * (1 - t))) for i in range(3)]
            pygame.draw.circle(surf, col, (r, r), rr)
        rng = random.Random(7)
        for _ in range(160):
            a = rng.uniform(0, 2 * math.pi)
            r0 = rng.uniform(0.25, 0.5) * r
            r1 = rng.uniform(0.75, 0.98) * r
            shade = rng.randint(-40, 45)
            col = [max(0, min(255, base[i] + shade)) for i in range(3)]
            p0 = (r + r0 * math.cos(a), r + r0 * math.sin(a))
            p1 = (r + r1 * math.cos(a), r + r1 * math.sin(a))
            pygame.draw.aaline(surf, col, p0, p1)
        pygame.draw.circle(surf, (25, 25, 30), (r, r), r, width=max(2, r // 22))
        return surf

    def _blink_amount(self, now):
        """0 = open, 1 = fully closed."""
        if self.blink_start is None:
            if now >= self.next_blink:
                self.blink_start = now
            return 0.0
        t = (now - self.blink_start) / BLINK_DURATION
        if t >= 1.0:
            self.blink_start = None
            self.next_blink = now + self.rng.uniform(BLINK_MIN_GAP, BLINK_MAX_GAP)
            return 0.0
        return math.sin(t * math.pi)   # close then open

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
            self.saccade[i] *= 0.94   # twitch decays back to true gaze

    def draw(self, now):
        s = self.screen
        s.blit(self.sclera, (self.cx - self.eye_rx, self.cy - self.eye_rx))
        s.blit(self.veins, (self.cx - self.eye_rx, self.cy - self.eye_rx))

        gx = self.cx + self.gaze[0] * self.max_travel
        gy = self.cy + self.gaze[1] * self.max_travel

        s.blit(self.iris_surf, (gx - self.iris_r, gy - self.iris_r))

        # pupil with slow random dilation
        d = 0.5 + 0.5 * math.sin(now * 2 * math.pi / DILATION_PERIOD + self.dil_seed)
        pupil_r = int(self.iris_r * (0.30 + 0.18 * d))
        pygame.draw.circle(s, (10, 8, 10), (gx, gy), pupil_r)

        # specular highlight
        pygame.draw.circle(s, (255, 255, 255),
                           (gx - self.iris_r * 0.35, gy - self.iris_r * 0.35),
                           max(3, self.iris_r // 7))
        pygame.draw.circle(s, (255, 255, 255, 120),
                           (gx + self.iris_r * 0.2, gy + self.iris_r * 0.3),
                           max(2, self.iris_r // 14))

        # eyelids (blink) drawn as black covers from top and bottom
        b = self._blink_amount(now)
        if b > 0:
            lid = int(self.eye_rx * b)
            top = pygame.Rect(self.cx - self.eye_rx, self.cy - self.eye_rx,
                              self.eye_rx * 2, lid)
            bot = pygame.Rect(self.cx - self.eye_rx, self.cy + self.eye_rx - lid,
                              self.eye_rx * 2, lid)
            pygame.draw.rect(s, (0, 0, 0), top)
            pygame.draw.rect(s, (0, 0, 0), bot)


# ----------------------------- Layout ---------------------------------------

def layout_eyes(screen):
    """Place NUM_EYES non-overlapping circles of varied radius via rejection
    sampling. Deterministic-ish: retries shrink radii until everything fits."""
    w, h = screen.get_size()
    base = min(w, h)
    rng = random.Random()   # different arrangement every launch
    placed = []
    attempts = 0
    scale = 1.0
    while len(placed) < NUM_EYES:
        r = int(base * rng.uniform(*EYE_SIZE_RANGE) * scale)
        x = rng.randint(r, w - r)
        y = rng.randint(r, h - r)
        pad = int(base * 0.015)
        if all((x - px) ** 2 + (y - py) ** 2 >= (r + pr + pad) ** 2
               for px, py, pr in placed):
            placed.append((x, y, r))
        attempts += 1
        if attempts > 4000:          # too crowded, shrink and restart
            scale *= 0.85
            placed = []
            attempts = 0
    return [Eye(screen, x, y, r, seed=i * 977 + r) for i, (x, y, r) in enumerate(placed)]


# ----------------------------- Main -----------------------------------------

def open_camera():
    """Must run on the main thread: macOS ties the camera permission prompt
    to the main run loop."""
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
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

    pygame.init()
    pygame.mouse.set_visible(False)
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    clock = pygame.time.Clock()

    tracker = Tracker(cap)
    tracker.start()

    eyes = layout_eyes(screen)
    sw, sh = screen.get_size()
    debug = False
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

        now = time.time()
        tracking = (now - tracker.last_seen) < LOST_TIMEOUT
        # Project the LED onto a screen point, then aim each eye from its own
        # position toward that point so the eyes converge instead of moving
        # in lockstep.
        tx = (tracker.pos[0] * 0.5 + 0.5) * sw
        ty = (tracker.pos[1] * 0.5 + 0.5) * sh

        screen.fill((0, 0, 0))
        for eye in eyes:
            dx = (tx - eye.cx) / (sw * 0.5)
            dy = (ty - eye.cy) / (sh * 0.5)
            mag = math.hypot(dx, dy)
            if mag > 1.0:
                dx, dy = dx / mag, dy / mag
            eye.update([dx, dy], tracking, now)
            eye.draw(now)

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
