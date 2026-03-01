"""
demo.py — SafeEdge Quick Demo
==============================
Simulates the full detection pipeline WITHOUT needing a camera or video file.
Generates synthetic frames that mimic:
  Phase 1 (0-3s):  Normal room — no signals
  Phase 2 (3-8s):  Heat shimmer starts — early detector triggers EARLY_WARNING
  Phase 3 (8-12s): Smoke appears — YOLO-style detection triggers FIRE ALERT

Run:
    python demo.py

Press Q to quit. Watch the banners change across the three phases.
"""

import cv2
import numpy as np
import time
import math
import sys
import random

# ── Try to import early_detector if available ─────────────────────────────────
try:
    from early_detector import EarlyFireDetector
    EARLY_DETECTOR_AVAILABLE = True
except ImportError:
    EARLY_DETECTOR_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# NTU CAMPUS LOCATIONS  (must match detector.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

NTU_LOCATIONS = [
    {
        "building": "The Hive",
        "floor":    2,
        "zone":     "Collaboration Studio",
        "campus":   "NTU",
        "lat":      1.34321,
        "lng":      103.68275,
    },
    {
        "building": "Northspine",
        "floor":    1,
        "zone":     "Food Court",
        "campus":   "NTU",
        "lat":      1.3431,
        "lng":      103.6805,
    },
    {
        "building": "School of Chemical and Biomedical Engineering",
        "floor":    3,
        "zone":     "Laboratory Wing",
        "campus":   "NTU",
        "lat":      1.34572,
        "lng":      103.67855,
    },
    {
        "building": "Hall of Residence 2",
        "floor":    4,
        "zone":     "Common Kitchen",
        "campus":   "NTU",
        "lat":      1.3547,
        "lng":      103.6853,
    },
]

# Pick a location at demo start; re-pick on each confirmed fire alert
_current_location = random.choice(NTU_LOCATIONS)

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC FRAME GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class SyntheticScene:
    """
    Generates fake CCTV-style room frames with controllable:
      - heat shimmer (sinusoidal pixel distortion)
      - smoke overlay (Gaussian blobs)
      - fire glow (orange gradient patch)
    """

    def __init__(self, w=960, h=540):
        self.w = w
        self.h = h
        # Build a static background once
        self._bg = self._make_background()

    def _make_background(self) -> np.ndarray:
        bg = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Floor
        bg[int(self.h*0.65):, :] = (60, 55, 50)
        # Wall
        bg[:int(self.h*0.65), :] = (85, 80, 78)
        # Table silhouette
        tx1, ty1, tx2, ty2 = int(self.w*0.3), int(self.h*0.55), int(self.w*0.7), int(self.h*0.65)
        cv2.rectangle(bg, (tx1, ty1), (tx2, ty2), (45, 40, 35), -1)
        # Window
        cv2.rectangle(bg, (int(self.w*0.6), int(self.h*0.1)),
                      (int(self.w*0.85), int(self.h*0.45)), (120, 140, 100), -1)
        cv2.rectangle(bg, (int(self.w*0.6), int(self.h*0.1)),
                      (int(self.w*0.85), int(self.h*0.45)), (90, 85, 80), 3)
        # Door
        cv2.rectangle(bg, (int(self.w*0.05), int(self.h*0.3)),
                      (int(self.w*0.18), int(self.h*0.65)), (55, 50, 45), -1)
        # Add subtle noise for texture
        noise = np.random.randint(0, 12, bg.shape, dtype=np.uint8)
        bg = cv2.add(bg, noise)
        return bg

    def render(
        self,
        shimmer: float = 0.0,   # 0-1, heat shimmer intensity
        smoke:   float = 0.0,   # 0-1, smoke density
        fire:    float = 0.0,   # 0-1, fire glow
        t:       float = 0.0,   # time for animation
    ) -> np.ndarray:
        frame = self._bg.copy()

        # ── Fire glow (bottom-center of table) ───────────────────────────────
        if fire > 0:
            cx, cy = int(self.w * 0.5), int(self.h * 0.58)
            # BGR: outer glow = deep red/orange, inner = bright yellow-white
            for r in range(90, 0, -5):
                ratio = 1.0 - r / 90.0
                intensity = fire * ratio
                # BGR: blue=0, green=mid, red=high → orange/yellow fire
                b = int(0)
                g = int(120 * intensity * ratio)
                red = int(220 * intensity)
                cv2.circle(frame, (cx, cy), r, (b, g, red), -1)
            # Bright yellow-white core
            flicker = 0.85 + 0.15 * math.sin(t * 15)
            core_r  = int(25 * fire * flicker)
            cv2.circle(frame, (cx, cy), core_r, (80, 220, 255), -1)  # BGR: bright yellow-white

        # ── Heat shimmer (pixel row displacement) ────────────────────────────
        if shimmer > 0:
            # Apply sinusoidal horizontal shift to rows in the "hot zone"
            y_start = int(self.h * 0.35)
            y_end   = int(self.h * 0.65)
            for y in range(y_start, y_end):
                shift = int(shimmer * 6 * math.sin(y * 0.3 + t * 8))
                if shift != 0:
                    row = frame[y, :].copy()
                    frame[y, :] = np.roll(row, shift, axis=0)

        # ── Smoke overlay ─────────────────────────────────────────────────────
        if smoke > 0:
            overlay = frame.copy()
            # Multiple blobs rising upward
            for i in range(6):
                bx = int(self.w * (0.35 + 0.05 * i + 0.03 * math.sin(t + i)))
                by = int(self.h * (0.55 - smoke * 0.4 - 0.05 * i))
                br = int(30 + 20 * smoke + 10 * i)
                cv2.circle(overlay, (bx, by), br, (160, 160, 155), -1)
            alpha = min(smoke * 0.65, 0.65)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame


# ══════════════════════════════════════════════════════════════════════════════
# DEMO PHASES
# ══════════════════════════════════════════════════════════════════════════════

PHASES = [
    # (duration_sec, label,           shimmer, smoke, fire,  early_warn, fire_alert)
    (3.5, "NORMAL — No activity",     0.0,     0.0,   0.0,   False,      False),
    (4.0, "HEAT SOURCE detected",     0.6,     0.0,   0.0,   True,       False),
    (2.5, "SMOKE beginning",          0.8,     0.4,   0.1,   True,       False),
    (3.0, "FIRE CONFIRMED",           1.0,     0.9,   0.8,   True,       True),
    (3.0, "FIRE CONFIRMED",           1.0,     1.0,   1.0,   True,       True),
]

TOTAL_DURATION = sum(p[0] for p in PHASES)


def get_phase(elapsed: float):
    t = 0.0
    for phase in PHASES:
        t += phase[0]
        if elapsed < t:
            return phase
    return PHASES[-1]


def lerp(a, b, alpha):
    return a + (b - a) * max(0.0, min(1.0, alpha))


# ══════════════════════════════════════════════════════════════════════════════
# HUD DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def draw_hud(frame: np.ndarray, phase, elapsed: float, early_warn: bool, fire_alert: bool, t: float, loc: dict) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    _, label, shimmer, smoke, fire_v, _, _ = phase

    # ── Top-left status panel ─────────────────────────────────────────────────
    panel_color = (20, 20, 20)
    cv2.rectangle(vis, (0, 0), (520, 130), panel_color, -1)
    cv2.rectangle(vis, (0, 0), (520, 130), (60, 60, 60), 1)

    risk = "CRITICAL" if fire_alert else ("EARLY_WARNING" if early_warn else "CLEAR")
    risk_color = {
        "CRITICAL":      (50,  50,  255),
        "EARLY_WARNING": (50,  165, 255),
        "CLEAR":         (50,  200, 50),
    }[risk]

    cv2.putText(vis, f"SafeEdge  |  NTU  |  {risk}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, risk_color, 2)
    cv2.putText(vis, f"Shimmer: {shimmer:.1f}   Smoke: {smoke:.1f}   Fire: {fire_v:.1f}",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

    # Location line — always visible, prominently coloured on fire
    loc_color = risk_color if (fire_alert or early_warn) else (160, 160, 160)
    loc_text  = f"{loc['building']}  |  Floor {loc['floor']}  |  {loc['zone']}"
    cv2.putText(vis, loc_text, (12, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, loc_color, 1)
    cv2.putText(vis, f"Phase: {label}", (12, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    # ── Timeline bar ──────────────────────────────────────────────────────────
    bar_x1, bar_y1 = 10, h - 20
    bar_x2, bar_y2 = w - 10, h - 8
    cv2.rectangle(vis, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 40, 40), -1)
    progress = min(elapsed / TOTAL_DURATION, 1.0)
    fill_x   = bar_x1 + int((bar_x2 - bar_x1) * progress)
    bar_fill_color = (50, 50, 220) if fire_alert else (50, 150, 220) if early_warn else (50, 180, 50)
    cv2.rectangle(vis, (bar_x1, bar_y1), (fill_x, bar_y2), bar_fill_color, -1)
    cv2.putText(vis, f"{elapsed:.1f}s / {TOTAL_DURATION:.0f}s",
                (bar_x1, bar_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

    # ── Early warning banner (orange) ─────────────────────────────────────────
    if early_warn and not fire_alert:
        pulse = 0.6 + 0.4 * abs(math.sin(t * 3))
        bcolor = (int(20 * pulse), int(100 * pulse), int(220 * pulse))
        cv2.rectangle(vis, (0, h - 55), (w, h - 28), bcolor, -1)
        cv2.putText(vis,
                    f"⚠  EARLY WARNING — {loc['building']} Floor {loc['floor']}  |  Monitoring...",
                    (12, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    # ── Fire confirmed banner (red) ───────────────────────────────────────────
    if fire_alert:
        pulse = 0.7 + 0.3 * abs(math.sin(t * 5))
        cv2.rectangle(vis, (0, h - 55), (w, h - 28), (int(30 * pulse), int(30 * pulse), int(220 * pulse)), -1)
        cv2.putText(vis,
                    f"🔥  FIRE @ {loc['building']} Floor {loc['floor']} ({loc['zone']}) — EVACUATE",
                    (12, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

        # Flashing red border when fire
        if int(t * 4) % 2 == 0:
            cv2.rectangle(vis, (2, 2), (w - 2, h - 2), (0, 0, 220), 4)

    # ── Signal indicators (top-right) ────────────────────────────────────────
    sigs = [
        ("OPTICAL FLOW",  shimmer > 0.3),
        ("BACKGROUND SUB", smoke  > 0.2 or shimmer > 0.5),
        ("TEXTURE VAR",   fire_v > 0.0 or shimmer > 0.4),
        ("YOLO FIRE",     fire_alert),
    ]
    for i, (name, active) in enumerate(sigs):
        color  = (50, 220, 50) if active else (80, 80, 80)
        dot    = (0, 200, 50) if active else (50, 50, 50)
        sx     = w - 230
        sy     = 20 + i * 26
        cv2.circle(vis, (sx, sy), 7, dot, -1)
        cv2.putText(vis, name, (sx + 16, sy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    return vis


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DEMO LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_demo():
    W, H    = 960, 540
    scene   = SyntheticScene(W, H)
    start   = time.time()

    # If early_detector is available, run it on synthetic frames too
    early   = EarlyFireDetector() if EARLY_DETECTOR_AVAILABLE else None

    print("━" * 60)
    print("  SafeEdge — Early Detection Demo")
    print("  Watch the 3 phases:")
    print("  [0-3s]  Normal scene")
    print("  [3-7s]  Heat shimmer → EARLY WARNING (orange)")
    print("  [7-12s] Fire/smoke   → FIRE CONFIRMED (red)")
    print("  Press Q to quit")
    print("━" * 60)

    frame_idx        = 0
    _prev_fire_flag  = False
    _current_loc     = random.choice(NTU_LOCATIONS)
    print(f"  Starting location: {_current_loc['building']} Floor {_current_loc['floor']}")
    print("━" * 60)

    while True:
        elapsed = time.time() - start
        t       = elapsed

        # Loop the demo — pick a new NTU location on each restart
        if elapsed > TOTAL_DURATION + 1.0:
            start        = time.time()
            elapsed      = 0.0
            _current_loc = random.choice(NTU_LOCATIONS)
            print(f"\n[Demo] Loop restart → new location: {_current_loc['building']}")
            if early:
                early.reset()

        phase = get_phase(elapsed)
        _, label, shimmer, smoke, fire_v, early_flag, fire_flag = phase

        # Repick NTU location the moment fire phase first activates
        if fire_flag and not _prev_fire_flag:
            _current_loc = random.choice(NTU_LOCATIONS)
            print(f"\n🔥 Fire alert → {_current_loc['building']} "
                  f"Floor {_current_loc['floor']} — {_current_loc['zone']}")
        _prev_fire_flag = fire_flag

        # Smooth transitions
        phase_start = 0.0
        for p in PHASES:
            if label == p[1]:
                break
            phase_start += p[0]
        phase_elapsed = elapsed - phase_start
        alpha = min(phase_elapsed / 1.0, 1.0)

        frame = scene.render(
            shimmer = lerp(0, shimmer, alpha),
            smoke   = lerp(0, smoke,   alpha),
            fire    = lerp(0, fire_v,  alpha),
            t       = t,
        )

        real_early = None
        if early:
            result = early.update(frame, frame_idx)
            if result:
                real_early = result

        vis = draw_hud(frame, phase, elapsed, early_flag, fire_flag, t, _current_loc)

        if real_early:
            cv2.putText(vis,
                        f"EarlyDetector: score={real_early.anomaly_score:.2f}  signals={real_early.active_signals}",
                        (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        cv2.imshow("SafeEdge — Fire Detection Demo", vis)
        frame_idx += 1

        key = cv2.waitKey(33) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()
    print("\nDemo ended.")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nDemo ended.")
