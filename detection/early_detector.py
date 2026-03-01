"""
early_detector.py — SafeEdge Detection Layer
=============================================
Pre-fire anomaly detection engine.

Detects EARLY WARNING signs BEFORE visible flames appear:
  1. Optical Flow Anomaly  — heat shimmer causes rising, irregular pixel motion
  2. Background Subtraction — sudden localised scene changes in a static zone
  3. Texture Variance Spike — heat haze disrupts local texture consistency

This module runs ALONGSIDE the existing YOLOv8n detector — it does NOT replace it.
When an anomaly is detected, it emits an EARLY_WARNING event with lower confidence
than a confirmed YOLO fire, giving residents/operators earlier notice.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO INTEGRATE INTO detector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  from early_detector import EarlyFireDetector, EarlyWarning

  # In FireDetector.__init__():
  self.early_detector = EarlyFireDetector()

  # In FireDetector.run() loop, BEFORE the YOLO inference block:
  warning = self.early_detector.update(frame, frame_idx)
  if warning and warning.should_alert:
      logger.warning(f"⚠️  EARLY WARNING [{warning.anomaly_type}] "
                     f"score={warning.anomaly_score:.2f}")
      # Optionally publish to fire_event bus at lower severity:
      # fire_event.publish(..., risk_level="EARLY_WARNING", ...)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNAL OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Signal              | What it catches
  --------------------|--------------------------------------------------
  Optical flow        | Heat shimmer: rising, turbulent, irregular motion
  Background sub      | Localised pixel changes in previously static areas
  Texture variance    | Blurring/distortion of surfaces near heat source
  Combined score      | Weighted fusion — reduces false positives

All three signals must clear their individual thresholds OR the combined
weighted score must exceed COMBINED_THRESHOLD before an alert fires.
This keeps false-positive rate low on normal indoor scenes.
"""

import cv2
import numpy as np
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Deque

logger = logging.getLogger("SafeEdge.EarlyDetector")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class EarlyDetectorConfig:
    # ── Optical Flow ──────────────────────────────────────────────────────────
    # Farneback params (see cv2.calcOpticalFlowFarneback docs)
    OF_PYR_SCALE    = 0.5
    OF_LEVELS       = 3
    OF_WINSIZE      = 15
    OF_ITERATIONS   = 3
    OF_POLY_N       = 5
    OF_POLY_SIGMA   = 1.2

    # Heat shimmer = upward-biased, high-magnitude, turbulent flow
    OF_MAG_THRESHOLD    = 1.5    # minimum mean flow magnitude to consider
    OF_UPWARD_RATIO     = 0.55   # fraction of flow vectors pointing upward
    OF_TURBULENCE_STD   = 0.8    # angular std dev (radians) — shimmer is chaotic
    OF_SCORE_WEIGHT     = 0.45   # contribution to combined score

    # ── Background Subtraction ────────────────────────────────────────────────
    BS_HISTORY          = 200    # MOG2 history frames
    BS_VAR_THRESHOLD    = 50     # MOG2 sensitivity (lower = more sensitive)
    BS_DETECT_SHADOWS   = False  # shadows waste compute here
    BS_AREA_THRESHOLD   = 0.02   # min fraction of frame area that must change
    BS_SCORE_WEIGHT     = 0.30

    # ── Texture Variance ──────────────────────────────────────────────────────
    TV_KERNEL_SIZE      = 15     # Laplacian kernel — local texture measurement
    TV_BASELINE_FRAMES  = 30     # frames to establish texture baseline
    TV_SPIKE_MULTIPLIER = 1.8    # anomaly if variance spikes >N× baseline
    TV_SCORE_WEIGHT     = 0.25

    # ── Combined Alert Logic ──────────────────────────────────────────────────
    COMBINED_THRESHOLD  = 0.55   # weighted score above this → EARLY_WARNING
    SIGNAL_MIN_COUNT    = 2      # at least 2 of 3 signals must be active
    CONFIRM_FRAMES      = 6      # consecutive anomaly frames before alerting
    COOLDOWN_SEC        = 45.0   # don't re-alert within this window

    # ── Visualisation ─────────────────────────────────────────────────────────
    DRAW_OVERLAY        = True   # draw debug overlay on display frame


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalResult:
    active:  bool
    score:   float          # 0.0 – 1.0
    detail:  str = ""       # human-readable reason


@dataclass
class EarlyWarning:
    frame_index:   int
    timestamp:     float
    anomaly_type:  str              # "optical_flow" | "background" | "texture" | "combined"
    anomaly_score: float            # 0.0 – 1.0
    active_signals: List[str]       # which sub-detectors triggered
    confirm_count: int              # how many consecutive frames triggered
    should_alert:  bool             # True only after CONFIRM_FRAMES threshold + cooldown
    bbox:          Optional[Tuple[int,int,int,int]] = None   # rough region of interest

    def summary(self) -> str:
        return (
            f"[EarlyWarning frame={self.frame_index}] "
            f"score={self.anomaly_score:.2f} "
            f"signals={self.active_signals} "
            f"confirm={self.confirm_count} "
            f"alert={self.should_alert}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL DETECTORS
# ══════════════════════════════════════════════════════════════════════════════

class OpticalFlowDetector:
    """
    Detects heat-shimmer motion:
      - Rising pixel motion  (upward Y-component dominant)
      - High turbulence      (angular standard deviation of flow vectors)
      - Minimum magnitude    (ignore sensor noise)
    """

    def __init__(self, cfg: EarlyDetectorConfig = EarlyDetectorConfig):
        self._cfg      = cfg
        self._prev_gray: Optional[np.ndarray] = None
        self._flow_buf: Deque[float] = deque(maxlen=10)

    def update(self, frame: np.ndarray) -> SignalResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return SignalResult(active=False, score=0.0, detail="initialising")

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            None,
            self._cfg.OF_PYR_SCALE,
            self._cfg.OF_LEVELS,
            self._cfg.OF_WINSIZE,
            self._cfg.OF_ITERATIONS,
            self._cfg.OF_POLY_N,
            self._cfg.OF_POLY_SIGMA,
            0,
        )
        self._prev_gray = gray

        fx, fy = flow[..., 0], flow[..., 1]
        magnitude = np.sqrt(fx**2 + fy**2)
        mean_mag  = float(np.mean(magnitude))

        # Not enough motion — skip
        if mean_mag < self._cfg.OF_MAG_THRESHOLD:
            self._flow_buf.append(0.0)
            return SignalResult(active=False, score=0.0,
                                detail=f"low motion mag={mean_mag:.2f}")

        # Upward bias — heat rises, so fy should be predominantly negative
        # (image coords: y increases downward, so upward = negative fy)
        total_vectors = fx.size
        upward_count  = float(np.sum(fy < -0.3))
        upward_ratio  = upward_count / total_vectors

        # Angular turbulence — shimmer is chaotic, not smooth camera motion
        angles       = np.arctan2(fy, fx)
        angular_std  = float(np.std(angles))

        # Score components
        upward_score     = min(upward_ratio / self._cfg.OF_UPWARD_RATIO, 1.0)
        turbulence_score = min(angular_std / (self._cfg.OF_TURBULENCE_STD * 2), 1.0)
        mag_score        = min(mean_mag / (self._cfg.OF_MAG_THRESHOLD * 3), 1.0)

        score = (upward_score * 0.4 + turbulence_score * 0.4 + mag_score * 0.2)
        self._flow_buf.append(score)

        active = (
            upward_ratio  >= self._cfg.OF_UPWARD_RATIO and
            angular_std   >= self._cfg.OF_TURBULENCE_STD and
            mean_mag      >= self._cfg.OF_MAG_THRESHOLD
        )

        return SignalResult(
            active=active,
            score=score,
            detail=(
                f"mag={mean_mag:.2f} "
                f"upward={upward_ratio:.2f} "
                f"turb={angular_std:.2f}"
            ),
        )


class BackgroundSubtractionDetector:
    """
    MOG2-based detector. Catches localised pixel changes in a previously
    static scene — pre-fire heating causes subtle colour/luminance shifts.
    """

    def __init__(self, cfg: EarlyDetectorConfig = EarlyDetectorConfig):
        self._cfg = cfg
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history        = cfg.BS_HISTORY,
            varThreshold   = cfg.BS_VAR_THRESHOLD,
            detectShadows  = cfg.BS_DETECT_SHADOWS,
        )
        self._frame_count = 0

    def update(self, frame: np.ndarray) -> Tuple[SignalResult, Optional[np.ndarray]]:
        self._frame_count += 1

        # MOG2 needs warm-up frames to build background model
        mask = self._mog2.apply(frame)

        if self._frame_count < 20:
            return SignalResult(active=False, score=0.0, detail="warming up"), mask

        # Morphological cleaning — remove noise pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        total_pixels   = frame.shape[0] * frame.shape[1]
        changed_pixels = int(np.sum(mask > 0))
        change_ratio   = changed_pixels / total_pixels

        score = min(change_ratio / (self._cfg.BS_AREA_THRESHOLD * 5), 1.0)

        active = change_ratio >= self._cfg.BS_AREA_THRESHOLD

        return SignalResult(
            active=active,
            score=score,
            detail=f"change_ratio={change_ratio:.3f}",
        ), mask


class TextureVarianceDetector:
    """
    Tracks local Laplacian variance across the frame.
    Heat haze blurs and distorts surface textures — variance spikes
    compared to a rolling baseline.
    """

    def __init__(self, cfg: EarlyDetectorConfig = EarlyDetectorConfig):
        self._cfg      = cfg
        self._baseline: Optional[float] = None
        self._baseline_buf: Deque[float] = deque(maxlen=cfg.TV_BASELINE_FRAMES)
        self._frame_count = 0

    def update(self, frame: np.ndarray) -> SignalResult:
        self._frame_count += 1
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap      = cv2.Laplacian(gray, cv2.CV_64F, ksize=self._cfg.TV_KERNEL_SIZE)
        variance = float(np.var(lap))

        # Build baseline during first N frames
        if self._frame_count <= self._cfg.TV_BASELINE_FRAMES:
            self._baseline_buf.append(variance)
            self._baseline = float(np.mean(self._baseline_buf))
            return SignalResult(active=False, score=0.0,
                                detail=f"building baseline var={variance:.1f}")

        if self._baseline is None or self._baseline < 1.0:
            return SignalResult(active=False, score=0.0, detail="baseline not ready")

        spike_ratio = variance / self._baseline

        # Update baseline slowly (exponential moving average) so slow drift
        # doesn't mask a real anomaly
        self._baseline = 0.95 * self._baseline + 0.05 * variance

        score  = min((spike_ratio - 1.0) / (self._cfg.TV_SPIKE_MULTIPLIER - 1.0), 1.0)
        score  = max(score, 0.0)
        active = spike_ratio >= self._cfg.TV_SPIKE_MULTIPLIER

        return SignalResult(
            active=active,
            score=score,
            detail=f"spike_ratio={spike_ratio:.2f} baseline={self._baseline:.1f}",
        )


# ══════════════════════════════════════════════════════════════════════════════
# EARLY FIRE DETECTOR  (main class — this is what detector.py imports)
# ══════════════════════════════════════════════════════════════════════════════

class EarlyFireDetector:
    """
    Fuses three pre-fire anomaly signals into a single early warning.

    Drop-in companion to the YOLOv8n FireDetector — call .update(frame, idx)
    on every processed frame and check EarlyWarning.should_alert.

    Does NOT require any ML model or API key — runs entirely on CPU with OpenCV.
    """

    def __init__(self, config: EarlyDetectorConfig = EarlyDetectorConfig):
        self._cfg        = config
        self._of         = OpticalFlowDetector(config)
        self._bs         = BackgroundSubtractionDetector(config)
        self._tv         = TextureVarianceDetector(config)

        self._confirm_count    = 0
        self._last_alert_time  = 0.0
        self._frame_idx        = 0
        self.total_warnings    = 0

        # Rolling score history for smoothing
        self._score_buf: Deque[float] = deque(maxlen=10)

        logger.info("EarlyFireDetector initialised (optical_flow + bg_sub + texture)")

    # ── Public ────────────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        frame_idx: Optional[int] = None,
    ) -> Optional[EarlyWarning]:
        """
        Process one frame. Returns EarlyWarning if anomaly detected, else None.

        Args:
            frame:      BGR frame from OpenCV (same frame passed to YOLO)
            frame_idx:  Optional frame counter for logging

        Returns:
            EarlyWarning dataclass, or None if no anomaly.
        """
        self._frame_idx = frame_idx or self._frame_idx + 1

        # ── Run all three signals ─────────────────────────────────────────────
        of_result             = self._of.update(frame)
        bs_result, bs_mask    = self._bs.update(frame)
        tv_result             = self._tv.update(frame)

        cfg = self._cfg

        # ── Weighted combined score ───────────────────────────────────────────
        combined_score = (
            of_result.score * cfg.OF_SCORE_WEIGHT +
            bs_result.score * cfg.BS_SCORE_WEIGHT +
            tv_result.score * cfg.TV_SCORE_WEIGHT
        )

        # Smooth score over last N frames to avoid single-frame spikes
        self._score_buf.append(combined_score)
        smoothed_score = float(np.mean(self._score_buf))

        # ── Determine active signals ──────────────────────────────────────────
        active_signals: List[str] = []
        if of_result.active:
            active_signals.append("optical_flow")
        if bs_result.active:
            active_signals.append("background_sub")
        if tv_result.active:
            active_signals.append("texture_variance")

        # ── Gate: need minimum signals active AND score above threshold ───────
        is_anomaly = (
            len(active_signals) >= cfg.SIGNAL_MIN_COUNT and
            smoothed_score >= cfg.COMBINED_THRESHOLD
        )

        if is_anomaly:
            self._confirm_count += 1
        else:
            # Decay confirm count — require sustained signal
            self._confirm_count = max(0, self._confirm_count - 1)

        # ── Decide whether to fire alert ──────────────────────────────────────
        should_alert = False
        if self._confirm_count >= cfg.CONFIRM_FRAMES:
            now = time.time()
            in_cooldown = (now - self._last_alert_time) < cfg.COOLDOWN_SEC
            if not in_cooldown:
                self._last_alert_time = now
                self.total_warnings  += 1
                should_alert          = True
                logger.warning(
                    f"⚠️  EARLY_WARNING score={smoothed_score:.2f} "
                    f"signals={active_signals} frame={self._frame_idx}"
                )

        # ── Debug logging every 30 frames ────────────────────────────────────
        if self._frame_idx % 30 == 0:
            logger.debug(
                f"EarlyDetector frame={self._frame_idx} "
                f"score={smoothed_score:.2f} confirm={self._confirm_count} "
                f"OF=[{of_result.detail}] "
                f"BS=[{bs_result.detail}] "
                f"TV=[{tv_result.detail}]"
            )

        # Only return a result if we have an anomaly or an alert
        if not is_anomaly and not should_alert:
            return None

        # Determine dominant anomaly type for labelling
        anomaly_type = "combined"
        if len(active_signals) == 1:
            anomaly_type = active_signals[0]

        # Rough region of interest from background subtraction mask
        bbox = self._extract_roi(bs_mask, frame.shape)

        return EarlyWarning(
            frame_index    = self._frame_idx,
            timestamp      = time.time(),
            anomaly_type   = anomaly_type,
            anomaly_score  = round(smoothed_score, 4),
            active_signals = active_signals,
            confirm_count  = self._confirm_count,
            should_alert   = should_alert,
            bbox           = bbox,
        )

    def draw_overlay(
        self,
        frame: np.ndarray,
        warning: Optional[EarlyWarning],
    ) -> np.ndarray:
        """
        Draw early-warning overlay onto the visualisation frame.
        Call this after FireDetector._draw() so it layers on top.
        """
        if not EarlyDetectorConfig.DRAW_OVERLAY:
            return frame

        vis = frame.copy()
        h, w = vis.shape[:2]

        # Status bar at top-right
        status_color = (0, 165, 255) if warning else (0, 200, 0)
        status_text  = "EARLY: ANOMALY" if warning else "EARLY: CLEAR"
        score_text   = f"score={warning.anomaly_score:.2f}" if warning else ""
        cv2.putText(vis, status_text, (w - 260, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        if score_text:
            cv2.putText(vis, score_text, (w - 260, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)

        # Draw region of interest box if we have one
        if warning and warning.bbox:
            x1, y1, x2, y2 = warning.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(vis, "HEAT ANOMALY", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        # Alert banner (orange — distinct from YOLO's red)
        if warning and warning.should_alert:
            cv2.rectangle(vis, (0, h - 90), (w, h - 52), (0, 100, 200), -1)
            sigs = " + ".join(warning.active_signals)
            cv2.putText(
                vis,
                f"⚠ EARLY WARNING — {sigs}",
                (10, h - 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
            )

        return vis

    def reset(self):
        """Reset internal state — call between video files or camera restarts."""
        self._confirm_count   = 0
        self._last_alert_time = 0.0
        self._score_buf.clear()
        logger.info("EarlyFireDetector state reset")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_roi(
        self,
        mask: Optional[np.ndarray],
        shape: Tuple,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Find bounding box of largest changed region from BG subtraction mask."""
        if mask is None or not np.any(mask > 0):
            return None

        try:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                return None

            largest = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)
            h, w = shape[:2]

            # Clamp to frame bounds with small padding
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + cw + pad)
            y2 = min(h, y + ch + pad)
            return x1, y1, x2, y2

        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION PATCH FOR detector.py
# ══════════════════════════════════════════════════════════════════════════════

INTEGRATION_INSTRUCTIONS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Add import at the top of detector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from early_detector import EarlyFireDetector, EarlyWarning

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Add to FireDetector.__init__() after self.metrics = EdgeMetrics()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

self.early_detector = EarlyFireDetector()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Add to FireDetector.run() loop, BEFORE the YOLO inference block
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Early pre-fire detection ───────────────────────────────────────────────
early_warning = self.early_detector.update(frame, frame_idx)
if early_warning and early_warning.should_alert:
    logger.warning(
        f"⚠️  EARLY_WARNING [{early_warning.anomaly_type}] "
        f"score={early_warning.anomaly_score:.2f} "
        f"signals={early_warning.active_signals}"
    )
    fire_event.publish(
        building   = self.generator.location.building,
        floor      = self.generator.location.floor,
        zone       = self.generator.location.zone,
        confidence = early_warning.anomaly_score,
        risk_level = "EARLY_WARNING",
        camera_id  = DetectorConfig.CAMERA_ID,
    )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — Add to FireDetector._draw() to show early overlay
         (add at the END of _draw(), before 'return vis')
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

vis = self.early_detector.draw_overlay(vis, getattr(self, '_last_early_warning', None))
return vis

And in the run() loop, store the warning:
self._last_early_warning = early_warning

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — Add "EARLY_WARNING" to fire_event.py subscriber messages
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In fire_event.py FireEventBus.get_latest(), the risk_level field will now
also contain "EARLY_WARNING" — teammates can filter on this to show a
different message: "⚠️ Possible heat source detected — monitoring..."
"""


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST (run directly to verify signals work)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Test EarlyFireDetector standalone")
    parser.add_argument("--input", default="0", help="Video source (file/webcam/RTSP)")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    detector = EarlyFireDetector()
    source   = int(args.input) if args.input.isdigit() else args.input
    cap      = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {args.input}")
        sys.exit(1)

    print("━" * 60)
    print("SafeEdge Early Fire Detector — Standalone Test")
    print("Press Q to quit")
    print("━" * 60)
    print(INTEGRATION_INSTRUCTIONS)

    frame_idx = 0
    warnings  = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            warning = detector.update(frame, frame_idx)

            if warning:
                warnings += 1
                print(warning.summary())

            if not args.no_display:
                vis = detector.draw_overlay(frame, warning)
                cv2.imshow("SafeEdge — Early Fire Detector", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone — {frame_idx} frames | {warnings} early warnings")
        print(f"Total confirmed alerts: {detector.total_warnings}")
