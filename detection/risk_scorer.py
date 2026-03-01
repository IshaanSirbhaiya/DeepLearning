"""
risk_scorer.py — SafeEdge Detection Layer
Multi-frame confidence-based fire risk scoring engine.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional
import time


# ─── Risk Levels ──────────────────────────────────────────────────────────────

class RiskLevel:
    IGNORE   = "IGNORE"
    WARNING  = "WARNING"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


# ─── Detection Frame Record ───────────────────────────────────────────────────

@dataclass
class FrameDetection:
    timestamp: float
    confidence: float
    label: str          # "fire" | "smoke"
    bbox: List[float]   # [x1, y1, x2, y2] normalised
    frame_index: int


# ─── Scorer Config ────────────────────────────────────────────────────────────

@dataclass
class ScorerConfig:
    # Confidence thresholds
    ignore_below:   float = 0.50
    warning_above:  float = 0.50
    high_above:     float = 0.70
    critical_above: float = 0.90

    # Multi-frame confirmation window
    window_size:  int = 8    # look at last N frames
    confirm_min:  int = 5    # need M positives in that window

    # How long (seconds) a confirmed alert stays active before auto-reset
    cooldown_sec: float = 30.0

    # Boost: if BOTH fire AND smoke detected together → push risk up one level
    dual_class_boost: bool = True


# ─── Risk Scorer ─────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Stateful multi-frame fire risk scorer.

    Usage:
        scorer = RiskScorer()
        result = scorer.update(detections_this_frame)
        if result.confirmed:
            print(result.risk_level, result.best_confidence)
    """

    def __init__(self, config: Optional[ScorerConfig] = None):
        self.cfg = config or ScorerConfig()
        self._window: Deque[Optional[FrameDetection]] = deque(maxlen=self.cfg.window_size)
        self._frame_idx: int = 0
        self._last_alert_time: float = 0.0
        self._consecutive_empty: int = 0
        self.total_alerts: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, detections: List[FrameDetection]) -> "ScoreResult":
        """
        Call once per frame with all detections from that frame.
        Returns a ScoreResult describing current risk state.
        """
        self._frame_idx += 1
        best = self._best_detection(detections)

        if best is None or best.confidence < self.cfg.ignore_below:
            self._window.append(None)
            self._consecutive_empty += 1
            return ScoreResult(
                confirmed=False,
                risk_level=RiskLevel.IGNORE,
                best_confidence=0.0,
                positive_frames=0,
                window_size=self.cfg.window_size,
                frame_index=self._frame_idx,
            )

        self._consecutive_empty = 0
        self._window.append(best)

        risk_level   = self._classify_confidence(best.confidence)
        positive_n   = self._count_positives()
        confirmed    = positive_n >= self.cfg.confirm_min
        dual_present = self._dual_class_present()

        # Apply dual-class boost (smoke + fire together = higher certainty)
        if confirmed and dual_present and self.cfg.dual_class_boost:
            risk_level = self._boost(risk_level)

        if confirmed:
            now = time.time()
            in_cooldown = (now - self._last_alert_time) < self.cfg.cooldown_sec
            if not in_cooldown:
                self._last_alert_time = now
                self.total_alerts += 1

        return ScoreResult(
            confirmed=confirmed,
            risk_level=risk_level,
            best_confidence=best.confidence,
            best_detection=best,
            positive_frames=positive_n,
            window_size=self.cfg.window_size,
            frame_index=self._frame_idx,
            dual_class=dual_present,
            in_cooldown=(time.time() - self._last_alert_time) < self.cfg.cooldown_sec
                        if confirmed else False,
        )

    def reset(self):
        """Clear window — call when scene changes or incident resolved."""
        self._window.clear()
        self._consecutive_empty = 0

    # ── Internals ─────────────────────────────────────────────────────────────

    def _best_detection(self, dets: List[FrameDetection]) -> Optional[FrameDetection]:
        """Return the highest-confidence detection above ignore threshold."""
        valid = [d for d in dets if d.confidence >= self.cfg.ignore_below]
        return max(valid, key=lambda d: d.confidence) if valid else None

    def _classify_confidence(self, conf: float) -> str:
        if conf >= self.cfg.critical_above:
            return RiskLevel.CRITICAL
        if conf >= self.cfg.high_above:
            return RiskLevel.HIGH
        if conf >= self.cfg.warning_above:
            return RiskLevel.WARNING
        return RiskLevel.IGNORE

    def _count_positives(self) -> int:
        return sum(1 for d in self._window if d is not None)

    def _dual_class_present(self) -> bool:
        """Check if both fire and smoke appear in current window."""
        labels = {d.label for d in self._window if d is not None}
        return "fire" in labels and "smoke" in labels

    def _boost(self, level: str) -> str:
        order = [RiskLevel.IGNORE, RiskLevel.WARNING, RiskLevel.HIGH, RiskLevel.CRITICAL]
        idx = order.index(level)
        return order[min(idx + 1, len(order) - 1)]


# ─── Score Result ─────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    confirmed:        bool
    risk_level:       str
    best_confidence:  float
    positive_frames:  int
    window_size:      int
    frame_index:      int
    best_detection:   Optional[FrameDetection] = None
    dual_class:       bool = False
    in_cooldown:      bool = False

    @property
    def should_alert(self) -> bool:
        """True if confirmed AND not in cooldown — i.e. actually send alert."""
        return self.confirmed and not self.in_cooldown

    def summary(self) -> str:
        return (
            f"[Frame {self.frame_index}] {self.risk_level} | "
            f"conf={self.best_confidence:.2f} | "
            f"positives={self.positive_frames}/{self.window_size} | "
            f"confirmed={self.confirmed} | alert={self.should_alert}"
        )
