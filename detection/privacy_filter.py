"""
privacy_filter.py — SafeEdge Detection Layer
Face detection + Gaussian blur. Raw frame never leaves the edge node.

Strategy: tries MediaPipe first (more accurate), falls back to
OpenCV Haar cascades if MediaPipe is unavailable.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

logger = logging.getLogger("SafeEdge.PrivacyFilter")


@dataclass
class FaceRegion:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0


class PrivacyFilter:
    """
    Detects faces in a frame and replaces them with a strong Gaussian blur.
    Initialise once and call .apply(frame) per snapshot.
    """

    def __init__(
        self,
        blur_strength: int = 51,       # Gaussian kernel size — must be odd
        expand_ratio: float = 0.20,    # expand bbox by 20% for hairline coverage
        use_mediapipe: bool = True,    # prefer MediaPipe if available
        haar_scale: float = 1.1,
        haar_min_neighbors: int = 4,
        min_face_px: int = 20,         # ignore tiny detections (reflections, etc.)
    ):
        self.blur_strength    = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self.expand_ratio     = expand_ratio
        self.min_face_px      = min_face_px
        self._haar_scale      = haar_scale
        self._haar_min_n      = haar_min_neighbors

        self._detector        = None
        self._haar            = None
        self._mode            = "none"

        if use_mediapipe:
            self._try_init_mediapipe()
        if self._mode == "none":
            self._try_init_haar()

        logger.info(f"PrivacyFilter initialised — mode={self._mode}")

    # ── Public ────────────────────────────────────────────────────────────────

    def apply(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect faces and blur them in-place on a copy of frame.
        Returns (blurred_frame, face_count).
        """
        output = frame.copy()
        faces  = self._detect(frame)

        for face in faces:
            x1, y1, x2, y2 = self._expand_bbox(face, frame.shape)
            roi = output[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            # Double-blur for strong anonymisation
            blurred = cv2.GaussianBlur(roi, (self.blur_strength, self.blur_strength), 0)
            blurred = cv2.GaussianBlur(blurred, (self.blur_strength, self.blur_strength), 0)
            output[y1:y2, x1:x2] = blurred

        return output, len(faces)

    @property
    def mode(self) -> str:
        return self._mode

    # ── Detectors ─────────────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> List[FaceRegion]:
        if self._mode == "mediapipe":
            return self._detect_mediapipe(frame)
        if self._mode == "haar":
            return self._detect_haar(frame)
        return []  # no detector — still return blurred-of-nothing (safe)

    def _detect_mediapipe(self, frame: np.ndarray) -> List[FaceRegion]:
        import mediapipe as mp
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._detector.process(rgb)
        faces  = []

        if result.detections:
            for det in result.detections:
                bb  = det.location_data.relative_bounding_box
                x1  = max(0, int(bb.xmin * w))
                y1  = max(0, int(bb.ymin * h))
                x2  = min(w, int((bb.xmin + bb.width)  * w))
                y2  = min(h, int((bb.ymin + bb.height) * h))
                if (x2 - x1) >= self.min_face_px and (y2 - y1) >= self.min_face_px:
                    faces.append(FaceRegion(x1, y1, x2, y2,
                                            det.score[0] if det.score else 1.0))
        return faces

    def _detect_haar(self, frame: np.ndarray) -> List[FaceRegion]:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        dets  = self._haar.detectMultiScale(
            gray,
            scaleFactor=self._haar_scale,
            minNeighbors=self._haar_min_n,
            minSize=(self.min_face_px, self.min_face_px),
        )
        faces = []
        for (x, y, fw, fh) in dets:
            faces.append(FaceRegion(x, y, x + fw, y + fh))
        return faces

    # ── Initialisers ──────────────────────────────────────────────────────────

    def _try_init_mediapipe(self):
        try:
            import mediapipe as mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,           # 1 = full-range model
                min_detection_confidence=0.5
            )
            self._mode = "mediapipe"
        except ImportError:
            logger.warning("MediaPipe not installed — falling back to Haar cascades")
        except Exception as e:
            logger.warning(f"MediaPipe init failed ({e}) — falling back to Haar")

    def _try_init_haar(self):
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            if self._haar.empty():
                raise RuntimeError("Haar cascade file not found or empty")
            self._mode = "haar"
        except Exception as e:
            logger.error(f"Haar cascade init failed: {e}. Faces will NOT be blurred.")
            self._mode = "none"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _expand_bbox(
        self, face: FaceRegion, shape: Tuple[int, ...]
    ) -> Tuple[int, int, int, int]:
        """Expand bounding box by expand_ratio to cover hair and ears."""
        h, w  = shape[:2]
        fw    = face.x2 - face.x1
        fh    = face.y2 - face.y1
        pad_x = int(fw * self.expand_ratio)
        pad_y = int(fh * self.expand_ratio)
        x1    = max(0, face.x1 - pad_x)
        y1    = max(0, face.y1 - pad_y)
        x2    = min(w, face.x2 + pad_x)
        y2    = min(h, face.y2 + pad_y)
        return x1, y1, x2, y2
