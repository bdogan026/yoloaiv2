"""hand_phone.py — el-yuz jestinden sofor_eylemi (MediaPipe Hands + Face).

Telefon/sigara nesneleri cam-arkasindan cok kucuk (~8-15px) -> tespit IMKANSIZ.
Ama EL gorunur (kafa hizasinda el = jest). Eli YUZE gore siniflandirir:
  * el KULAGA yakin  -> telefonla_konusma
  * el AGZA yakin    -> sigara_icme  (surucu bolgesinde sise/bardak varsa -> su_icme)
Yuz bulunamazsa (crop/aci) -> el-kafa varsayilani telefonla_konusma (geri uyum).

Veri/egitim GEREKMEZ (pretrained). Komite dogrulamasi: el-kulak ayrimi calisiyor
(video_2_156: d_ear=0.029 < d_mouth=0.055 -> telefon).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.schema import KATEGORI_SOFOR_EYLEMI
from src.utils import get_logger

LOG = get_logger()


class HandPhoneDetector:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        h = (cfg.get("hand_phone", {}) or {})
        self.enabled = bool(h.get("enabled", True))
        paths = cfg.get("paths", {}) or {}
        self.hand_task = paths.get("hand_landmarker_task")
        self.face_task = paths.get("face_landmarker_task")
        self.min_frames = int(h.get("min_frames", 2))
        self.head_y_max = float(h.get("head_region_y_max", 0.7))
        self.min_det_conf = float(h.get("min_detection_confidence", 0.3))
        self.upscale_target = int(h.get("upscale_target_px", 600))
        self.conf = float(h.get("confidence", 0.7))
        self.sise_classes = list(h.get("drink_object_classes", ["sise"]))
        self._run = 0
        self._hand = None
        self._face = None
        if self.enabled:
            self._load()

    def _load(self) -> None:
        if not self.hand_task or not os.path.exists(self.hand_task):
            LOG.info("Hand .task yok; el-jest kapali (%s).", self.hand_task)
            self.enabled = False
            return
        try:
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            self._hand = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=mpp.BaseOptions(model_asset_path=self.hand_task),
                    num_hands=2, running_mode=vision.RunningMode.IMAGE,
                    min_hand_detection_confidence=self.min_det_conf))
            if self.face_task and os.path.exists(self.face_task):
                self._face = vision.FaceLandmarker.create_from_options(
                    vision.FaceLandmarkerOptions(
                        base_options=mpp.BaseOptions(model_asset_path=self.face_task),
                        num_faces=1, running_mode=vision.RunningMode.IMAGE))
            LOG.info("El-jest yuklendi (kulak/agiz ayrimi face=%s).",
                     self._face is not None)
        except Exception as exc:
            LOG.error("El-jest yuklenemedi (%s); kapali.", exc)
            self.enabled = False
            self._hand = None

    @property
    def active(self) -> bool:
        return self.enabled and self._hand is not None

    def update(self, crop_bgr: np.ndarray, t_seconds: float,
               dets: Optional[List[Any]] = None,
               driver_box: Optional[Tuple[float, float, float, float]] = None
               ) -> List[Dict[str, Any]]:
        if not self.active or crop_bgr is None or crop_bgr.size == 0:
            return []
        h, w = crop_bgr.shape[:2]
        if min(h, w) < self.upscale_target:
            s = min(4.0, self.upscale_target / float(min(h, w)))
            crop_bgr = cv2.resize(crop_bgr, None, fx=s, fy=s,
                                  interpolation=cv2.INTER_CUBIC)
        try:
            import mediapipe as mp
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            hr = self._hand.detect(img)
        except Exception as exc:
            LOG.debug("Hand detect hatasi: %s", exc)
            return []
        hands = getattr(hr, "hand_landmarks", None) or []
        hand_c = None
        for hd in hands:
            cy = sum(p.y for p in hd) / len(hd)
            if cy <= self.head_y_max:
                hand_c = (sum(p.x for p in hd) / len(hd), cy)
                break
        if hand_c is None:
            self._run = 0
            return []
        self._run += 1
        if self._run < self.min_frames:
            return []
        # Jest tipi: yuz varsa kulak/agiz ayrimi; yoksa varsayilan telefon
        gesture = "telefonla_konusma"
        if self._face is not None:
            try:
                fr = self._face.detect(img)
            except Exception:
                fr = None
            if fr and getattr(fr, "face_landmarks", None):
                import math
                lm = fr.face_landmarks[0]
                xs = [p.x for p in lm]
                ny = lm[1].y
                mouth = (lm[13].x, lm[13].y)
                earL = (min(xs), ny)
                earR = (max(xs), ny)
                dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
                d_ear = min(dist(hand_c, earL), dist(hand_c, earR))
                d_mouth = dist(hand_c, mouth)
                if d_mouth < d_ear:   # el AGIZDA -> sigara/su
                    gesture = ("su_icme" if self._sise_near(dets, driver_box)
                               else "sigara_icme")
        return [{"kategori": KATEGORI_SOFOR_EYLEMI, "etiket": gesture,
                 "confidence_score": float(self.conf),
                 "zaman_saniye": float(t_seconds)}]

    def _sise_near(self, dets, driver_box) -> bool:
        for d in (dets or []):
            if getattr(d, "cls_name", None) in self.sise_classes:
                if driver_box is None:
                    return True
                cx = (d.box[0] + d.box[2]) / 2.0
                cy = (d.box[1] + d.box[3]) / 2.0
                if (driver_box[0] <= cx <= driver_box[2]
                        and driver_box[1] <= cy <= driver_box[3]):
                    return True
        return False
