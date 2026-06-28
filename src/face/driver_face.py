"""driver_face.py — MediaPipe Face Landmarker ile surucu yuzu analizi.

Uretilen sofor_eylemi olaylari:
  * arkaya_bakma   : buyuk bas yaw acisi (yana/arkaya donus), sureli
  * etrafa_bakinma : tekrarlayan bas yaw salinimi (pencere uzerinde)
  * esneme         : agiz acikligi (jawOpen blendshape), sureli
  * (bonus) yorgunluk: goz kapanma orani (config ile acilabilir)

.task dosyasi image'a gomulu; LOKAL yoldan yuklenir (offline). Yalnizca surucu
yuz kutusu varken (kabin/arac crop'unda) calistirilir (gecikme/butce).

Bu modul yuz-alaninda kendi sure/salinim mantigini tutar; GLOBAL debounce ve
zaman damgalama temporal.py'de yapilir.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.face.drowsiness_cnn import DrowsinessCNN
from src.schema import KATEGORI_SOFOR_EYLEMI
from src.utils import get_logger

LOG = get_logger()


@dataclass
class FaceSignals:
    valid: bool
    yaw_deg: float = 0.0
    jaw_open: float = 0.0
    eye_blink: float = 0.0


class DriverFaceAnalyzer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        f = cfg.get("face", {}) or {}
        self.enabled: bool = bool(f.get("enabled", True))
        self.task_path: Optional[str] = (cfg.get("paths", {}) or {}).get(
            "face_landmarker_task")
        self.driver_selection: str = str(f.get("driver_selection", "largest"))
        self.min_face_px: int = int(f.get("min_face_size_px", 24))
        # Kucuk yuz crop'unu MediaPipe oncesi buyutme (cam-arkasi uzak surucu).
        self.upscale_target: int = int(f.get("face_upscale_target_px", 600))
        self.upscale_max: float = float(f.get("face_upscale_max", 4.0))

        self.arkaya_yaw: float = float(f.get("arkaya_bakma_yaw_deg", 45))
        self.arkaya_min: int = int(f.get("arkaya_bakma_min_frames", 4))
        self.etraf_std: float = float(f.get("etrafa_bakinma_yaw_std_deg", 18))
        self.etraf_win: int = int(f.get("etrafa_bakinma_window", 30))
        self.etraf_cross: int = int(f.get("etrafa_bakinma_min_crossings", 3))
        self.esneme_jaw: float = float(f.get("esneme_jaw_open", 0.45))
        self.esneme_min: int = int(f.get("esneme_min_frames", 5))

        dz = f.get("drowsiness", {}) or {}
        self.drowsy_enabled: bool = bool(dz.get("enabled", False))
        self.drowsy_thr: float = float(dz.get("eye_blink_threshold", 0.5))
        self.drowsy_min: int = int(dz.get("min_frames", 8))

        # sure sayaclari + salinim penceresi
        self._arkaya_run = 0
        self._esneme_run = 0
        self._drowsy_run = 0
        self._yaw_window: Deque[float] = deque(maxlen=self.etraf_win)

        self._landmarker = None
        if self.enabled:
            self._init_landmarker()

        # Yedek CNN (MediaPipe basarisiz olunca esneme/yorgunluk; offline)
        self.cnn = DrowsinessCNN(cfg)

    def _init_landmarker(self) -> None:
        if not self.task_path:
            LOG.warning("Face .task yolu yok; surucu yuz analizi kapali.")
            self.enabled = False
            return
        try:
            import os
            if not os.path.exists(self.task_path):
                LOG.warning("Face .task bulunamadi (%s); yuz analizi kapali.",
                            self.task_path)
                self.enabled = False
                return
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            base = mp_python.BaseOptions(model_asset_path=self.task_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
                running_mode=vision.RunningMode.IMAGE,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            LOG.info("MediaPipe Face Landmarker yuklendi: %s", self.task_path)
        except Exception as exc:  # gurbuzluk
            LOG.error("Face Landmarker yuklenemedi (%s); yuz analizi kapali.", exc)
            self.enabled = False
            self._landmarker = None

    @property
    def ready(self) -> bool:
        return self.enabled and self._landmarker is not None

    @property
    def active(self) -> bool:
        """MediaPipe ya da yedek CNN'den en az biri kullanilabilir mi?"""
        return self.ready or self.cnn.ready

    def analyze(self, crop_bgr: np.ndarray) -> FaceSignals:
        """Surucu kabin/yuz crop'unda tek kare yuz sinyalleri."""
        if not self.ready or crop_bgr is None or crop_bgr.size == 0:
            return FaceSignals(valid=False)
        h, w = crop_bgr.shape[:2]
        if min(h, w) < self.min_face_px:
            return FaceSignals(valid=False)
        # Kucuk crop'u buyut: MediaPipe yuz detektorunun mutlak bir min boyutu var;
        # cam-arkasi uzak surucu yuzunde upscale tespiti ciddi artirir
        # (komite video_3: gevsek/uzak crop 0/12 -> siki+upscale 9/12).
        if min(h, w) < self.upscale_target:
            s = min(self.upscale_max, self.upscale_target / float(min(h, w)))
            crop_bgr = cv2.resize(crop_bgr, None, fx=s, fy=s,
                                  interpolation=cv2.INTER_CUBIC)
        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._landmarker.detect(mp_img)
        except Exception as exc:
            LOG.debug("Face detect hatasi: %s", exc)
            return FaceSignals(valid=False)

        if not getattr(res, "face_landmarks", None):
            return FaceSignals(valid=False)

        yaw = self._yaw_from_matrix(res)
        jaw = self._blendshape(res, "jawOpen")
        blink = 0.5 * (self._blendshape(res, "eyeBlinkLeft") +
                       self._blendshape(res, "eyeBlinkRight"))
        return FaceSignals(valid=True, yaw_deg=yaw, jaw_open=jaw, eye_blink=blink)

    def update(self, crop_bgr: np.ndarray, t_seconds: float) -> List[Dict[str, Any]]:
        """Karedeki yuz sinyallerinden ham sofor_eylemi olay adaylari uretir.

        MediaPipe yuz/landmark bulursa bas-poz (arkaya/etrafa) + esneme oradan.
        MediaPipe basarisiz olursa (uzak/acili/cam) yedek CNN ile esneme/yorgunluk.
        """
        events: List[Dict[str, Any]] = []
        sig = self.analyze(crop_bgr)
        pose_valid = sig.valid

        # --- esneme sinyali: MediaPipe (jaw_open) veya yedek CNN ---
        is_yawn, yawn_conf = False, 0.0
        if pose_valid:
            is_yawn = sig.jaw_open >= self.esneme_jaw
            yawn_conf = min(1.0, sig.jaw_open + 0.2)
        elif self.cnn.ready and self.cnn.use_when in ("mediapipe_failed", "always"):
            label, conf = self.cnn.predict(crop_bgr)
            if label == "esneme" and conf >= self.cnn.min_conf:
                is_yawn, yawn_conf = True, conf
            # label == "goz_kapali" -> yorgunluk sinyali (canli risk; sema'da
            # ayri etiket yok, esneme ile birlikte degerlendirilir)

        if is_yawn:
            self._esneme_run += 1
        else:
            self._esneme_run = 0
        if self._esneme_run >= self.esneme_min:
            events.append(self._ev("esneme", yawn_conf, t_seconds))

        # --- bas-poz tabanli olaylar yalnizca MediaPipe gecerliyken ---
        if pose_valid:
            # arkaya_bakma (sureli buyuk yaw)
            if abs(sig.yaw_deg) >= self.arkaya_yaw:
                self._arkaya_run += 1
            else:
                self._arkaya_run = 0
            if self._arkaya_run >= self.arkaya_min:
                conf = min(1.0, 0.5 + abs(sig.yaw_deg) / 90.0)
                events.append(self._ev("arkaya_bakma", conf, t_seconds))

            # etrafa_bakinma (yaw salinimi, pencere)
            self._yaw_window.append(sig.yaw_deg)
            if len(self._yaw_window) >= self.etraf_win:
                arr = np.asarray(self._yaw_window, dtype=np.float32)
                std = float(arr.std())
                crossings = self._zero_crossings(arr - arr.mean())
                if std >= self.etraf_std and crossings >= self.etraf_cross:
                    conf = min(1.0, 0.5 + std / 60.0)
                    events.append(self._ev("etrafa_bakinma", conf, t_seconds))

            # (bonus) yorgunluk: goz kapanmasi (canli risk sinyali)
            if self.drowsy_enabled:
                if sig.eye_blink >= self.drowsy_thr:
                    self._drowsy_run += 1
                else:
                    self._drowsy_run = 0
        else:
            self._arkaya_run = 0  # poz yoksa arkaya sayacini sifirla

        return events

    # ------------------------------------------------------------------
    @staticmethod
    def _ev(etiket: str, conf: float, t: float) -> Dict[str, Any]:
        return {
            "kategori": KATEGORI_SOFOR_EYLEMI,
            "etiket": etiket,
            "confidence_score": float(max(0.0, min(1.0, conf))),
            "zaman_saniye": float(t),
        }

    @staticmethod
    def _zero_crossings(x: np.ndarray) -> int:
        s = np.sign(x)
        s[s == 0] = 1
        return int(np.sum(s[1:] != s[:-1]))

    @staticmethod
    def _yaw_from_matrix(res: Any) -> float:
        """facial_transformation_matrixes[0]'tan yaw (derece)."""
        try:
            M = np.array(res.facial_transformation_matrixes[0]).reshape(4, 4)
            R = M[:3, :3]
            # XYZ Euler: yaw = atan2(-R20, sqrt(R21^2 + R22^2))
            yaw = math.degrees(math.atan2(-R[2, 0],
                                          math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
            return float(yaw)
        except Exception:
            return 0.0

    @staticmethod
    def _blendshape(res: Any, name: str) -> float:
        try:
            for cat in res.face_blendshapes[0]:
                if cat.category_name == name:
                    return float(cat.score)
        except Exception:
            pass
        return 0.0
