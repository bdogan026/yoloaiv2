"""sliced.py — SAHI (Slicing Aided Hyper Inference) dilimli cikarim.

Kucuk/uzak nesneler (plaka, teknocan, sigara) duz cikarimda kacabilir. SAHI
kareyi/bolgeyi dilimlere bolup her dilimde cikarim yapar, sonra kutulari
birlestirir. AYNI best_model.pt'yi kullanir; YENIDEN egitim gerektirmez.
Ciktisi yine kutu listesi; akisin geri kalani (OCR, renk, yuz, zamansal,
schema) DEGISMEZ.

BUTCE: SAHI bir kareyi ~10-15 dilime boler -> maliyet o kat artar. Bu yuzden:
  * mode=region : SAHI'yi yalnizca verilen bolgeye (arac/plaka crop) uygula.
  * trigger     : always | small_object | plate_unreadable | qod_active
SAHI pip paketi offline calisir (lokal .pt), Docker/internet-kapali ile uyumlu.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.detection.detector import Detection
from src.utils import clamp_box, get_logger

LOG = get_logger()


class SlicedDetector:
    def __init__(self, cfg: Dict[str, Any], weights_path: Optional[str]) -> None:
        s = cfg.get("sahi", {}) or {}
        self.enabled: bool = bool(s.get("enabled", False))
        self.mode: str = str(s.get("mode", "region")).lower()
        self.trigger: str = str(s.get("trigger", "plate_unreadable")).lower()
        self.slice_size: int = int(s.get("slice_size", 256))
        self.overlap: float = float(s.get("overlap_ratio", 0.2))
        self.conf: float = float(s.get("conf_threshold", 0.3))
        self.small_ratio: float = float(s.get("small_object_area_ratio", 0.01))
        self.weights_path = weights_path
        self._model = None
        self._names: Dict[int, str] = {}
        if self.enabled and self.mode != "off":
            self._load()

    def _load(self) -> None:
        if not self.weights_path:
            LOG.warning("SAHI: agirlik yok; dilimli cikarim devre disi.")
            self.enabled = False
            return
        try:
            from sahi import AutoDetectionModel  # lazy
            self._model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self.weights_path,
                confidence_threshold=self.conf,
                device="cuda",  # T4 yoksa sahi cpu'ya duser
            )
            try:
                self._names = {int(k): str(v) for k, v in
                               self._model.model.names.items()}
            except Exception:
                self._names = {}
            LOG.info("SAHI yuklendi (mode=%s, trigger=%s).", self.mode, self.trigger)
        except Exception as exc:
            LOG.error("SAHI yuklenemedi (%s); dilimli cikarim kapali.", exc)
            self.enabled = False
            self._model = None

    @property
    def ready(self) -> bool:
        return self.enabled and self._model is not None

    def should_trigger(self, *, plate_unreadable: bool = False,
                       small_object: bool = False, qod_active: bool = False
                       ) -> bool:
        """Config trigger'ina gore SAHI calistirilsin mi?"""
        if not self.ready:
            return False
        if self.trigger == "always":
            return True
        if self.trigger == "plate_unreadable":
            return plate_unreadable
        if self.trigger == "small_object":
            return small_object
        if self.trigger == "qod_active":
            return qod_active
        return False

    def predict_region(self, frame_bgr: np.ndarray,
                       region: Optional[Tuple[float, float, float, float]] = None
                       ) -> List[Detection]:
        """Bolgeye (yoksa tum kareye) SAHI dilimli cikarim; kutular KARE
        koordinatlarinda doner."""
        if not self.ready:
            return []
        h, w = frame_bgr.shape[:2]
        if region is not None and self.mode == "region":
            x1, y1, x2, y2 = clamp_box(region, w, h)
        else:
            x1, y1, x2, y2 = 0, 0, w, h
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        try:
            from sahi.predict import get_sliced_prediction
            result = get_sliced_prediction(
                crop,
                self._model,
                slice_height=self.slice_size,
                slice_width=self.slice_size,
                overlap_height_ratio=self.overlap,
                overlap_width_ratio=self.overlap,
                verbose=0,
            )
        except Exception as exc:
            LOG.debug("SAHI cikarim hatasi: %s", exc)
            return []

        dets: List[Detection] = []
        for op in getattr(result, "object_prediction_list", []) or []:
            bb = op.bbox  # minx, miny, maxx, maxy (crop koordinatlari)
            box = (bb.minx + x1, bb.miny + y1, bb.maxx + x1, bb.maxy + y1)
            cid = int(getattr(op.category, "id", -1))
            cname = str(getattr(op.category, "name", self._names.get(cid, str(cid))))
            score = float(getattr(op.score, "value", 0.0))
            dets.append(Detection(cid, cname, score, box))
        return dets
