"""drowsiness_cnn.py — Yorgunluk/esneme yedek CNN'i (OFFLINE).

Gelistirme sari-1 maddesinin cozumu: yol kenari kamerasinda surucu yuzu uzak/
acili/cam ardinda olunca MediaPipe Face Landmarker yuz/landmark bulamayabilir.
Bu CNN, surucu crop'undan DOGRUDAN (landmark gerektirmeden) esneme/goz-kapali
durumunu kestirir ve MediaPipe basarisiz oldugunda devreye girer (yedek).

Neden offline? Yarisma kurali geregi calisma aninda internet KAPALI; agirlik
(drowsiness_cnn.pt) image'a gomulur, yalnizca lokal yuklenir, indirme yok.

Mimari: torchvision mobilenet_v3_small (3 sinif). Egitim: training/train_drowsiness.py
Siniflar (egitim bu sirayi kullanmali): ["normal", "esneme", "goz_kapali"]
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from src.utils import get_logger, select_device

LOG = get_logger()

DROWSY_CLASSES = ["normal", "esneme", "goz_kapali"]
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DrowsinessCNN:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        f = (cfg.get("face", {}) or {}).get("fallback_cnn", {}) or {}
        self.enabled = bool(f.get("enabled", False))
        self.input_size = int(f.get("input_size", 96))
        self.min_conf = float(f.get("min_confidence", 0.5))
        self.use_when = str(f.get("use_when", "mediapipe_failed"))
        self.weights_path: Optional[str] = (cfg.get("paths", {}) or {}).get(
            "drowsiness_cnn_weights")
        self.device = select_device(cfg)
        self.model = None
        if self.enabled:
            self._load()

    def _load(self) -> None:
        if not self.weights_path or not os.path.exists(self.weights_path):
            LOG.info("Yorgunluk yedek CNN agirligi yok; yedek devre disi (%s).",
                     self.weights_path)
            return
        try:
            import torch
            from torchvision.models import mobilenet_v3_small
            model = mobilenet_v3_small(weights=None,  # OFFLINE: indirme YOK
                                       num_classes=len(DROWSY_CLASSES))
            state = torch.load(self.weights_path, map_location=self.device)
            state = state.get("model", state) if isinstance(state, dict) else state
            model.load_state_dict(state)
            model.eval().to(self.device)
            self.model = model
            LOG.info("Yorgunluk yedek CNN yuklendi: %s (device=%s)",
                     self.weights_path, self.device)
        except Exception as exc:
            LOG.error("Yorgunluk yedek CNN yuklenemedi (%s); yedek devre disi.", exc)
            self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Surucu crop'u -> (etiket, guven). Hazir degilse ("", 0.0)."""
        if not self.ready or crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0
        try:
            import torch
            x = self._preprocess(crop_bgr)
            with torch.no_grad():
                logits = self.model(x.to(self.device))
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            idx = int(probs.argmax())
            return DROWSY_CLASSES[idx], float(probs[idx])
        except Exception as exc:
            LOG.debug("Yorgunluk CNN tahmin hatasi: %s", exc)
            return "", 0.0

    def _preprocess(self, crop_bgr: np.ndarray):
        import torch
        s = self.input_size
        img = cv2.resize(crop_bgr, (s, s), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        chw = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        return torch.from_numpy(chw).float()
