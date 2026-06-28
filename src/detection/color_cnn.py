"""color_cnn.py — Hafif arac rengi siniflandirma CNN'i (OFFLINE).

Neden offline? Yarisma kurali: calisma aninda internet KAPALI. Bu CNN'in
agirligi (color_cnn.pt) image'a gomulur ve YALNIZCA lokal yoldan yuklenir;
hicbir indirme yapilmaz (torchvision pretrained=False ile mimari kurulur,
agirlik lokalden gelir).

Mimari: torchvision mobilenet_v3_small (num_classes=9). Siniflar schema'daki
ARAC_RENKLERI sirasiyla BIREBIR (egitim de bu sirayi kullanmali).

Agirlik yoksa ready=False doner; cagiran taraf HSV'ye duser (vehicle.py).
Egitim: training/train_color.py.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from src.schema import ARAC_RENKLERI
from src.utils import get_logger, select_device

LOG = get_logger()

# Sinif sirasi schema ile birebir (egitim de bu listeyi kullanir)
COLOR_CLASSES = list(ARAC_RENKLERI)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ColorCNN:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        c = cfg.get("color", {}) or {}
        self.input_size = int(c.get("cnn_input_size", 64))
        self.min_conf = float(c.get("cnn_min_confidence", 0.5))
        self.weights_path: Optional[str] = (cfg.get("paths", {}) or {}).get(
            "color_cnn_weights")
        self.device = select_device(cfg)
        self.model = None
        self._load()

    def _load(self) -> None:
        if not self.weights_path or not os.path.exists(self.weights_path):
            LOG.info("Renk CNN agirligi yok; HSV'ye dusulecek (%s).",
                     self.weights_path)
            return
        try:
            import torch
            from torchvision.models import mobilenet_v3_small
            model = mobilenet_v3_small(weights=None,  # OFFLINE: pretrained indirme YOK
                                       num_classes=len(COLOR_CLASSES))
            state = torch.load(self.weights_path, map_location=self.device)
            state = state.get("model", state) if isinstance(state, dict) else state
            model.load_state_dict(state)
            model.eval().to(self.device)
            self.model = model
            LOG.info("Renk CNN yuklendi: %s (device=%s)", self.weights_path,
                     self.device)
        except Exception as exc:  # gurbuzluk: yuklenemese de HSV devreye girer
            LOG.error("Renk CNN yuklenemedi (%s); HSV'ye dusulecek.", exc)
            self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Arac kirpintisi -> (renk, guven). Hazir degilse ("", 0.0)."""
        if not self.ready or crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0
        try:
            import torch
            x = self._preprocess(crop_bgr)
            with torch.no_grad():
                logits = self.model(x.to(self.device))
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            idx = int(probs.argmax())
            conf = float(probs[idx])
            color = COLOR_CLASSES[idx]
            return (color, conf) if conf >= self.min_conf else ("", conf)
        except Exception as exc:
            LOG.debug("Renk CNN tahmin hatasi: %s", exc)
            return "", 0.0

    def _preprocess(self, crop_bgr: np.ndarray):
        import torch
        s = self.input_size
        img = cv2.resize(crop_bgr, (s, s), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        chw = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]  # 1,C,H,W
        return torch.from_numpy(chw).float()
