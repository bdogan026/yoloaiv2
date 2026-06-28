"""preprocessing.py — Adaptif, gecikme odakli on isleme.

Bos akis:
  CLAHE (HER karede, dusuk gecikme, ana iyilestirme) +
  sis-yogunlugu skoru (ucuz, dark-channel tabanli) +
  DCP dehazing (kosullu: adaptive / qod_gated / always / off) + histerezis.

DCP sifirdan OpenCV/numpy ile: dark channel -> atmospheric light ->
transmission map -> guided filter -> restorasyon.

Her adim bagimsiz acilip kapanabilir (ablation/baseline icin). Pipeline her
karede hangi adimlarin calistigini ve toplam ms'yi loglar (bkz. pipeline.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


# =============================================================================
# CLAHE — her karede calisan dusuk gecikmeli iyilestirme
# =============================================================================
class Clahe:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        c = (cfg.get("preprocessing", {}) or {}).get("clahe", {}) or {}
        self.enabled: bool = bool(c.get("enabled", True))
        clip = float(c.get("clip_limit", 2.5))
        grid = tuple(c.get("tile_grid_size", [8, 8]))
        self.color_space: str = str(c.get("color_space", "lab")).lower()
        self._clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame_bgr
        if self.color_space == "yuv":
            conv, back, idx = cv2.COLOR_BGR2YUV, cv2.COLOR_YUV2BGR, 0
        else:  # lab
            conv, back, idx = cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR, 0
        img = cv2.cvtColor(frame_bgr, conv)
        ch = list(cv2.split(img))
        ch[idx] = self._clahe.apply(ch[idx])
        return cv2.cvtColor(cv2.merge(ch), back)


# =============================================================================
# Sis-yogunlugu skoru — ucuz, dark-channel tabanli [0..1]
# =============================================================================
def fog_score(frame_bgr: np.ndarray, cfg: Dict[str, Any]) -> float:
    """Hizli sis yogunlugu tahmini. Yuksek skor = yogun sis/dusuk kontrast.

    Dark channel'in ortalamasi yuksekse (parlak/dumanli) sis olasidir; ayrica
    dusuk global kontrast da sisi gosterir. Skoru kucuk kopyada hesaplariz.
    """
    fcfg = (cfg.get("preprocessing", {}) or {}).get("fog", {}) or {}
    scale = float(fcfg.get("score_downscale", 0.25))
    patch = int(fcfg.get("dark_channel_patch", 15))
    small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else frame_bgr
    dc = _dark_channel(small, max(3, patch // 2 | 1))
    dc_mean = float(dc.mean()) / 255.0  # 0..1; sis arttikca artar
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std()) / 128.0  # 0..~1; sis arttikca DUSER
    contrast = min(1.0, contrast)
    # Birlesik skor: yuksek dark-channel + dusuk kontrast -> yuksek sis
    score = 0.6 * dc_mean + 0.4 * (1.0 - contrast)
    return float(max(0.0, min(1.0, score)))


# =============================================================================
# DCP (Dark Channel Prior) dehazing — sifirdan
# =============================================================================
def _dark_channel(img_bgr: np.ndarray, patch: int) -> np.ndarray:
    """Min over channels, sonra min filter (erosion) ile dark channel."""
    min_ch = np.min(img_bgr, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(min_ch, kernel)


def _atmospheric_light(img_bgr: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Dark channel'in en parlak %0.1'inden atmosferik isigi tahmin et."""
    h, w = dark.shape
    n = max(1, int(0.001 * h * w))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    flat = img_bgr.reshape(-1, 3).astype(np.float64)
    A = flat[idx].max(axis=0)  # her kanal icin en parlak
    return A.reshape(1, 1, 3)


def _guided_filter(guide_gray: np.ndarray, src: np.ndarray,
                   radius: int, eps: float) -> np.ndarray:
    """Gri rehber goruntu ile guided filter (transmission map yumusatma)."""
    guide = guide_gray.astype(np.float32) / 255.0
    p = src.astype(np.float32)
    win = (radius, radius)
    mean_I = cv2.boxFilter(guide, cv2.CV_32F, win)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, win)
    mean_Ip = cv2.boxFilter(guide * p, cv2.CV_32F, win)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, win)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, win)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, win)
    return mean_a * guide + mean_b


def dehaze_dcp(frame_bgr: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Tek kare DCP dehazing. omega, t0, pencere config'den."""
    d = (cfg.get("preprocessing", {}) or {}).get("dcp", {}) or {}
    omega = float(d.get("omega", 0.95))
    t0 = float(d.get("t0", 0.1))
    patch = int(d.get("patch_size", 15))
    radius = int(d.get("guided_radius", 40))
    eps = float(d.get("guided_eps", 1e-3))

    img = frame_bgr.astype(np.float64)
    dark = _dark_channel(frame_bgr, patch)
    A = _atmospheric_light(frame_bgr, dark)
    A = np.clip(A, 1e-6, 255.0)

    # transmission tahmini
    norm = img / A
    trans = 1.0 - omega * _dark_channel((norm * 255).astype(np.uint8), patch) / 255.0
    # guided filter ile yumusat
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    trans = _guided_filter(gray, trans.astype(np.float32), radius, eps)
    trans = np.clip(trans, t0, 1.0)[:, :, np.newaxis]

    recovered = (img - A) / trans + A
    return np.clip(recovered, 0, 255).astype(np.uint8)


# =============================================================================
# Histerezis denetleyici — DCP kare kare acilip kapanmasin
# =============================================================================
@dataclass
class _Hysteresis:
    on_frames: int
    off_frames: int
    _active: bool = False
    _above: int = 0
    _below: int = 0

    def update(self, condition: bool) -> bool:
        if condition:
            self._above += 1
            self._below = 0
        else:
            self._below += 1
            self._above = 0
        if not self._active and self._above >= self.on_frames:
            self._active = True
        elif self._active and self._below >= self.off_frames:
            self._active = False
        return self._active


# =============================================================================
# Orkestrasyon: kalite/QoD gecidi + CLAHE + (kosullu) DCP
# =============================================================================
class Preprocessor:
    """Kare basi on isleme orkestrasyonu.

    mode:
      adaptive  -> FTR: sis skoru esigi asarsa DCP (histerezisli)
      qod_gated -> canli: DCP yalnizca qod_active=True iken
      always    -> her karede DCP
      off       -> DCP kapali (sadece CLAHE)
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.clahe = Clahe(cfg)
        d = (cfg.get("preprocessing", {}) or {}).get("dcp", {}) or {}
        self.mode: str = str(d.get("mode", "adaptive")).lower()
        self.fog_threshold: float = float(d.get("fog_score_threshold", 0.45))
        self._hyst = _Hysteresis(
            on_frames=int(d.get("hysteresis_on_frames", 3)),
            off_frames=int(d.get("hysteresis_off_frames", 5)),
        )

    def process(self, frame_bgr: np.ndarray, *, qod_active: bool = False
                ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Isle ve hangi adimlarin calistigini bildiren info doner."""
        info: Dict[str, Any] = {"steps": [], "fog_score": None, "dcp": False}
        out = frame_bgr

        # 1) CLAHE her zaman (etkinse)
        if self.clahe.enabled:
            out = self.clahe.apply(out)
            info["steps"].append("clahe")

        # 2) DCP gecidi
        run_dcp = False
        if self.mode == "always":
            run_dcp = True
        elif self.mode == "off":
            run_dcp = False
        elif self.mode == "qod_gated":
            # canli: yalnizca QoD aktifken (kritik durum) DCP
            run_dcp = bool(qod_active)
            # QoD aktifken yine de gereksiz harcamamak icin sis skoruna bakilir
            if run_dcp:
                score = fog_score(out, self.cfg)
                info["fog_score"] = round(score, 3)
                run_dcp = self._hyst.update(score >= self.fog_threshold) or True
        elif self.mode == "adaptive":
            score = fog_score(out, self.cfg)
            info["fog_score"] = round(score, 3)
            run_dcp = self._hyst.update(score >= self.fog_threshold)

        if run_dcp:
            out = dehaze_dcp(out, self.cfg)
            info["dcp"] = True
            info["steps"].append("dcp")

        return out, info


if __name__ == "__main__":
    # Tek basina hizli test (sentetik sisli kare)
    import yaml as _yaml
    import os as _os
    cfg_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                             "config", "config.yaml")
    with open(cfg_path) as _f:
        _cfg = _yaml.safe_load(_f)
    img = (np.ones((240, 320, 3), np.uint8) * 200)  # parlak/dumanli sahte kare
    pre = Preprocessor(_cfg)
    out, info = pre.process(img)
    print("info:", info, "out shape:", out.shape)
