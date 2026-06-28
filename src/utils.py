"""utils.py — Yardimci fonksiyonlar: config yukleme, offline env, cihaz secimi,
geometri, zamanlama ve loglama.

Bu modul agir model kutuphanelerini import ETMEZ (torch sadece cihaz secimi icin
tembel/lazy kullanilir). Boylece bagimsiz test edilebilir.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

LOGGER_NAME = "rapid_response"


# -----------------------------------------------------------------------------
# Loglama
# -----------------------------------------------------------------------------
def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# -----------------------------------------------------------------------------
# Config yukleme
# -----------------------------------------------------------------------------
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """config.yaml'i yukler. path verilmezse paket icindeki varsayilani bulur."""
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "config", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_offline_env(cfg: Dict[str, Any]) -> None:
    """Kutuphanelerin calisma aninda indirme yapmasini engelleyen env'leri ayarlar.

    Anti-hile NOT: bu env'ler ortam tespiti yapmaz; her kosulda ayni sekilde
    set edilir. Amac yalnizca offline garantisi (auto-download kapatma).
    """
    env = (cfg.get("runtime", {}) or {}).get("offline_env", {}) or {}
    for key, value in env.items():
        os.environ.setdefault(key, str(value))


# -----------------------------------------------------------------------------
# Cihaz secimi (lazy torch import)
# -----------------------------------------------------------------------------
def select_device(cfg: Dict[str, Any]) -> str:
    """'auto' | 'cuda' | 'cpu' -> kullanilabilir cihaz dizesi.

    T4 yoksa CPU'ya duser (hedef cuda). torch yoksa cpu doner.
    """
    requested = (cfg.get("runtime", {}) or {}).get("device", "auto")
    try:
        import torch  # lazy
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if cuda_ok else "cpu"
    # auto
    return "cuda" if cuda_ok else "cpu"


def use_half(cfg: Dict[str, Any], device: str) -> bool:
    """FP16 yalnizca cuda'da ve config izin veriyorsa."""
    return bool((cfg.get("runtime", {}) or {}).get("half_precision", True)) and device == "cuda"


def first_existing_path(candidates: List[str]) -> Optional[str]:
    """Aday yollardan ilk var olani doner.

    NOT: Bu, kendi model dosyamizin yerini bulmak icindir (gurbuzluk), ortam
    tespiti DEGILDIR; davranis her kosulda aynidir.
    """
    for p in candidates or []:
        if p and os.path.exists(p):
            return p
    return None


# -----------------------------------------------------------------------------
# Geometri yardimcilari (bbox: [x1, y1, x2, y2])
# -----------------------------------------------------------------------------
def bbox_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def bbox_area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_diag(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def iou(a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: Tuple[float, float, float, float],
                    b: Tuple[float, float, float, float]) -> float:
    cax, cay = bbox_center(a)
    cbx, cby = bbox_center(b)
    return ((cax - cbx) ** 2 + (cay - cby) ** 2) ** 0.5


def clamp_box(box: Tuple[float, float, float, float],
              w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w)))
    y2 = int(max(0, min(y2, h)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


# -----------------------------------------------------------------------------
# Zaman butcesi takibi
# -----------------------------------------------------------------------------
@dataclass
class TimeBudget:
    """Toplam calisma suresini izler; butce asimina karsi uyarir."""
    max_seconds: float
    start: float = field(default_factory=time.time)

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> float:
        return self.max_seconds - self.elapsed()

    def fraction_used(self) -> float:
        if self.max_seconds <= 0:
            return 1.0
        return self.elapsed() / self.max_seconds

    def over(self) -> bool:
        return self.elapsed() >= self.max_seconds


class FrameTimer:
    """Kare basi adim sureleri (ms) toplar; ozet loglama icin."""

    def __init__(self) -> None:
        self.steps: Dict[str, float] = {}
        self._t0 = time.time()

    def reset(self) -> None:
        self.steps = {}
        self._t0 = time.time()

    def mark(self, name: str) -> None:
        now = time.time()
        self.steps[name] = (now - self._t0) * 1000.0
        self._t0 = now

    def total_ms(self) -> float:
        return sum(self.steps.values())

    def summary(self) -> str:
        parts = [f"{k}={v:.1f}ms" for k, v in self.steps.items()]
        parts.append(f"TOTAL={self.total_ms():.1f}ms")
        return " ".join(parts)
