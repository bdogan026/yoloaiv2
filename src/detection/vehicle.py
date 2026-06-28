"""vehicle.py — Arac tespiti + govde tipi + renk + ana arac konsolidasyonu.

Takip (BoTSORT) ile: her arac track ID'si icin AYRI oy birikimi tutulur; video
sonunda en baskin track (en cok mevcudiyet = kumulatif alan) secilip TEK
arac_bilgisi uretilir. Bu, coklu arac gecisinde tip/plaka/renk karismasini onler.
Takip kapaliyken tum araclar tek kovaya (-1) dusurulur (eski davranis).

Renk: method=auto ise once CNN (color_cnn.pt, offline), yoksa/dusuk guvende HSV
merkez yontemi (agirliksiz, offline). method=cnn|hsv_centroid ile zorlanabilir.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detection.color_cnn import ColorCNN
from src.detection.tip_cnn import TipCNN
from src.detection.detector import Detection
from src.schema import ARAC_RENKLERI, ARAC_TIPLERI
from src.utils import bbox_area, clamp_box, get_logger

LOG = get_logger()


# =============================================================================
# Renk siniflandirma — HSV merkez tabanli (gomulu, agirliksiz) [HSV fallback]
# =============================================================================
def classify_color_hsv(crop_bgr: np.ndarray, cfg: Dict[str, Any]
                       ) -> Tuple[str, float]:
    """Arac kirpintisi -> (renk, guven). Izinli renk kumesinden secer (HSV)."""
    c = cfg.get("color", {}) or {}
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0

    h = crop_bgr.shape[0]
    top_ratio = float(c.get("crop_top_ratio", 0.55))
    crop = crop_bgr[: max(1, int(h * top_ratio)), :, :]
    crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(np.float32), hsv[..., 1], hsv[..., 2]

    min_sat = float(c.get("min_saturation_gray", 40))
    v_black = float(c.get("value_black_max", 60))
    v_white = float(c.get("value_white_min", 190))

    achroma_mask = S < min_sat
    achroma_ratio = float(achroma_mask.mean())

    if achroma_ratio > 0.6:
        v_med = float(np.median(V[achroma_mask])) if achroma_mask.any() else float(np.median(V))
        if v_med <= v_black:
            return "siyah", round(0.5 + 0.5 * achroma_ratio, 4)
        if v_med >= v_white:
            return "beyaz", round(0.5 + 0.5 * achroma_ratio, 4)
        return "gri", round(0.4 + 0.4 * achroma_ratio, 4)

    chroma_mask = ~achroma_mask
    if not chroma_mask.any():
        return "gri", 0.4
    hues = H[chroma_mask]
    vals = V[chroma_mask]
    hist = np.bincount(hues.astype(int), minlength=180).astype(np.float32)
    dom_hue = int(np.argmax(hist))
    conf = float(hist.max() / max(1.0, hist.sum()))

    color = _hue_to_color(dom_hue, float(np.median(vals)), c)
    if color not in ARAC_RENKLERI:
        color = "gri"
    return color, round(min(1.0, 0.4 + conf), 4)


def _hue_to_color(hue: int, v_med: float, c: Dict[str, Any]) -> str:
    ranges = c.get("hue_ranges", {}) or {}
    kahve_v = float(c.get("kahverengi_value_max", 150))

    def in_ranges(h: int, rs) -> bool:
        return any(lo <= h <= hi for lo, hi in rs)

    is_redish = in_ranges(hue, ranges.get("kirmizi", [])) or \
        in_ranges(hue, ranges.get("turuncu", []))
    if is_redish and v_med < kahve_v:
        return "kahverengi"
    for name, rs in ranges.items():
        if in_ranges(hue, rs):
            return name
    return "gri"


class ColorClassifier:
    """CNN (offline) + HSV fallback secici. method: auto | cnn | hsv_centroid."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.method = str((cfg.get("color", {}) or {}).get("method", "auto"))
        self.cnn: Optional[ColorCNN] = None
        if self.method in ("auto", "cnn"):
            self.cnn = ColorCNN(cfg)

    def classify(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        if self.method == "hsv_centroid":
            return classify_color_hsv(crop_bgr, self.cfg)
        if self.method == "cnn":
            return self.cnn.predict(crop_bgr) if self.cnn else ("", 0.0)
        # auto: CNN varsa ve guvenli -> CNN; aksi halde HSV
        if self.cnn and self.cnn.ready:
            color, conf = self.cnn.predict(crop_bgr)
            if color:
                return color, conf
        return classify_color_hsv(crop_bgr, self.cfg)


# Geriye uyumluluk: eski isim (HSV) — disaridan import edenler icin
def classify_color(crop_bgr: np.ndarray, cfg: Dict[str, Any]) -> Tuple[str, float]:
    return classify_color_hsv(crop_bgr, cfg)


# =============================================================================
# Arac analizoru + ana arac konsolidasyonu (per-track)
# =============================================================================
@dataclass
class _TrackAgg:
    type_votes: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    color_votes: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    type_conf_sum: float = 0.0
    color_conf_sum: float = 0.0
    n_type: int = 0
    n_color: int = 0
    area_sum: float = 0.0


class VehicleAnalyzer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        det = cfg.get("detection", {}) or {}
        self.type_map: Dict[str, str] = det.get("vehicle_types", {}) or {}
        vs = cfg.get("vehicle_summary", {}) or {}
        self.type_vote = str(vs.get("type_vote", "weighted"))
        self.color_vote = str(vs.get("color_vote", "weighted"))
        self.w = vs.get("weights", {"tip": 0.4, "plaka": 0.35, "renk": 0.25})
        self.color_clf = ColorClassifier(cfg)
        # Govde tipi CNN (offline). Agirlik yoksa ready=False -> detektor tip'i
        # (type_map) kullanilir; davranis degismez (geriye donuk uyumlu).
        self.tip_method = str((cfg.get("tip", {}) or {}).get("method", "auto"))
        self.tip_cnn: Optional[TipCNN] = (
            TipCNN(cfg) if self.tip_method in ("auto", "cnn") else None)
        self._tracks: Dict[int, _TrackAgg] = defaultdict(_TrackAgg)

    def vehicle_detections(self, dets: List[Detection]) -> List[Detection]:
        return [d for d in dets if d.cls_name in self.type_map]

    def main_vehicle(self, dets: List[Detection]) -> Optional[Detection]:
        """Bu karedeki ana arac: en buyuk alanli arac kutusu (per-frame)."""
        vehicles = self.vehicle_detections(dets)
        if not vehicles:
            return None
        return max(vehicles, key=lambda d: bbox_area(d.box))

    def update(self, frame_bgr: np.ndarray, dets: List[Detection]) -> None:
        """Karedeki tum araclarin tip/alan oyunu track bazli biriktirir; renk
        yalnizca per-frame en buyuk arac icin siniflandirilir (butce)."""
        vehicles = self.vehicle_detections(dets)
        if not vehicles:
            return
        for d in vehicles:
            key = d.track_id if d.track_id is not None else -1
            agg = self._tracks[key]
            agg.area_sum += bbox_area(d.box)
            tip = self.type_map.get(d.cls_name, "")
            if tip in ARAC_TIPLERI:
                weight = d.conf if self.type_vote == "weighted" else 1.0
                agg.type_votes[tip] += weight
                agg.type_conf_sum += d.conf
                agg.n_type += 1

        # Renk: per-frame en buyuk araci siniflandir ve track'ine yaz
        main = max(vehicles, key=lambda d: bbox_area(d.box))
        key = main.track_id if main.track_id is not None else -1
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = clamp_box(main.box, w, h)
        crop = frame_bgr[y1:y2, x1:x2]
        renk, cconf = self.color_clf.classify(crop)
        if renk in ARAC_RENKLERI:
            agg = self._tracks[key]
            weight = cconf if self.color_vote == "weighted" else 1.0
            agg.color_votes[renk] += weight
            agg.color_conf_sum += cconf
            agg.n_color += 1

        # Tip: ana arac crop'unu CNN ile siniflandir (hazirsa) -> guven-agirlikli
        # tip oyu. Detektor tip veriyorsa onu destekler; genel "arac" veriyorsa
        # tip'i bu tasir. tip_cnn yoksa (ready=False) hic etki etmez.
        if self.tip_cnn is not None and self.tip_cnn.ready:
            tip_label, tconf = self.tip_cnn.predict(crop)
            if tip_label in ARAC_TIPLERI:
                agg = self._tracks[key]
                agg.type_votes[tip_label] += tconf
                agg.type_conf_sum += tconf
                agg.n_type += 1

    def _dominant_track(self) -> Optional[int]:
        if not self._tracks:
            return None
        return max(self._tracks, key=lambda k: self._tracks[k].area_sum)

    def summarize(self, plate: str, plate_conf: float) -> Dict[str, Any]:
        """En baskin track'in oylarindan tek arac_bilgisi uret."""
        tid = self._dominant_track()
        if tid is None:
            return {"tip": "", "plaka": plate, "renk": "",
                    "confidence_score": float(max(0.0, min(1.0,
                        self.w.get("plaka", 0.35) * plate_conf * (1.0 if plate else 0.0))))}
        agg = self._tracks[tid]
        tip = max(agg.type_votes, key=agg.type_votes.get) if agg.type_votes else ""
        renk = max(agg.color_votes, key=agg.color_votes.get) if agg.color_votes else ""
        tip_conf = (agg.type_conf_sum / agg.n_type) if agg.n_type else 0.0
        renk_conf = (agg.color_conf_sum / agg.n_color) if agg.n_color else 0.0
        plate_present = 1.0 if plate else 0.0
        conf = (self.w.get("tip", 0.4) * tip_conf +
                self.w.get("plaka", 0.35) * (plate_conf * plate_present) +
                self.w.get("renk", 0.25) * renk_conf)
        return {
            "tip": tip,
            "plaka": plate,
            "renk": renk,
            "confidence_score": float(max(0.0, min(1.0, conf))),
        }
