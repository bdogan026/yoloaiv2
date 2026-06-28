"""passengers.py — "yolcular" kategorisi: on_koltuk, arka_koltuk_1/2.

Kisi (person) tespitlerini arac/kabin ROI'sine gore koltuk bolgesine esler.
ROI icinde normalize konum:
  * y < front_y_max            -> on bolge -> on_koltuk
  * aksi halde (arka bolge):
       x < back_left_x_max     -> arka_koltuk_1 (sol)
       degilse                 -> arka_koltuk_2 (sag)

Esikler config'den; ROI olarak ana arac kutusu kullanilir (yoksa tam kare).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.detection.detector import Detection
from src.schema import KATEGORI_YOLCULAR, YOLCULAR_ETIKETLERI
from src.utils import bbox_area, bbox_center


class PassengerMapper:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        p = cfg.get("passengers", {}) or {}
        self.enabled: bool = bool(p.get("enabled", True))
        self.front_y_max: float = float(p.get("front_y_max", 0.55))
        self.back_left_x_max: float = float(p.get("back_left_x_max", 0.5))
        self.min_area_ratio: float = float(p.get("min_person_area_ratio", 0.005))
        self.person_class: str = str((cfg.get("detection", {}) or {}).get(
            "person_class", "kisi"))

    def map_seats(self, dets: List[Detection],
                  roi: Optional[Tuple[float, float, float, float]],
                  frame_wh: Tuple[int, int],
                  t: float = 0.0) -> List[Dict[str, Any]]:
        """Karedeki kisileri koltuk etiketine cevirir (olay adaylari)."""
        if not self.enabled:
            return []
        w, h = frame_wh
        if roi is None:
            roi = (0.0, 0.0, float(w), float(h))
        rx1, ry1, rx2, ry2 = roi
        rw, rh = max(1.0, rx2 - rx1), max(1.0, ry2 - ry1)
        frame_area = float(w * h)

        events: List[Dict[str, Any]] = []
        for d in dets:
            if d.cls_name != self.person_class:
                continue
            if bbox_area(d.box) / frame_area < self.min_area_ratio:
                continue
            cx, cy = bbox_center(d.box)
            # ROI icine normalize
            nx = (cx - rx1) / rw
            ny = (cy - ry1) / rh
            if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
                continue  # ROI disindaki kisi (gecen yaya vb.) atlanir
            etiket = self._seat_label(nx, ny)
            if etiket in YOLCULAR_ETIKETLERI:
                events.append({
                    "kategori": KATEGORI_YOLCULAR,
                    "etiket": etiket,
                    "confidence_score": d.conf,
                    "zaman_saniye": t,
                    "box": d.box,
                })
        return events

    def _seat_label(self, nx: float, ny: float) -> str:
        if ny < self.front_y_max:
            return "on_koltuk"
        return "arka_koltuk_1" if nx < self.back_left_x_max else "arka_koltuk_2"
