"""seatbelt.py — emniyet_kemeri_ihlali (VIDEO-SEVIYE karar).

Kemer cam-arkasindan zor tespit edilir; eski kare-bazli "kemer yok -> ihlal"
tek kare kacirinca FALSE POSITIVE veriyordu (kemer TAKAN surucude bile; AI
gunduz testinde dogrulandi). Bunun yerine TUM video boyunca kemer-gorunme
oranina bakilir: kemer karelerin yeterli kisminda goruldu -> ihlal YOK;
neredeyse hic gorulmedi -> ihlal.

Sema (config detection.seatbelt):
  * present_classes : kemer TAKILI sinifi (orn. "emniyet_kemeri")
  * absent_classes  : (ops.) kemer TAKILMAMIS sinifi -> dogrudan ihlal kaniti
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.detection.detector import Detection
from src.schema import KATEGORI_SOFOR_EYLEMI


def _center_in(box, region) -> bool:
    """Kutu MERKEZI bolge icinde mi (kucuk kemer kutusu + buyuk driver_box'ta
    IoU yanlis; containment dogru olcut)."""
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


class SeatbeltChecker:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        s = (cfg.get("detection", {}) or {}).get("seatbelt", {}) or {}
        self.present: List[str] = list(s.get("present_classes", []) or [])
        self.absent: List[str] = list(s.get("absent_classes", []) or [])
        self.present_frac: float = float(s.get("present_fraction_no_violation", 0.10))
        self.absent_frac: float = float(s.get("absent_fraction_violation", 0.20))
        self.min_driver_frames: int = int(s.get("min_driver_frames", 3))
        self._driver_n = 0
        self._belt_n = 0
        self._absent_n = 0
        self._absent_conf = 0.0
        self._first_t: Optional[float] = None

    def update(self, dets: List[Detection],
               driver_box: Optional[Tuple[float, float, float, float]],
               t: float = 0.0) -> None:
        """Her karede surucu/kemer durumunu biriktirir (olay URETMEZ)."""
        if driver_box is None:
            return
        self._driver_n += 1
        if self._first_t is None:
            self._first_t = float(t)
        if any(d.cls_name in self.present and _center_in(d.box, driver_box)
               for d in dets):
            self._belt_n += 1
        for d in dets:
            if d.cls_name in self.absent and _center_in(d.box, driver_box):
                self._absent_n += 1
                self._absent_conf = max(self._absent_conf, d.conf)
                break

    def summarize(self) -> Optional[Dict[str, Any]]:
        """Video sonunda TEK emniyet_kemeri_ihlali karari (yoksa None)."""
        if self._driver_n < self.min_driver_frames:
            return None
        belt_frac = self._belt_n / self._driver_n
        absent_frac = self._absent_n / self._driver_n
        t = float(self._first_t or 0.0)
        # Net "takilmamis" sinifi yeterince goruldu -> ihlal (yuksek guven)
        if self.absent and absent_frac >= self.absent_frac:
            return {"kategori": KATEGORI_SOFOR_EYLEMI,
                    "etiket": "emniyet_kemeri_ihlali",
                    "confidence_score": float(max(0.5, self._absent_conf)),
                    "zaman_saniye": t}
        # Kemer neredeyse HIC gorulmedi -> ihlal (orta-dusuk guven).
        # Kemer karelerin >= present_frac kisminda goruldu -> TAKILI say -> ihlal YOK.
        if belt_frac < self.present_frac:
            return {"kategori": KATEGORI_SOFOR_EYLEMI,
                    "etiket": "emniyet_kemeri_ihlali",
                    "confidence_score": 0.5,
                    "zaman_saniye": t}
        return None
