"""actions.py — Birlesik sofor_eylemi tespiti (YOLO nesne + surucu kafa/agiz yakinligi).

Eslesmeler:
  * telefon nesnesi + kafa yakinligi -> telefonla_konusma
  * sigara nesnesi  + agiz yakinligi -> sigara_icme
  * sise/bardak     + agiz yakinligi -> su_icme

Yakinlik: nesne kutusu ile surucu KAFA bolgesi (driver_box ust kismi) merkezleri
arasindaki mesafe / kafa bolgesi kosegeni. Bu oran config esiginin altindaysa
ve min_frames sureli ise olay uretilir.

(emniyet_kemeri_ihlali ayri modulde: detection/seatbelt.py. slalom: temporal.py.)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.detection.detector import Detection
from src.schema import KATEGORI_SOFOR_EYLEMI
from src.utils import bbox_diag, center_distance


class ActionDetector:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        det = cfg.get("detection", {}) or {}
        # model sinif adi -> aksiyon etiketi
        self.obj_to_action: Dict[str, str] = {}
        action_map = det.get("action_object_classes", {}) or {}
        # config: {telefon: telefon, sigara: sigara, sise: sise}
        # nesne -> eylem eslemesi:
        self._defaults = {
            "telefon": "telefonla_konusma",
            "sigara": "sigara_icme",
            "sise": "su_icme",
        }
        for cls_name in action_map.values():
            if cls_name in self._defaults:
                self.obj_to_action[cls_name] = self._defaults[cls_name]

        a = cfg.get("actions", {}) or {}
        self.params: Dict[str, Dict[str, Any]] = {
            "telefonla_konusma": a.get("telefonla_konusma", {}) or {},
            "sigara_icme": a.get("sigara_icme", {}) or {},
            "su_icme": a.get("su_icme", {}) or {},
        }
        # sure sayaclari (eylem bazli)
        self._run: Dict[str, int] = {k: 0 for k in self.params}

    def detect(self, dets: List[Detection],
               driver_box: Optional[Tuple[float, float, float, float]],
               t_seconds: float) -> List[Dict[str, Any]]:
        """Karedeki birlesik eylem olay adaylari."""
        events: List[Dict[str, Any]] = []
        if driver_box is None:
            for k in self._run:
                self._run[k] = 0
            return events

        head_box = self._head_region(driver_box)
        head_diag = max(1.0, bbox_diag(head_box))

        # Bu karede hangi eylemler aday? (her eylem icin en yakin nesne)
        active: Dict[str, float] = {}  # action -> conf
        for d in dets:
            action = self.obj_to_action.get(d.cls_name)
            if not action:
                continue
            ratio = center_distance(d.box, head_box) / head_diag
            thr = float(self.params[action].get("proximity_ratio", 0.6))
            if ratio <= thr:
                # daha yakin/guvenli olani sec
                conf = d.conf * (1.0 - min(1.0, ratio / max(thr, 1e-6)) * 0.3)
                active[action] = max(active.get(action, 0.0), conf)

        # sure sayaclarini guncelle + esige ulasanlari uret
        for action in self._run:
            if action in active:
                self._run[action] += 1
            else:
                self._run[action] = 0
            min_frames = int(self.params[action].get("min_frames", 3))
            if self._run[action] >= min_frames:
                events.append({
                    "kategori": KATEGORI_SOFOR_EYLEMI,
                    "etiket": action,
                    "confidence_score": float(max(0.0, min(1.0, active[action]))),
                    "zaman_saniye": float(t_seconds),
                })
        return events

    @staticmethod
    def _head_region(driver_box: Tuple[float, float, float, float]
                     ) -> Tuple[float, float, float, float]:
        """Surucu kutusunun ust ~%35'ini kafa/agiz bolgesi olarak al."""
        x1, y1, x2, y2 = driver_box
        h = y2 - y1
        return (x1, y1, x2, y1 + 0.35 * h)
