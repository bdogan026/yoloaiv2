"""objects.py — "nesneler" kategorisi: teknocan, bilgisayar.

YOLO ciktisindan ilgili nesne siniflarini secip ham olay adaylari uretir
(kategori="nesneler"). Zaman damgalama ve debounce temporal.py'de yapilir.

NOT: teknocan yarismaya ozel bir nesnedir; egitim verisi acik kaynak +
sentetik/toplanan orneklerle olusturulur (bkz. training/ ve README).
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.detection.detector import Detection
from src.schema import KATEGORI_NESNELER, NESNELER_ETIKETLERI


class ObjectDetector:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        det = cfg.get("detection", {}) or {}
        # model sinif adi -> sema etiketi
        self.object_map: Dict[str, str] = det.get("object_classes", {}) or {}

    def detect(self, dets: List[Detection], t: float = 0.0) -> List[Dict[str, Any]]:
        """Karedeki nesne tespitlerini ham olay adaylarina cevirir."""
        events: List[Dict[str, Any]] = []
        for d in dets:
            etiket = self.object_map.get(d.cls_name)
            if etiket and etiket in NESNELER_ETIKETLERI:
                events.append({
                    "kategori": KATEGORI_NESNELER,
                    "etiket": etiket,
                    "confidence_score": d.conf,
                    "zaman_saniye": t,
                    "box": d.box,
                })
        return events
