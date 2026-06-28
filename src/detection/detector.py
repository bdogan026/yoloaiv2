"""detector.py — Paylasilan YOLO cikarim sarmalayicisi.

Tek best_model.pt bir kez yuklenir; her kare icin TEK YOLO gecisi yapilir ve
ciktilar tum yorumlayici modullere (vehicle/plate/objects/passengers/seatbelt)
dagitilir. Bu, "Tier-1/Tier-2 ayni agirlik; kazanc buyuk modelden degil"
tasariminin (bkz. README) cekirdegidir ve 10 dk butcesini korur.

Offline: agirlik LOKAL .pt yolundan yuklenir; ultralytics auto-download env ile
kapatilir (utils.apply_offline_env). Hicbir indirme yapilmaz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils import (first_existing_path, get_logger, select_device,
                       use_half)

LOG = get_logger()


@dataclass
class Detection:
    """Tek bir tespit kutusu (kare uzayinda)."""
    cls_id: int
    cls_name: str
    conf: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    track_id: Optional[int] = None          # BoTSORT track ID (takip aktifse)


class YOLODetector:
    """Ultralytics YOLO sarmalayicisi (FP16, batch, lokal agirlik)."""

    def __init__(self, cfg: Dict[str, Any], weights_path: Optional[str] = None) -> None:
        self.cfg = cfg
        det = cfg.get("detection", {}) or {}
        perf = cfg.get("performance", {}) or {}
        self.conf = float(det.get("conf_threshold", 0.35))
        self.iou = float(det.get("iou_threshold", 0.6))
        self.max_det = int(det.get("max_det", 100))
        self.imgsz = int(perf.get("imgsz", 640))
        self.device = select_device(cfg)
        self.half = use_half(cfg, self.device)

        trk = cfg.get("tracking", {}) or {}
        self.tracking_enabled = bool(trk.get("enabled", False))
        self.tracker = str(trk.get("tracker", "botsort.yaml"))

        # Agirlik yolu: parametre > config aday yollari
        if weights_path is None:
            cands = (cfg.get("paths", {}) or {}).get("detector_weights_candidates", [])
            weights_path = first_existing_path(cands)
        self.weights_path = weights_path
        self.model = None
        self.names: Dict[int, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.weights_path:
            LOG.warning("Detektor agirligi bulunamadi; tespitler bos donecek "
                        "(surec cokmeyecek).")
            return
        try:
            from ultralytics import YOLO  # lazy import
            self.model = YOLO(self.weights_path)
            # model.names: {id: name}
            raw = getattr(self.model, "names", {}) or {}
            self.names = {int(k): str(v) for k, v in raw.items()}
            LOG.info("Detektor yuklendi: %s | device=%s half=%s | siniflar=%s",
                     self.weights_path, self.device, self.half,
                     list(self.names.values()))
        except Exception as exc:  # gurbuzluk: yuklenemese de cokme yok
            LOG.error("Detektor yuklenemedi (%s); bos tespitle devam.", exc)
            self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Kare listesine batch cikarim. Her kare icin Detection listesi doner."""
        if not self.loaded or not frames:
            return [[] for _ in frames]
        try:
            results = self.model.predict(
                frames,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                half=self.half,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            LOG.error("Batch cikarim hatasi (%s); bu batch bos.", exc)
            return [[] for _ in frames]
        return [self._parse(r) for r in results]

    def predict_one(self, frame: np.ndarray) -> List[Detection]:
        """Tek kare/crop cikarimi (SAHI/bolgesel tetikleme icin)."""
        out = self.predict_batch([frame])
        return out[0] if out else []

    def track(self, frame: np.ndarray) -> List[Detection]:
        """Tek kare BoTSORT takipli cikarim (STATEFUL/ardisik; persist=True).

        Detection.track_id alanini doldurur. Batch DESTEKLEMEZ (takip kareleri
        sirayla bekler). Track kurulamamis kutularda track_id None kalir.
        """
        if not self.loaded:
            return []
        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                half=self.half,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            LOG.error("Takipli cikarim hatasi (%s); bu kare bos.", exc)
            return []
        return self._parse(results[0]) if results else []

    def _parse(self, result: Any) -> List[Detection]:
        dets: List[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return dets
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            return dets
        # Takip ID'leri (varsa); track() sonrasi boxes.id dolu olur
        ids = None
        try:
            if getattr(boxes, "id", None) is not None:
                ids = boxes.id.cpu().numpy().astype(int)
        except Exception:
            ids = None
        for i, ((x1, y1, x2, y2), c, k) in enumerate(zip(xyxy, confs, clss)):
            name = self.names.get(int(k), str(int(k)))
            tid = int(ids[i]) if ids is not None and i < len(ids) else None
            dets.append(Detection(int(k), name, float(c),
                                  (float(x1), float(y1), float(x2), float(y2)),
                                  track_id=tid))
        return dets
