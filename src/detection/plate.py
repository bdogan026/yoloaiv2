"""plate.py — Plaka bolgesi -> AYRI OCR -> normalize -> resmi regex dogrulama.

Akis: YOLO plaka kutusunu kirp -> (gerekirse buyut) -> OCR ile metin ->
schema.normalize_plate ile bosluk/ASCII normalize + resmi regex kontrolu.
Regex disindaki sonuc "tespit edilemedi" sayilir -> "". Esik altindaki OCR
sonuclari da bos gecilir.

OCR motoru EasyOCR (offline): model_storage_directory image'a gomulu klasore
isaret eder ve download_enabled=False ile calisma aninda indirme kapatilir.
"""
from __future__ import annotations

import os

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detection.detector import Detection
from src.schema import normalize_plate
from src.utils import clamp_box, get_logger, select_device

LOG = get_logger()


class PlateReader:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        det = cfg.get("detection", {}) or {}
        self.plate_class: str = str(det.get("plate_class", "plaka"))
        p = cfg.get("plate_ocr", {}) or {}
        self.enabled: bool = bool(p.get("enabled", True))
        self.engine: str = str(p.get("engine", "easyocr")).lower()
        self.languages: List[str] = list(p.get("languages", ["en"]))
        self.min_conf: float = float(p.get("min_confidence", 0.30))
        self.upscale_h: int = int(p.get("upscale_min_height", 48))
        self.regex: str = str(p.get("regex", ""))
        self.model_dir: Optional[str] = (cfg.get("paths", {}) or {}).get("ocr_model_dir")
        self.device = select_device(cfg)
        self._reader = None
        self._fast = None  # fast-plate-ocr (TR/global plaka ozel)
        self._init_engine()

        # En iyi plaka adayini kareler boyunca sakla
        self.best_text: str = ""
        self.best_conf: float = 0.0
        # Cok-kareli OYLAMA: gecerli plaka -> kumulatif conf (tek kotu kareye dayanikli)
        self._votes: Dict[str, float] = {}
        self._vote_conf: Dict[str, float] = {}

    def _restore_fast_cache(self) -> None:
        """fast-plate-ocr modelini weights'ten ~/.cache'e geri yukler (offline)."""
        try:
            import shutil
            wd = (self.cfg.get("paths", {}) or {}).get("weights_dir", "/app/weights")
            src = os.path.join(wd, "fast_plate_ocr")
            if os.path.isdir(src):
                dst = os.path.expanduser("~/.cache/fast-plate-ocr")
                os.makedirs(dst, exist_ok=True)
                shutil.copytree(src, dst, dirs_exist_ok=True)
        except Exception as exc:
            LOG.debug("fast-plate cache geri-yukleme: %s", exc)

    def _init_engine(self) -> None:
        if not self.enabled:
            LOG.info("Plaka OCR config'te kapali; plaka okunmayacak (hafif yol).")
            return
        if self.engine == "fast_plate_ocr":
            try:
                from fast_plate_ocr import LicensePlateRecognizer
                model = str((self.cfg.get("plate_ocr", {}) or {}).get(
                    "fast_model", "cct-xs-v1-global-model"))
                self._restore_fast_cache()   # OFFLINE: weights/fast_plate_ocr -> ~/.cache
                self._fast = LicensePlateRecognizer(model)
                LOG.info("fast-plate-ocr yuklendi: %s", model)
                return
            except Exception as exc:
                LOG.error("fast-plate-ocr yuklenemedi (%s); EasyOCR'a dusulecek.", exc)
                self._fast = None
        elif self.engine != "easyocr":
            LOG.warning("Bilinmeyen OCR motoru '%s'; EasyOCR denenecek.", self.engine)
        try:
            import easyocr  # lazy import
            self._reader = easyocr.Reader(
                self.languages,
                gpu=(self.device == "cuda"),
                model_storage_directory=self.model_dir,
                download_enabled=False,   # OFFLINE: calisma aninda indirme YOK
                verbose=False,
            )
            LOG.info("OCR (EasyOCR) yuklendi | dir=%s | gpu=%s",
                     self.model_dir, self.device == "cuda")
        except Exception as exc:  # gurbuzluk: OCR yoksa plaka bos kalir, cokme yok
            LOG.error("OCR yuklenemedi (%s); plaka okunamayacak.", exc)
            self._reader = None

    @property
    def ready(self) -> bool:
        return self._reader is not None or self._fast is not None

    def plate_detections(self, dets: List[Detection]) -> List[Detection]:
        return [d for d in dets if d.cls_name == self.plate_class]

    def read_from_frame(self, frame_bgr: np.ndarray, dets: List[Detection]
                        ) -> Tuple[str, float]:
        """Karedeki plaka kutu(lar)indan en guvenilir gecerli plakayi oku.

        Doner: (gecerli_plaka_or_bos, conf). En iyi sonucu iceride biriktirir.
        """
        plates = self.plate_detections(dets)
        if not plates or not self.ready:
            return "", 0.0
        # En buyuk/guvenilir plaka kutusunu sec
        plates.sort(key=lambda d: d.conf, reverse=True)
        h, w = frame_bgr.shape[:2]
        for pd in plates:
            x1, y1, x2, y2 = clamp_box(pd.box, w, h)
            crop = frame_bgr[y1:y2, x1:x2]
            text, conf = self._ocr_crop(crop)
            valid = normalize_plate(text, self.regex)
            if valid and conf >= self.min_conf:
                self._votes[valid] = self._votes.get(valid, 0.0) + conf
                self._vote_conf[valid] = max(self._vote_conf.get(valid, 0.0), conf)
                if conf > self.best_conf:
                    self.best_text, self.best_conf = valid, conf
                return valid, conf
        return "", 0.0

    def read_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Verilen plaka kirpintisini oku (SAHI/bolgesel yol icin)."""
        text, conf = self._ocr_crop(crop_bgr)
        valid = normalize_plate(text, self.regex)
        if valid and conf >= self.min_conf:
            self._votes[valid] = self._votes.get(valid, 0.0) + conf
            self._vote_conf[valid] = max(self._vote_conf.get(valid, 0.0), conf)
            if conf > self.best_conf:
                self.best_text, self.best_conf = valid, conf
            return valid, conf
        return "", 0.0

    def _fast_ocr(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """fast-plate-ocr ile oku (TR/global plaka; EasyOCR'dan dogru)."""
        try:
            out = self._fast.run(crop_bgr)
        except Exception as exc:
            LOG.debug("fast-plate-ocr hatasi: %s", exc)
            return "", 0.0

        def _extract(o):
            if o is None:
                return ""
            if hasattr(o, "plate"):
                return str(o.plate)
            if isinstance(o, str):
                return o
            if isinstance(o, (list, tuple)):
                for x in o:
                    s = _extract(x)
                    if s:
                        return s
            return ""
        text = _extract(out)
        return (text, 0.85) if text else ("", 0.0)

    def _ocr_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        if crop_bgr is None or crop_bgr.size == 0 or not self.ready:
            return "", 0.0
        if self._fast is not None:                 # fast-plate-ocr (kendi on-isleme)
            return self._fast_ocr(crop_bgr)
        img = self._prep(crop_bgr)
        try:
            # detail=1 -> [ (bbox, text, conf), ... ]
            res = self._reader.readtext(img, detail=1, paragraph=False,
                allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        except Exception as exc:
            LOG.debug("OCR readtext hatasi: %s", exc)
            return "", 0.0
        if not res:
            return "", 0.0
        # Plaka tek satir: tum parcalari birlestir, conf'lari ortala
        texts, confs = [], []
        for item in res:
            try:
                _, t, cf = item
            except Exception:
                continue
            texts.append(str(t))
            confs.append(float(cf))
        if not texts:
            return "", 0.0
        text = "".join(texts)
        conf = float(np.mean(confs)) if confs else 0.0
        return text, conf

    def _prep(self, crop_bgr: np.ndarray) -> np.ndarray:
        """OCR oncesi kirpintiyi buyut + gri + kontrast."""
        h = crop_bgr.shape[0]
        if h < self.upscale_h and h > 0:
            scale = self.upscale_h / h
            crop_bgr = cv2.resize(crop_bgr, (0, 0), fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        # CLAHE (kontrast) + unsharp (keskinlik). Eski bilateralFilter rakam
        # bosluklarini kapatiyordu (3->0, 8->0); bu kenarlari KORUR/keskinlestirir
        # -> dogru okuma (komite plakasi: bilateral=04TC0532, bu=34TC8532).
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
        return gray

    def best(self) -> Tuple[str, float]:
        # Oylama: en cok kumulatif conf alan plaka (tek kotu kare kazanmasin)
        if self._votes:
            plate = max(self._votes, key=self._votes.get)
            return plate, self._vote_conf.get(plate, self.best_conf)
        return self.best_text, self.best_conf
