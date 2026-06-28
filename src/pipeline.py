"""pipeline.py — Kare donguso + orkestrasyon (FTR ve canli icin ortak cekirdek).

Kare basina:
  kare -> on isleme (CLAHE + adaptif DCP) -> YOLO tespit (batch, FP16) ->
  (kirpilan bolgelerde) yuz/OCR/renk -> zamansal mantik -> olay uretimi
Sonunda tum video icin tek results.json (sema dogrulamali).

Performans butcesi (T4 / 10 dk): kare ornekleme + batch + FP16 + agir islerin
(OCR, MediaPipe, DCP, SAHI) yalnizca gerektiginde tetiklenmesi + butce asimina
karsi adaptif ornekleme.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.actions import ActionDetector
from src.detection.detector import Detection, YOLODetector
from src.detection.objects import ObjectDetector
from src.detection.passengers import PassengerMapper
from src.detection.plate import PlateReader
from src.detection.seatbelt import SeatbeltChecker
from src.detection.sliced import SlicedDetector
from src.detection.vehicle import VehicleAnalyzer
from src.face.driver_face import DriverFaceAnalyzer
from src.face.hand_phone import HandPhoneDetector
from src.preprocessing import Preprocessor
from src.schema import empty_results, validate_results
from src.temporal import EventAggregator, SlalomDetector
from src.utils import (TimeBudget, apply_offline_env, bbox_area, clamp_box,
                       get_logger, iou)

LOG = get_logger()


class Pipeline:
    def __init__(self, cfg: Dict[str, Any], weights_path: Optional[str] = None,
                 *, qod_active_provider=None) -> None:
        """qod_active_provider: canli modda QoD durumunu doner (FTR'de None)."""
        self.cfg = cfg
        apply_offline_env(cfg)

        self.detector = YOLODetector(cfg, weights_path)
        self.sliced = SlicedDetector(cfg, self.detector.weights_path)
        self.preproc = Preprocessor(cfg)
        self.vehicle = VehicleAnalyzer(cfg)
        self.plate = PlateReader(cfg)
        self.objects = ObjectDetector(cfg)
        self.passengers = PassengerMapper(cfg)
        self.seatbelt = SeatbeltChecker(cfg)
        self.actions = ActionDetector(cfg)
        self.face = DriverFaceAnalyzer(cfg)
        self.hand_phone = HandPhoneDetector(cfg)
        self.slalom = SlalomDetector(cfg)
        self.aggregator = EventAggregator(cfg)

        self.person_class = str((cfg.get("detection", {}) or {}).get(
            "person_class", "kisi"))
        self.plate_class = str((cfg.get("detection", {}) or {}).get(
            "plate_class", "plaka"))

        # SAHI butce korumasi (seyrek/sinirli tetikleme)
        sahi_cfg = cfg.get("sahi", {}) or {}
        self.sahi_min_interval = float(sahi_cfg.get("region_min_interval_seconds", 0.5))
        self.sahi_max_attempts = int(sahi_cfg.get("region_max_attempts", 60))
        self._sahi_last_t = -1e9
        self._sahi_attempts = 0

        perf = cfg.get("performance", {}) or {}
        self.analiz_fps = float(perf.get("analiz_fps", 6))
        self.batch_size = int(perf.get("batch_size", 8))
        self.adaptive = bool(perf.get("adaptive_sampling", True))
        self.min_fps = float(perf.get("adaptive_min_fps", 2))
        self.max_runtime = float(perf.get("max_runtime_seconds", 540))
        self.log_timing = bool(perf.get("log_per_frame_timing", True))

        # Takip aktifse cikarim ardisik (batch yerine) calisir -> dogru track ID
        self.tracking_enabled = self.detector.tracking_enabled

        self._qod = qod_active_provider  # callable | None

    # ------------------------------------------------------------------
    def run(self, video_path: str, video_id: Optional[str] = None) -> Dict[str, Any]:
        if video_id is None:
            video_id = os.path.basename(video_path) or "video.mp4"
        budget = TimeBudget(self.max_runtime)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            LOG.error("Video acilamadi: %s -> bos sonuc.", video_path)
            return empty_results(video_id)

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 1 or np.isnan(src_fps):
            src_fps = 25.0  # gurbuzluk: fps okunamazsa varsayilan
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(src_fps / max(1e-6, self.analiz_fps))))
        LOG.info("Video: fps=%.2f total=%d step=%d (analiz_fps=%.1f) takip=%s",
                 src_fps, total, step, self.analiz_fps, self.tracking_enabled)

        batch_proc: List[np.ndarray] = []
        batch_meta: List[Tuple[np.ndarray, float, int]] = []
        frame_idx = -1
        analyzed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % step != 0:
                    continue

                t = frame_idx / src_fps
                qod_active = bool(self._qod()) if callable(self._qod) else False
                proc, _info = self.preproc.process(frame, qod_active=qod_active)

                if self.tracking_enabled:
                    # ARDISIK takipli cikarim (BoTSORT, persist) — batch yok
                    dets = self.detector.track(proc)
                    self._process_frame(proc, dets, t, qod_active)
                    analyzed += 1
                else:
                    # BATCH cikarim (takip yok)
                    batch_proc.append(proc)
                    batch_meta.append((proc, t, frame_idx))
                    if len(batch_proc) >= self.batch_size:
                        self._flush(batch_proc, batch_meta, qod_active)
                        analyzed += len(batch_proc)
                        batch_proc, batch_meta = [], []

                # butce kontrolu + adaptif ornekleme (her iki modda)
                if budget.over():
                    LOG.warning("Sure butcesi doldu (%.1fs); donguden cikiliyor.",
                                budget.elapsed())
                    break
                step = self._maybe_adapt(step, src_fps, frame_idx, total, budget)

            # kalan batch (yalnizca batch modunda olur)
            if batch_proc and not budget.over():
                self._flush(batch_proc, batch_meta,
                            bool(self._qod()) if callable(self._qod) else False)
                analyzed += len(batch_proc)
        except Exception as exc:  # gurbuzluk: dongu cokerse eldekiyle devam
            LOG.error("Kare dongusunde hata (%s); eldeki sonuclarla devam.", exc)
        finally:
            cap.release()

        plate_txt, plate_conf = self.plate.best()
        arac_bilgisi = self.vehicle.summarize(plate_txt, plate_conf)
        tespitler = self.aggregator.finalize()
        belt_event = self.seatbelt.summarize()   # video-seviye emniyet karari
        if belt_event:
            tespitler.append({k: belt_event[k] for k in
                              ("zaman_saniye", "kategori", "etiket", "confidence_score")})
            tespitler.sort(key=lambda x: x["zaman_saniye"])
        results = {
            "video_id": video_id,
            "arac_bilgisi": arac_bilgisi,
            "tespitler": tespitler,
        }
        out_cfg = self.cfg.get("output", {}) or {}
        results = validate_results(
            results,
            drop_invalid=bool(out_cfg.get("drop_invalid_labels", True)),
            round_confidence=int(out_cfg.get("round_confidence", 4)),
            round_time=int(out_cfg.get("round_time", 2)),
        )
        LOG.info("Analiz tamam: %d kare islendi, %d tespit, %.1fs.",
                 analyzed, len(results["tespitler"]), budget.elapsed())
        return results

    # ------------------------------------------------------------------
    def _flush(self, frames: List[np.ndarray],
               meta: List[Tuple[np.ndarray, float, int]],
               qod_active: bool) -> None:
        """Bir batch'i tespit edip per-frame mantigi calistir (takip kapaliyken)."""
        batch_dets = self.detector.predict_batch(frames)
        for dets, (proc, t, fidx) in zip(batch_dets, meta):
            self._process_frame(proc, dets, t, qod_active)

    def _process_frame(self, proc: np.ndarray, dets: List[Detection],
                       t: float, qod_active: bool) -> None:
        h, w = proc.shape[:2]
        main = self.vehicle.main_vehicle(dets)
        self.vehicle.update(proc, dets)   # per-track tip/renk/alan oylama

        driver_box = self._driver_box(dets, main)

        # --- plaka OCR (yalnizca plaka kutusu varken; iyi plaka bulunduysa atla) ---
        best_txt, best_conf = self.plate.best()
        plate_present = any(d.cls_name == self.plate_class for d in dets)
        plate_unreadable = False
        if plate_present and best_conf < 0.9:
            txt, _c = self.plate.read_from_frame(proc, dets)
            plate_unreadable = (txt == "")
        elif not best_txt and main is not None:
            # arac var ama plaka kutusu yok -> okunamadi say (SAHI adayi)
            plate_unreadable = True

        # --- SAHI (bolgesel/secici) — yalnizca tetiklenirse + butce korumali ---
        sahi_ok = (self._sahi_attempts < self.sahi_max_attempts and
                   (t - self._sahi_last_t) >= self.sahi_min_interval)
        if sahi_ok and main is not None and \
                self.sliced.should_trigger(plate_unreadable=plate_unreadable,
                                           small_object=True, qod_active=qod_active):
            self._sahi_last_t = t
            self._sahi_attempts += 1
            extra = self.sliced.predict_region(proc, main.box)
            if extra:
                dets = dets + extra
                # yeni plaka kutularini OCR'a sok
                self.plate.read_from_frame(proc, extra)
                # yeni kucuk nesneleri de degerlendir (objects asagida tekrar tarar)

        # --- nesneler (teknocan, bilgisayar) ---
        self.aggregator.add_many(self.objects.detect(dets, t))

        # --- yolcular (koltuk eslemesi) ---
        roi = main.box if main is not None else None
        self.aggregator.add_many(self.passengers.map_seats(dets, roi, (w, h), t))

        # --- emniyet kemeri (video-seviye: her karede biriktir, karar sonda) ---
        self.seatbelt.update(dets, driver_box, t)

        # --- birlesik eylemler (telefon/sigara/su) ---
        self.aggregator.add_many(self.actions.detect(dets, driver_box, t))

        # --- surucu yuzu (arkaya_bakma/esneme/etrafa_bakinma; MediaPipe + yedek CNN) ---
        if driver_box is not None and (self.face.active or self.hand_phone.active):
            x1, y1, x2, y2 = clamp_box(driver_box, w, h)
            face_crop = proc[y1:y2, x1:x2]
            if self.face.active:
                try:
                    self.aggregator.add_many(self.face.update(face_crop, t))
                except Exception as exc:
                    LOG.debug("Yuz analizi hatasi (kare atlandi): %s", exc)
            if self.hand_phone.active:  # el-kulak jesti -> telefonla_konusma
                try:
                    self.aggregator.add_many(self.hand_phone.update(face_crop, t, dets, driver_box))
                except Exception as exc:
                    LOG.debug("El-jest analizi hatasi (kare atlandi): %s", exc)

        # --- slalom (yanal yorunge, PER-TRACK) ---
        main_tid = main.track_id if main is not None else None
        slal = self.slalom.update(main_tid, main.box if main else None, w, t)
        if slal:
            self.aggregator.add(slal)

    # ------------------------------------------------------------------
    def _driver_box(self, dets: List[Detection],
                    main: Optional[Detection]
                    ) -> Optional[Tuple[float, float, float, float]]:
        """Surucu bolgesi: arac icindeki en buyuk kisi; yoksa kabin (arac ust %60)."""
        persons = [d for d in dets if d.cls_name == self.person_class and
                   (main is None or iou(d.box, main.box) > 0.1)]
        if persons:
            return max(persons, key=lambda d: bbox_area(d.box)).box
        if main is not None:
            x1, y1, x2, y2 = main.box
            return (x1, y1, x2, y1 + 0.6 * (y2 - y1))  # kabin (ust kisim)
        return None

    def _maybe_adapt(self, step: int, src_fps: float, frame_idx: int,
                     total: int, budget: TimeBudget) -> int:
        """Butce videodan hizli tukeniyorsa analiz_fps'i dusur (step'i artir)."""
        if not self.adaptive or total <= 0:
            return step
        frac_time = budget.fraction_used()
        frac_video = frame_idx / float(total)
        if frac_time > frac_video + 0.12:
            cur_fps = src_fps / step
            new_fps = max(self.min_fps, cur_fps * 0.75)
            new_step = max(1, int(round(src_fps / new_fps)))
            if new_step > step:
                LOG.info("Adaptif: analiz_fps %.1f->%.1f (step %d->%d) "
                         "[time=%.0f%% video=%.0f%%]",
                         cur_fps, new_fps, step, new_step,
                         frac_time * 100, frac_video * 100)
                return new_step
        return step
