"""stream_pipeline.py — Canli final iki-yollu cikarim (image'a GIRMEZ).

Cekirdek src/ modulleri burada YENIDEN KULLANILIR (tek kaynak, iki dagitim
hedefi). Iki cikarim yolu:

  * Hafif yol (QoD YOK)  : CLAHE + (tek) YOLO, dusuk gecikme, dusuk data rate.
  * Agir yol (QoD AKTIF) : DCP acik + SAHI (bolgesel) + plaka OCR -> daha yuksek
    basarim. AYNI best_model.pt (kazanc QoD+SAHI+OCR'den, buyuk modelden DEGIL).

Akis (her kare):
  1) Hafif yol: YOLO tespit (her zaman).
  2) Risk + QoD ihtiyaci (need) hesapla (risk_model).
  3) QoDController histerezisli karar -> qod_active.
  4) qod_active ise: QoD oturumu ac + agir yol zenginlestirme (DCP/SAHI/OCR);
     ihtiyac dusunce oturumu serbest birak.
  5) Metrikleri logla: QoD ACIK vs KAPALI basarim farki (puanlama kaniti).

NOT: source bir RTSP/webcam olabilir; bu modul gercek 5G ortaminda calistirilir.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from live.camara_auth import CamaraAuthClient
from live.number_verification import NumberVerificationClient
from live.qod_client import QoDClient, QoDController
from live.risk_model import RiskModel
from live.speed import SpeedEstimator
from src.actions import ActionDetector
from src.detection.detector import YOLODetector
from src.detection.objects import ObjectDetector
from src.detection.plate import PlateReader
from src.detection.seatbelt import SeatbeltChecker
from src.detection.sliced import SlicedDetector
from src.detection.vehicle import VehicleAnalyzer
from src.preprocessing import Preprocessor
from src.utils import apply_offline_env, bbox_area, get_logger, iou

LOG = get_logger("rapid_response.stream")


@dataclass
class PathMetrics:
    """QoD ACIK/KAPALI bir yol icin biriken basarim kaniti."""
    frames: int = 0
    det_count: int = 0
    conf_sum: float = 0.0
    plate_reads: int = 0
    small_obj: int = 0

    def add(self, n_det: int, mean_conf: float, plate: bool, small: int) -> None:
        self.frames += 1
        self.det_count += n_det
        self.conf_sum += mean_conf
        self.plate_reads += int(plate)
        self.small_obj += small

    def summary(self) -> Dict[str, float]:
        f = max(1, self.frames)
        return {
            "frames": self.frames,
            "avg_detections": round(self.det_count / f, 3),
            "avg_confidence": round(self.conf_sum / f, 4),
            "plate_read_rate": round(self.plate_reads / f, 4),
            "avg_small_objects": round(self.small_obj / f, 3),
        }


class StreamPipeline:
    def __init__(self, cfg: Dict[str, Any], weights_path: Optional[str] = None,
                 *, phone_number: Optional[str] = None) -> None:
        self.cfg = cfg
        apply_offline_env(cfg)

        # Canli config zorlamasi: DCP qod_gated, SAHI qod_active tetikli
        cfg.setdefault("preprocessing", {}).setdefault("dcp", {})["mode"] = "qod_gated"
        cfg.setdefault("sahi", {})["trigger"] = "qod_active"

        # Cekirdek bilesenler (src/ yeniden kullanim)
        self.detector = YOLODetector(cfg, weights_path)
        self.sliced = SlicedDetector(cfg, self.detector.weights_path)
        self.preproc = Preprocessor(cfg)
        self.vehicle = VehicleAnalyzer(cfg)
        self.plate = PlateReader(cfg)
        self.objects = ObjectDetector(cfg)
        self.seatbelt = SeatbeltChecker(cfg)
        self.actions = ActionDetector(cfg)

        # Risk + hiz + 5G kontrol
        self.risk = RiskModel(window_frames=90)
        self.speed = SpeedEstimator(cfg)   # IPM hiz (Sartname §5, %40 YZ)
        self.qod_ctrl = QoDController()
        # Paylasimli CAMARA token saglayici (CIBA/client_credentials, otomatik token)
        auth = CamaraAuthClient(config=cfg)
        asec = (cfg.get("auth") or {})
        self.qod = QoDClient(auth=auth, scope=asec.get("qod_scope"))
        self.nv = NumberVerificationClient(auth=auth, scope=asec.get("nv_scope"))
        self.phone_number = phone_number

        self.person_class = str((cfg.get("detection", {}) or {}).get(
            "person_class", "kisi"))
        self.plate_class = str((cfg.get("detection", {}) or {}).get(
            "plate_class", "plaka"))
        self.small_ratio = float((cfg.get("sahi", {}) or {}).get(
            "small_object_area_ratio", 0.01))

        # Puanlama kaniti: QoD acik vs kapali metrikleri
        self.m_on = PathMetrics()
        self.m_off = PathMetrics()

    # ------------------------------------------------------------------
    def authenticate(self) -> bool:
        """Ilk adim: sessiz, sebeke-tabanli Number Verification."""
        if not self.phone_number:
            LOG.info("Number Verification atlandi (telefon numarasi verilmedi).")
            return True
        return self.nv.verify(self.phone_number)

    def run(self, source: Any = 0, *, app_server: Optional[Dict[str, Any]] = None,
            max_seconds: Optional[float] = None, on_frame=None,
            should_stop=None) -> Dict[str, Any]:
        """Canli akisi isle.

        on_frame(result_dict, frame_bgr): her kare icin cagrilir (backend WS bunu
            mobil uygulamaya yayinlar). None ise sadece loglanir.
        should_stop(): True donerse dongu durur (backend baglanti kopunca).
        """
        if not self.authenticate():
            LOG.error("Number Verification basarisiz; akis baslatilmadi.")
            return {"error": "number_verification_failed"}

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            LOG.error("Akis acilamadi: %s", source)
            return {"error": "stream_open_failed"}

        device = {"phoneNumber": self.phone_number} if self.phone_number else \
                 {"ipv4Address": {"publicAddress": "0.0.0.0"}}
        app_server = app_server or {"ipv4Address": "0.0.0.0", "ports": {}}

        t0 = time.time()
        frame_idx = 0
        try:
            while True:
                if callable(should_stop) and should_stop():
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if max_seconds and (time.time() - t0) >= max_seconds:
                    break
                result = self._step(frame, frame_idx, device, app_server)
                if on_frame and result is not None:
                    try:
                        on_frame(result, frame)
                    except Exception as exc:
                        LOG.debug("on_frame callback hatasi: %s", exc)
        except KeyboardInterrupt:
            LOG.info("Akis kullanici tarafindan durduruldu.")
        finally:
            cap.release()
            if self.qod.active:
                self.qod.delete_session()

        report = self.evidence_report()
        LOG.info("QoD ACIK/KAPALI basarim ozeti: %s", report)
        return report

    # ------------------------------------------------------------------
    def _step(self, frame: np.ndarray, frame_idx: int,
              device: Dict[str, Any], app_server: Dict[str, Any]) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        qod_active = self.qod_ctrl.active

        # 1) On isleme (qod_gated: DCP yalnizca qod_active iken)
        proc, _info = self.preproc.process(frame, qod_active=qod_active)

        # 2) Hafif yol: tek YOLO (her zaman). Takip acaksa BoTSORT (per-arac profil).
        dets = (self.detector.track(proc) if self.detector.tracking_enabled
                else self.detector.predict_one(proc))
        main = self.vehicle.main_vehicle(dets)
        self.vehicle.update(proc, dets)
        driver_box = self._driver_box(dets, main)

        # 3) Sofor eylemleri (risk girdisi)
        t = frame_idx / 30.0  # canli icin nominal

        # Hiz tahmini (IPM/homografi) — ana arac (UI'da gosterilir, GT ile olculur)
        speed_kmh = self.speed.update(main.track_id, main.box, t) \
            if main is not None else None
        events: List[Dict[str, Any]] = []
        events += self.actions.detect(dets, driver_box, t)
        belt = self.seatbelt.check(dets, driver_box)
        if belt:
            belt["zaman_saniye"] = t
            events.append(belt)

        # 4) Risk + QoD ihtiyaci
        prox = RiskModel.proximity_factor(main.box, (w, h)) if main else 1.0
        state = self.risk.update(events, proximity=prox)
        plate_present = any(d.cls_name == self.plate_class for d in dets)
        plate_unreadable = (not self.plate.best()[0])
        state = self.risk.add_plate_need(state, plate_unreadable,
                                         vehicle_present=main is not None)

        # 5) QoD karari (histerezisli, ihtiyac-tabanli)
        now_active = self.qod_ctrl.update(state.need)

        plate_read_this = False
        small_now = sum(1 for d in dets
                        if bbox_area(d.box) / float(w * h) < self.small_ratio)

        if now_active:
            if not self.qod.active:
                self.qod.create_session(device, app_server, duration_s=60)
            # AGIR YOL zenginlestirme: SAHI (bolgesel) + OCR
            if main is not None and self.sliced.should_trigger(qod_active=True):
                extra = self.sliced.predict_region(proc, main.box)
                if extra:
                    dets = dets + extra
                    small_now += sum(1 for d in extra
                                     if bbox_area(d.box) / float(w * h) < self.small_ratio)
            txt, _c = self.plate.read_from_frame(proc, dets)
            plate_read_this = bool(txt)
        else:
            if self.qod.active:
                self.qod.delete_session()

        # 6) Puanlama kaniti metrikleri
        mean_conf = float(np.mean([d.conf for d in dets])) if dets else 0.0
        target = self.m_on if now_active else self.m_off
        target.add(len(dets), mean_conf, plate_read_this, small_now)

        # UI'ya yayin (burada log): tespitler + risk profili + hiz + QoD durumu
        LOG.debug("f=%d risk=%.0f(%s) need=%.2f hiz=%s qod=%s det=%d",
                  frame_idx, state.risk, state.profile, state.need,
                  (f"{speed_kmh:.0f}km/h" if speed_kmh is not None else "-"),
                  now_active, len(dets))

        # Mobil uygulamaya yayinlanacak yapilandirilmis sonuc (backend WS kullanir)
        arac = self.vehicle.summarize(*self.plate.best())  # calisan tahmin
        arac["hiz_kmh"] = round(speed_kmh, 1) if speed_kmh is not None else None
        return {
            "frame_idx": frame_idx,
            "width": w,
            "height": h,
            "arac_bilgisi": arac,
            "detections": [{
                "cls": d.cls_name,
                "conf": round(d.conf, 3),
                "box": [round(float(c), 1) for c in d.box],
                "track_id": d.track_id,
            } for d in dets],
            "tespitler": [{
                "zaman_saniye": round(float(e.get("zaman_saniye", t)), 2),
                "kategori": e.get("kategori"),
                "etiket": e.get("etiket"),
                "confidence_score": round(float(e.get("confidence_score", 0.0)), 3),
            } for e in events],
            "risk": {"score": round(state.risk, 1), "profile": state.profile,
                     "need": round(state.need, 3)},
            "qod": {"active": bool(now_active)},
        }

    # ------------------------------------------------------------------
    def _driver_box(self, dets, main):
        persons = [d for d in dets if d.cls_name == self.person_class and
                   (main is None or iou(d.box, main.box) > 0.1)]
        if persons:
            return max(persons, key=lambda d: bbox_area(d.box)).box
        if main is not None:
            x1, y1, x2, y2 = main.box
            return (x1, y1, x2, y1 + 0.6 * (y2 - y1))
        return None

    def evidence_report(self) -> Dict[str, Any]:
        """QoD ACIK vs KAPALI olculen basarim farki (Gelistirme notu: %40 kanit)."""
        on = self.m_on.summary()
        off = self.m_off.summary()
        delta = {}
        for k in ("avg_detections", "avg_confidence", "plate_read_rate",
                  "avg_small_objects"):
            delta[k] = round(on.get(k, 0) - off.get(k, 0), 4)
        return {"qod_on": on, "qod_off": off, "delta_on_minus_off": delta}


if __name__ == "__main__":
    import sys
    from src.utils import load_config
    cfg = load_config()
    sp = StreamPipeline(cfg)
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    print(sp.run(src, max_seconds=30))
