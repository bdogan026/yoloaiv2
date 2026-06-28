"""speed.py — IPM/homografi tabanli arac hiz tahmini — image'a GIRMEZ.

Sartname §5: GERCEK HIZ, final YZ puaninin (%40) parcasi. Bu modul yalnizca
live/ + olcum tarafindadir; FTR results.json'a hiz YAZILMAZ (FTR semasinda hiz
alani yok; FTR > Sartname onceligi).

Yontem:
  1) Homografi H (3x3): goruntu pikseli (u,v) -> yer duzlemi dunya koordinati
     (X,Y) METRE. H, kalibrasyondan gelir (>=4 nokta ciftiyle), bkz.
     training/calibrate_speed.py.
  2) Her BoTSORT track'i icin aracin YER-TEMAS noktasi (varsayilan bbox alt-orta,
     teker hizasi) H ile yer duzlemine yansitilir.
  3) Track'in (t, X, Y) gecmisi pencere uzerinde tutulur; X(t) ve Y(t) dogrusal
     regresyonla hiz vektorune (vx, vy) cevrilir; hiz = |(vx,vy)| * 3.6 -> km/h.

Neden alt-orta nokta? Yalnizca aracin YERE temas ettigi nokta yer duzlemindedir;
bbox merkezi havada kalir ve homografi onu yanlis yansitir.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from src.utils import get_logger

LOG = get_logger("rapid_response.speed")


class SpeedEstimator:
    """Per-track IPM hiz tahmini (km/h)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        s = cfg.get("speed", {}) or {}
        self.enabled: bool = bool(s.get("enabled", False))
        self.contact_point: str = str(s.get("contact_point", "bottom_center"))
        self.window_s: float = float(s.get("window_seconds", 1.0))
        self.min_span: float = float(s.get("min_time_span", 0.3))
        self.min_disp: float = float(s.get("min_displacement_m", 0.2))
        self.max_kmh: float = float(s.get("max_speed_kmh", 250))
        self.smoothing: str = str(s.get("smoothing", "regression"))
        self.max_stale: float = float(
            (cfg.get("tracking", {}) or {}).get("max_stale_seconds", 2.0))

        self.H: Optional[np.ndarray] = None
        if self.enabled:
            self.H = self._build_homography(s)
        # track_id -> deque[(t, X, Y)]
        self._bufs: Dict[int, Deque[Tuple[float, float, float]]] = {}

    # ------------------------------------------------------------------
    def _build_homography(self, s: Dict[str, Any]) -> Optional[np.ndarray]:
        h = s.get("homography")
        if h:
            try:
                H = np.array(h, dtype=np.float64).reshape(3, 3)
                LOG.info("Hiz: homografi matrisi config'ten yuklendi.")
                return H
            except Exception as exc:
                LOG.error("Hiz: gecersiz homography matrisi (%s).", exc)
        img = s.get("image_points") or []
        wld = s.get("world_points") or []
        if len(img) >= 4 and len(img) == len(wld):
            try:
                import cv2
                src = np.array(img, dtype=np.float32)
                dst = np.array(wld, dtype=np.float32)
                if len(img) == 4:
                    H = cv2.getPerspectiveTransform(src, dst)
                else:
                    H, _ = cv2.findHomography(src, dst, method=0)
                LOG.info("Hiz: homografi %d nokta ciftinden turetildi.", len(img))
                return np.array(H, dtype=np.float64)
            except Exception as exc:
                LOG.error("Hiz: homografi nokta ciftlerinden kurulamadi (%s).", exc)
        LOG.warning("Hiz: homografi yok; hiz tahmini devre disi (kalibrasyon gerekli).")
        return None

    @property
    def ready(self) -> bool:
        return self.enabled and self.H is not None

    # ------------------------------------------------------------------
    def project(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """Goruntu pikseli (u,v) -> yer duzlemi (X,Y) metre."""
        if self.H is None:
            return None
        p = self.H @ np.array([u, v, 1.0], dtype=np.float64)
        if abs(p[2]) < 1e-9:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])

    def _contact(self, box: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        if self.contact_point == "center":
            return (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        # bottom_center (varsayilan): yer-temas (teker hizasi)
        return (0.5 * (x1 + x2), y2)

    def update(self, track_id: Optional[int],
               box: Tuple[float, float, float, float],
               t_seconds: float) -> Optional[float]:
        """Track'in bu karedeki konumunu ekler ve guncel hizi (km/h) doner.

        Yeterli ornek/zaman yoksa None; hareketsizse 0.0; mantik disi (>max) None.
        """
        if not self.ready or box is None:
            return None
        u, v = self._contact(box)
        world = self.project(u, v)
        if world is None:
            return None
        key = track_id if track_id is not None else -1
        buf = self._bufs.setdefault(key, deque())
        buf.append((t_seconds, world[0], world[1]))
        while buf and (t_seconds - buf[0][0]) > self.window_s:
            buf.popleft()
        self._cleanup(t_seconds)

        if len(buf) < 3:
            return None
        ts = np.array([p[0] for p in buf], dtype=np.float64)
        xs = np.array([p[1] for p in buf], dtype=np.float64)
        ys = np.array([p[2] for p in buf], dtype=np.float64)
        span = float(ts[-1] - ts[0])
        if span < self.min_span or np.var(ts) <= 1e-9:
            return None

        # Hareketsizlik kontrolu (pencere ucu yer degistirmesi)
        disp = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
        if disp < self.min_disp:
            return 0.0

        speed_ms = self._speed_ms(ts, xs, ys)
        if speed_ms is None:
            return None
        kmh = speed_ms * 3.6
        if kmh > self.max_kmh or kmh < 0:
            return None
        return float(kmh)

    def _speed_ms(self, ts: np.ndarray, xs: np.ndarray, ys: np.ndarray
                  ) -> Optional[float]:
        try:
            if self.smoothing == "endpoints":
                dt = ts[-1] - ts[0]
                if dt <= 0:
                    return None
                return math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) / dt
            if self.smoothing == "median":
                vs = []
                for i in range(1, len(ts)):
                    dt = ts[i] - ts[i - 1]
                    if dt <= 0:
                        continue
                    vs.append(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) / dt)
                return float(np.median(vs)) if vs else None
            # regression (varsayilan, en kararli): X(t) ve Y(t) egimleri
            vx = float(np.polyfit(ts, xs, 1)[0])
            vy = float(np.polyfit(ts, ys, 1)[0])
            return math.hypot(vx, vy)
        except Exception as exc:
            LOG.debug("Hiz hesap hatasi: %s", exc)
            return None

    def _cleanup(self, t_now: float) -> None:
        stale = [k for k, b in self._bufs.items()
                 if not b or (t_now - b[-1][0]) > self.max_stale]
        for k in stale:
            del self._bufs[k]


# ============================================================================
# Video uzerinde ana aracin hizini tahmin et (calibrate_speed + evaluate icin)
# ============================================================================
def estimate_main_vehicle_speed(video_path: str, cfg: Dict[str, Any],
                                detector: Any = None) -> Optional[float]:
    """Bir videoda BASKIN aracin tahmini hizini (km/h, medyan) doner.

    Her video icin TAZE detector kullanilmali (BoTSORT durumu videolar arasi
    karismasin); detector=None ise burada kurulur. Homografi yoksa None.
    """
    import cv2
    from src.detection.detector import YOLODetector

    est = SpeedEstimator(cfg)
    if not est.ready:
        LOG.error("Hiz tahmini icin homografi gerekli (config speed.*).")
        return None

    if detector is None:
        detector = YOLODetector(cfg, weights_path=None)
    if not detector.loaded:
        LOG.error("Detektor agirligi yok; hiz tahmini yapilamaz.")
        return None

    vehicle_classes = set((cfg.get("detection", {}) or {}).get("vehicle_types", {}).keys())
    perf = cfg.get("performance", {}) or {}
    analiz_fps = float(perf.get("analiz_fps", 6))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        LOG.error("Video acilamadi: %s", video_path)
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if src_fps <= 1:
        src_fps = 25.0
    step = max(1, int(round(src_fps / max(1e-6, analiz_fps))))

    per_track_speeds: Dict[int, List[float]] = {}
    per_track_area: Dict[int, float] = {}
    fidx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fidx += 1
        if fidx % step != 0:
            continue
        t = fidx / src_fps
        dets = detector.track(frame)
        vehicles = [d for d in dets if d.cls_name in vehicle_classes]
        if not vehicles:
            continue
        main = max(vehicles, key=lambda d: (d.box[2]-d.box[0])*(d.box[3]-d.box[1]))
        key = main.track_id if main.track_id is not None else -1
        per_track_area[key] = per_track_area.get(key, 0.0) + \
            (main.box[2]-main.box[0])*(main.box[3]-main.box[1])
        kmh = est.update(main.track_id, main.box, t)
        if kmh is not None:
            per_track_speeds.setdefault(key, []).append(kmh)
    cap.release()

    if not per_track_area:
        return None
    dominant = max(per_track_area, key=per_track_area.get)
    speeds = per_track_speeds.get(dominant, [])
    if not speeds:
        return None
    return float(np.median(speeds))


if __name__ == "__main__":
    # Tek basina test: olcekli homografi (piksel->metre) ile bilinen hiz
    H = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 1]]  # 1 px = 0.05 m
    cfg = {"speed": {"enabled": True, "homography": H, "window_seconds": 1.0,
                     "min_time_span": 0.3, "min_displacement_m": 0.1,
                     "smoothing": "regression"},
           "tracking": {"max_stale_seconds": 2.0}}
    est = SpeedEstimator(cfg)
    fps = 30.0
    last = None
    for i in range(60):
        t = i / fps
        # 10 px/kare yatay -> 0.5 m/kare -> 15 m/s -> 54 km/h
        cx = 100 + 10 * i
        last = est.update(1, (cx-20, 200, cx+20, 260), t)
    print("Tahmini hiz (beklenen ~54 km/h):", round(last, 2) if last else None)
