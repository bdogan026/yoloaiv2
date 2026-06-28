"""calibrate_geometry.py — Surucu/koltuk esiklerini ORNEK VIDEOLARDAN otomatik
kalibre eder — image'a GIRMEZ.

Sorun: yol kenari kamerasinin acisi bilinmeden "on/arka" ve "sol/sag" koltuk
esikleri tahminidir. Bu arac, komiteden gelen ornek videolar uzerinde calisarak
esikleri VERIYE DAYALI cikarir (tahmin yerine olcum):

  1) Tespit+takip ile her karede ana arac ROI'si + icindeki kisi merkezleri
     ROI'ye gore normalize edilir (nx, ny ∈ [0,1]).
  2) ny uzerinde 1B k-means (k=2) -> on/arka sinirini bulur -> front_y_max.
  3) Arka kisilerde nx uzerinde 1B k-means -> sol/sag siniri -> back_left_x_max.
  4) On bolgede baskin x kumesi -> surucu tarafi -> driver_selection onerisi.

Cikti: config'e yapistirilabilir oneri degerleri (config'i OTOMATIK YAZMAZ;
yorumlari korumak icin elle yapistirilir).

Kullanim:
    python training/calibrate_geometry.py --videos-dir sample_videos/ \
        --weights weights/best_model.pt --max-frames-per-video 300
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import bbox_center, iou, load_config  # noqa: E402


def kmeans_1d(x: np.ndarray, k: int = 2, iters: int = 100) -> Optional[np.ndarray]:
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return None
    centers = np.quantile(x, np.linspace(0.2, 0.8, k))
    for _ in range(iters):
        d = np.abs(x[:, None] - centers[None, :])
        labels = d.argmin(1)
        new = np.array([x[labels == j].mean() if np.any(labels == j) else centers[j]
                        for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return np.sort(centers)


def main() -> int:
    ap = argparse.ArgumentParser(description="Surucu/koltuk geometri kalibrasyonu")
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--weights", default="weights/best_model.pt")
    ap.add_argument("--max-frames-per-video", type=int, default=300)
    args = ap.parse_args()

    try:
        import cv2
        from src.detection.detector import YOLODetector
    except ImportError as exc:
        print(f"cv2/torch/ultralytics gerekli: {exc}", file=sys.stderr)
        return 1

    cfg = load_config()
    det = YOLODetector(cfg, args.weights)
    if not det.loaded:
        print("Detektor agirligi yok; kalibrasyon yapilamaz.", file=sys.stderr)
        return 1
    vehicle_classes = set((cfg.get("detection", {}) or {}).get("vehicle_types", {}).keys())
    person_class = str((cfg.get("detection", {}) or {}).get("person_class", "kisi"))

    nxs: List[float] = []
    nys: List[float] = []
    vids = [f for f in os.listdir(args.videos_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    if not vids:
        print("Ornek video bulunamadi.", file=sys.stderr)
        return 1

    for v in vids:
        cap = cv2.VideoCapture(os.path.join(args.videos_dir, v))
        n = 0
        while n < args.max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1
            dets = det.predict_one(frame)
            vehicles = [d for d in dets if d.cls_name in vehicle_classes]
            if not vehicles:
                continue
            main = max(vehicles, key=lambda d: (d.box[2]-d.box[0])*(d.box[3]-d.box[1]))
            rx1, ry1, rx2, ry2 = main.box
            rw, rh = max(1.0, rx2-rx1), max(1.0, ry2-ry1)
            for d in dets:
                if d.cls_name != person_class:
                    continue
                if iou(d.box, main.box) <= 0.05:
                    continue
                cx, cy = bbox_center(d.box)
                nx, ny = (cx-rx1)/rw, (cy-ry1)/rh
                if 0 <= nx <= 1 and 0 <= ny <= 1:
                    nxs.append(nx)
                    nys.append(ny)
        cap.release()
        print(f"[calib] {v}: toplam kisi ornegi={len(nys)}")

    if len(nys) < 10:
        print("Yeterli kisi ornegi yok (>=10 gerekir). Daha fazla/uygun video gerek.")
        return 1

    ny = np.array(nys)
    nx = np.array(nxs)
    cy = kmeans_1d(ny, 2)
    front_y_max = float((cy[0] + cy[1]) / 2) if cy is not None else 0.55

    back_mask = ny >= front_y_max
    back_left_x_max = 0.5
    if back_mask.sum() >= 4:
        cx_back = kmeans_1d(nx[back_mask], 2)
        if cx_back is not None:
            back_left_x_max = float((cx_back[0] + cx_back[1]) / 2)

    front_mask = ny < front_y_max
    driver_side = "largest"
    if front_mask.sum() >= 4:
        front_x_mean = float(nx[front_mask].mean())
        driver_side = "leftmost" if front_x_mean < 0.5 else "rightmost"

    print("\n=== ONERILEN KALIBRASYON (config'e elle yapistirin) ===")
    print("passengers:")
    print(f"  front_y_max: {round(front_y_max, 3)}")
    print(f"  back_left_x_max: {round(back_left_x_max, 3)}")
    print("face:")
    print(f"  driver_selection: \"{driver_side}\"")
    print(f"\n(ornek sayisi: {len(nys)}; on={int(front_mask.sum())}, "
          f"arka={int(back_mask.sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
