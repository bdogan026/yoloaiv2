"""calibrate_speed.py — Hiz tahmini icin homografi kalibrasyonu + dogrulama —
image'a GIRMEZ.

Uc mod:
  extract    : ORNEK VIDEOLARDAN temiz bir referans kare (medyan arka plan) cikarir
               ve sahne isaretlerini (serit cizgileri) Canny+Hough ile tespit edip
               isaretli bir resme + JSON'a yazar. Boylece kullanici, gercek dunya
               mesafelerini OLCEREK image_points'i bu temiz kareden secebilir.
               (Not: metrik homografi icin gercek mesafeler kullanicidan gelir;
               piksele bakip metre uydurulmaz.)
  homography : Goruntu noktalari (u,v) ile bunlarin GERCEK DUNYA (X,Y, metre)
               karsiliklarindan homografi H'yi hesaplar; config.speed.homography'e
               yapistirilacak 3x3 matrisi yazar. (>=4 nokta cifti gerekir.)
  validate   : Verilen GT hizlariyla (komiteden gelen "gercek hiz") tahminleri
               karsilastirir; MAE/RMSE (km/h) raporlar. Gelistirme notundaki
               referans (Gajdoš IPM ~0.58 km/h MAE) ile kiyas icin yazdirir.

Kullanim:
  # 0) Ornek videolardan temiz kare + serit adaylari cikar (annotasyon icin):
  python training/calibrate_speed.py --mode extract --videos-dir sample_videos/
  # 1) Homografi uret (noktalari temiz kareden secip gercek mesafeleri olc):
  python training/calibrate_speed.py --mode homography --points points.json
  #    points.json: {"image_points": [[u,v],...], "world_points": [[X,Y],...]}  (>=4)
  # 2) Dogrula (config.speed.homography ayarlandiktan sonra):
  python training/calibrate_speed.py --mode validate \
      --gt-speeds gt_speeds.json --videos-dir test_videos/ --weights weights/best_model.pt
  #    gt_speeds.json: {"video_001.mp4": 52.3, ...}  (km/h)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config  # noqa: E402

# Gelistirme notundan referans (kullanicinin kendi belgesinde verdigi deger)
GAJDOS_REF_MAE_KMH = 0.58


def mode_extract(args) -> int:
    """Ornek videolardan temiz referans kare (medyan arka plan) + serit adaylari."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        print(f"cv2/numpy gerekli: {exc}", file=sys.stderr)
        return 1
    if not args.videos_dir:
        print("extract: --videos-dir gerekli.", file=sys.stderr)
        return 1
    vids = [f for f in os.listdir(args.videos_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    if not vids:
        print("Ornek video bulunamadi.", file=sys.stderr)
        return 1

    # Medyan arka plan: videolardan esit araliklarla kare ornekle (hareketli
    # araclar medyanla elenir -> temiz yol sahnesi)
    frames = []
    for v in vids:
        cap = cv2.VideoCapture(os.path.join(args.videos_dir, v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        take = max(1, total // max(1, args.max_frames)) if total > 0 else 10
        idx = 0
        while len(frames) < args.max_frames:
            ret, fr = cap.read()
            if not ret:
                break
            if idx % take == 0:
                frames.append(fr)
            idx += 1
        cap.release()
        if len(frames) >= args.max_frames:
            break
    if not frames:
        print("Kare okunamadi.", file=sys.stderr)
        return 1

    h, w = frames[0].shape[:2]
    frames = [cv2.resize(f, (w, h)) for f in frames]
    bg = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

    # Serit/sahne isaretleri: Canny + olasiliksal Hough
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=int(0.08*w), maxLineGap=20)
    annotated = bg.copy()
    segments = []
    if lines is not None:
        for ln in lines[:200]:
            x1, y1, x2, y2 = [int(c) for c in ln[0]]
            segments.append([[x1, y1], [x2, y2]])
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(args.out_image, bg)
    cv2.imwrite(os.path.splitext(args.out_image)[0] + "_lines.png", annotated)
    with open(args.out_segments, "w", encoding="utf-8") as f:
        json.dump({"image_size": [w, h], "line_segments": segments}, f, indent=2)

    print(f"\n=== EXTRACT TAMAM ===")
    print(f"Temiz referans kare : {args.out_image}")
    print(f"Serit adaylari resmi: {os.path.splitext(args.out_image)[0]}_lines.png")
    print(f"Serit segment JSON  : {args.out_segments}  ({len(segments)} segment)")
    print("\nSonraki adim: temiz kareden >=4 nokta secin, GERCEK mesafelerini "
          "(metre) olcun, points.json yapip --mode homography calistirin.")
    return 0


def mode_homography(args) -> int:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        print(f"cv2/numpy gerekli: {exc}", file=sys.stderr)
        return 1
    with open(args.points, "r", encoding="utf-8") as f:
        pts = json.load(f)
    img = pts.get("image_points") or []
    wld = pts.get("world_points") or []
    if len(img) < 4 or len(img) != len(wld):
        print("En az 4 nokta cifti gerekir ve image/world esit uzunlukta olmali.",
              file=sys.stderr)
        return 1
    src = np.array(img, dtype=np.float32)
    dst = np.array(wld, dtype=np.float32)
    if len(img) == 4:
        H = cv2.getPerspectiveTransform(src, dst)
    else:
        H, _ = cv2.findHomography(src, dst, method=0)
    H = np.array(H, dtype=float)

    # Yeniden-yansitma hatasi (kalibrasyon kalitesi)
    errs = []
    for (u, v), (X, Y) in zip(img, wld):
        p = H @ np.array([u, v, 1.0])
        if abs(p[2]) > 1e-9:
            errs.append(math.hypot(p[0]/p[2] - X, p[1]/p[2] - Y))
    rms = math.sqrt(sum(e*e for e in errs) / len(errs)) if errs else float("nan")

    print("\n=== config.yaml -> speed.homography (yapistir) ===")
    print("speed:")
    print("  homography:")
    for row in H:
        print(f"    - [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}]")
    print(f"\nYeniden-yansitma RMS hatasi: {rms:.4f} m "
          f"(dusuk = iyi kalibrasyon)")
    return 0


def mode_validate(args) -> int:
    from live.speed import estimate_main_vehicle_speed
    from src.detection.detector import YOLODetector

    cfg = load_config()
    if not (cfg.get("speed", {}) or {}).get("homography") and \
       not (cfg.get("speed", {}) or {}).get("image_points"):
        print("UYARI: config.speed.homography ayarli degil; once 'homography' "
              "moduyla uretip config'e yapistirin.", file=sys.stderr)

    with open(args.gt_speeds, "r", encoding="utf-8") as f:
        gt = json.load(f)

    abs_errs, rows = [], []
    for vid, gt_kmh in gt.items():
        path = os.path.join(args.videos_dir, vid)
        if not os.path.exists(path):
            print(f"  atlandi (bulunamadi): {vid}")
            continue
        # Her video icin TAZE detector (BoTSORT durumu karismasin)
        det = YOLODetector(cfg, args.weights)
        est = estimate_main_vehicle_speed(path, cfg, detector=det)
        if est is None:
            print(f"  {vid}: tahmin uretilemedi (homografi/agirlik/tespit?)")
            continue
        err = abs(est - float(gt_kmh))
        abs_errs.append(err)
        rows.append((vid, float(gt_kmh), est, err))

    print("\n=== HIZ DOGRULAMA ===")
    print(f"{'video':<24}{'GT(km/h)':>10}{'tahmin':>10}{'|hata|':>10}")
    for vid, g, e, er in rows:
        print(f"{vid:<24}{g:>10.2f}{e:>10.2f}{er:>10.2f}")
    if abs_errs:
        mae = sum(abs_errs) / len(abs_errs)
        rmse = math.sqrt(sum(e*e for e in abs_errs) / len(abs_errs))
        print(f"\nMAE  = {mae:.3f} km/h")
        print(f"RMSE = {rmse:.3f} km/h")
        print(f"Referans (Gelistirme notu, Gajdoš IPM): ~{GAJDOS_REF_MAE_KMH} km/h MAE")
    else:
        print("Karsilastirilacak ortak (video+GT) ornek bulunamadi.")
    if args.out and abs_errs:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "mae": mae, "rmse": rmse,
                       "reference_mae": GAJDOS_REF_MAE_KMH}, f,
                      ensure_ascii=False, indent=2)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Hiz homografi kalibrasyonu/dogrulama")
    ap.add_argument("--mode", required=True,
                    choices=["extract", "homography", "validate"])
    ap.add_argument("--points", help="homography modu: image/world noktalari JSON")
    ap.add_argument("--gt-speeds", help="validate modu: {video: kmh} JSON")
    ap.add_argument("--videos-dir", help="extract/validate modu: video klasoru")
    ap.add_argument("--weights", default="weights/best_model.pt")
    ap.add_argument("--max-frames", type=int, default=40,
                    help="extract: medyan arka plan icin kare sayisi")
    ap.add_argument("--out-image", default="speed_ref_frame.png",
                    help="extract: temiz referans kare cikti yolu")
    ap.add_argument("--out-segments", default="speed_ref_lines.json",
                    help="extract: serit segment JSON cikti yolu")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.mode == "extract":
        return mode_extract(args)
    if args.mode == "homography":
        if not args.points:
            print("homography: --points gerekli.", file=sys.stderr)
            return 1
        return mode_homography(args)
    if not args.gt_speeds or not args.videos_dir:
        print("validate: --gt-speeds ve --videos-dir gerekli.", file=sys.stderr)
        return 1
    return mode_validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
