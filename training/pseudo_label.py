"""pseudo_label.py — DETRAC (veya herhangi) videodan pseudo-label + augment —
image'a GIRMEZ (yalnizca egitim veri hazirligi).

Akis:
  video(lar) -> kare ornekle -> pretrained COCO YOLO ile ARAC tespiti ->
  YOLO formatinda etiket yaz -> (opsiyonel) hava-augment varyantlari +
  (opsiyonel) arac crop'lari.

DURUST SINIR (onemli):
  COCO detector "car/truck/bus" verir; FTR'nin sedan/suv/hatchback AYRIMINI
  YAPAMAZ. Bu yuzden cikti TEK SINIF 'arac' (lokalizasyon). Faydasi:
    1) Dedektor lokalizasyon gurbuzlugu (hava-augment ile zenginlesir).
    2) --crops ile arac kirpintilari -> sonradan tip/renk siniflandirmasi
       (renk pipeline'i gibi) icin hazir veri.
  FTR 7 tip sinifi icin crop'lar ELLE/BoxCars ile etiketlenmeli (ayri is;
  bkz. training/sources.yaml notu).

Augment label-guvenli: donusumler fotometrik/bulaniklik -> bbox kaymaz; YOLO
etiketleri NORMALIZE oldugundan boyut degisse bile (lowres) gecerli kalir.

Kullanim:
  python training/pseudo_label.py --videos "~/Downloads/Detrac Videos" \
      --out datasets/detrac_pseudo --fps 2 --augment 2 --crops
  python training/pseudo_label.py --videos clip.mp4 --out datasets/dp --max-frames 50
"""
from __future__ import annotations

import argparse
import os
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np

from augment import TRANSFORMS, sample_params  # training/ path'te

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
# COCO arac sinif id'leri: 2=car, 3=motorcycle, 5=bus, 7=truck
COCO_VEHICLES = [2, 3, 5, 7]


def _list_videos(path: str) -> List[str]:
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return [path]
    return [os.path.join(path, f) for f in sorted(os.listdir(path))
            if f.lower().endswith(VIDEO_EXTS)]


def _augment_frame(frame: np.ndarray, k: int = 2) -> Tuple[np.ndarray, str]:
    """Kareye rastgele k hava-donusumu uygula (label-guvenli). (img, etiket) doner."""
    h, w = frame.shape[:2]
    names = random.sample(list(TRANSFORMS.keys()), k=min(k, len(TRANSFORMS)))
    out = frame
    for n in names:
        out = TRANSFORMS[n](out, **sample_params(n, h, w))
    return out, "_".join(names)


def run(videos: str, out: str, model_name: str = "yolo11n.pt",
        fps: float = 2.0, conf: float = 0.35, imgsz: int = 960,
        augment: int = 0, crops: bool = False, max_frames: int = 0) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics gerekli: pip install ultralytics")
        return 1

    vids = _list_videos(videos)
    if not vids:
        print(f"video bulunamadi: {videos}")
        return 1

    img_dir = os.path.join(out, "images", "train")
    lbl_dir = os.path.join(out, "labels", "train")
    crop_dir = os.path.join(out, "crops")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    if crops:
        os.makedirs(crop_dir, exist_ok=True)

    model = YOLO(model_name)  # ilk calismada COCO agirligini indirir
    print(f"[pseudo] {len(vids)} video, model={model_name}, fps={fps}, "
          f"augment={augment}, crops={crops}")

    n_img = n_box = n_crop = 0
    for vi, vpath in enumerate(vids):
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  [atla] acilamadi: {vpath}")
            continue
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(src_fps / max(0.1, fps))))
        stem = os.path.splitext(os.path.basename(vpath))[0]
        fi = kept = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fi % step != 0:
                fi += 1
                continue
            fi += 1
            if max_frames and kept >= max_frames:
                break

            res = model.predict(frame, conf=conf, imgsz=imgsz,
                                classes=COCO_VEHICLES, verbose=False)[0]
            xywhn = res.boxes.xywhn.cpu().numpy() if res.boxes is not None else []
            if len(xywhn) == 0:
                continue  # araci olmayan kareyi atla (gurultu azalt)

            base = f"{stem}_{fi:06d}"
            lines = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in xywhn]

            # 1) orijinal kare + etiket
            cv2.imwrite(os.path.join(img_dir, base + ".jpg"), frame)
            with open(os.path.join(lbl_dir, base + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            n_img += 1
            n_box += len(lines)

            # 2) crop'lar (sonradan tip/renk siniflandirmasi icin)
            if crops:
                H, W = frame.shape[:2]
                xyxy = res.boxes.xyxy.cpu().numpy()
                for bi, (x1, y1, x2, y2) in enumerate(xyxy):
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(W, int(x2)), min(H, int(y2))
                    if x2 - x1 < 16 or y2 - y1 < 16:
                        continue
                    cv2.imwrite(os.path.join(crop_dir, f"{base}_{bi}.jpg"),
                                frame[y1:y2, x1:x2])
                    n_crop += 1

            # 3) hava-augment varyantlari (ayni etiket; normalize bbox degismez)
            for ai in range(augment):
                aug, tag = _augment_frame(frame, k=2)
                an = f"{base}_aug{ai}_{tag}"
                cv2.imwrite(os.path.join(img_dir, an + ".jpg"), aug)
                with open(os.path.join(lbl_dir, an + ".txt"), "w") as f:
                    f.write("\n".join(lines) + "\n")
                n_img += 1
                n_box += len(lines)

            kept += 1
        cap.release()
        print(f"  [{vi+1}/{len(vids)}] {stem}: {kept} kare islendi")

    # data.yaml (tek sinif)
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write("# pseudo_label.py ciktisi — TEK SINIF arac (lokalizasyon)\n")
        f.write(f"path: {os.path.abspath(out)}\n")
        f.write("train: images/train\nval: images/train\n")
        f.write("nc: 1\nnames:\n  0: arac\n")

    print("=" * 52)
    print(f"BITTI -> {out}")
    print(f"  goruntu: {n_img}  (orijinal + {augment}x augment)")
    print(f"  kutu   : {n_box}")
    if crops:
        print(f"  crop   : {n_crop}  (-> tip/renk etiketlemesi icin)")
    print(f"  data.yaml: {os.path.join(out, 'data.yaml')}  (nc=1: arac)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="DETRAC pseudo-label + augment (tek sinif arac)")
    ap.add_argument("--videos", required=True, help="video dosyasi veya klasoru")
    ap.add_argument("--out", default="datasets/detrac_pseudo", help="cikti dataset koku")
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained COCO YOLO (auto-indirir)")
    ap.add_argument("--fps", type=float, default=2.0, help="saniyede ornek kare")
    ap.add_argument("--conf", type=float, default=0.35, help="tespit guven esigi")
    ap.add_argument("--imgsz", type=int, default=960, help="dedektor cikarim boyutu")
    ap.add_argument("--augment", type=int, default=0, help="kare basina hava-augment kopya")
    ap.add_argument("--crops", action="store_true", help="arac crop'larini da kaydet")
    ap.add_argument("--max-frames", type=int, default=0, help="video basina maks kare (0=hepsi)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    return run(args.videos, args.out, model_name=args.model, fps=args.fps,
               conf=args.conf, imgsz=args.imgsz, augment=args.augment,
               crops=args.crops, max_frames=args.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
