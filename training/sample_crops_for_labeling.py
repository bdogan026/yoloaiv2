"""sample_crops_for_labeling.py — tip etiketlemesi icin crop ornekle — image'a GIRMEZ.

pseudo_label.py'nin urettigi arac crop'larindan, govde tipi GORUNUR olacak
sekilde (yeterince buyuk) dengeli bir ornek secip elle siniflama icin hazirlar.

Cikti:
  <out>/{sedan,suv,hatchback,pickup,minibus,panelvan,kamyon}/   (BOS — sen doldur)
  <unsorted>/                                                   (ornek crop'lar)

Is akisi:
  1) Bu script'i calistir.
  2) <unsorted> klasorunu Finder'da ac; her crop'u dogru tip klasorune SURUKLE
     (<out>/sedan, <out>/suv, ...). Emin olmadigini ATLA (silebilirsin).
  3) python training/train_tip.py --data <out>

DURUST NOT: DETRAC tepeden/uzak cekim -> sedan/suv/hatchback ayrimi cogu karede
ZOR. Net gorunenleri etiketle; kamyon/minibus/pickup/panelvan tepeden de ayirt
edilebilir. Ince uclu (sedan/suv/hatchback) icin on/yan cekim (BoxCars ya da
yarisma videolari) daha iyi kaynak.

Kullanim:
  python training/sample_crops_for_labeling.py --n 700 --min-side 110
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.schema import ARAC_TIPLERI  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png")


def main() -> int:
    ap = argparse.ArgumentParser(description="tip etiketlemesi icin crop ornekle")
    ap.add_argument("--crops", default="datasets/detrac_pseudo/crops",
                    help="pseudo_label.py crop klasoru")
    ap.add_argument("--out", default="datasets/tip",
                    help="etiketli tip dataset koku (7 bos sinif klasoru olusur)")
    ap.add_argument("--unsorted", default="datasets/tip_unsorted",
                    help="ornek crop'larin konacagi (elle ayiklanacak) klasor")
    ap.add_argument("--n", type=int, default=700, help="ornek crop sayisi")
    ap.add_argument("--min-side", type=int, default=110,
                    help="min kenar (px) — kucuk/uzak araclar tip icin elenir")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        print("opencv gerekli: pip install opencv-python", file=sys.stderr)
        return 1

    if not os.path.isdir(args.crops):
        print(f"crop klasoru yok: {args.crops}", file=sys.stderr)
        return 1

    # 7 bos sinif klasoru (egitim hedefi)
    for t in ARAC_TIPLERI:
        os.makedirs(os.path.join(args.out, t), exist_ok=True)
    os.makedirs(args.unsorted, exist_ok=True)

    files = [f for f in os.listdir(args.crops) if f.lower().endswith(IMG_EXTS)]
    random.seed(args.seed)
    random.shuffle(files)

    picked = 0
    for f in files:
        if picked >= args.n:
            break
        src = os.path.join(args.crops, f)
        img = cv2.imread(src)
        if img is None:
            continue
        h, w = img.shape[:2]
        if min(h, w) < args.min_side:
            continue  # tip icin cok kucuk
        shutil.copy2(src, os.path.join(args.unsorted, f))
        picked += 1

    print("=" * 56)
    print(f"{picked} crop -> {args.unsorted}  (min-kenar>={args.min_side}px)")
    print(f"Bos sinif klasorleri -> {args.out}/{{{','.join(ARAC_TIPLERI)}}}")
    print("\nSiradaki:")
    print(f"  1) Finder'da ac: open {args.unsorted}")
    print(f"  2) Her crop'u dogru tip klasorune surukle ({args.out}/sedan, ...)")
    print(f"  3) python training/train_tip.py --data {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
