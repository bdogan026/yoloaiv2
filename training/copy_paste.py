"""copy_paste.py — nadir nesneleri gercek sahnelere yapistirip YOLO-etiketli
sentetik veri uretir (Simple Copy-Paste). Image'a GIRMEZ.

teknocan / bilgisayar gibi domain-uyumlu hazir verisi olmayan siniflar icin:
az sayida nesne crop'unu + gercek sahne karelerini (komite videolari) birlestirir.

Nesne gorselleri: --objects altinda her sinif icin alt klasor (teknocan/, bilgisayar/).
  RGBA PNG ise alpha maske ile harmanlanir; degilse seamlessClone ile yapistirilir.
Arka planlar: --backgrounds altindaki sahne kareleri.
ROI (opsiyonel): nesnenin yapistirilacagi bolge (norm. x1,y1,x2,y2) — or. cam bolgesi.
Cikti: --out/images + --out/labels (YOLO). Sinif id'leri data.yaml'dan.

Kullanim:
  python training/copy_paste.py --objects datasets/obj --backgrounds datasets/frames \
      --data-yaml training/data.yaml --out datasets/synth_obj --per-bg 3
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_names(data_yaml: str) -> Dict[str, int]:
    d = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    names = d["names"]
    if isinstance(names, dict):
        return {str(v): int(k) for k, v in names.items()}
    return {str(n): i for i, n in enumerate(names)}


def list_imgs(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]


def load_objects(objects_dir: Path, name2id: Dict[str, int]):
    """{sinif_id: [ (bgr, alpha|None), ... ]}  alt klasor adi = sinif."""
    out: Dict[int, list] = {}
    for sub in sorted(objects_dir.iterdir()):
        if not sub.is_dir() or sub.name not in name2id:
            continue
        cid = name2id[sub.name]
        for p in list_imgs(sub):
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            if im.ndim == 3 and im.shape[2] == 4:
                bgr, a = im[:, :, :3], im[:, :, 3]
            else:
                bgr, a = im[:, :, :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), None
            out.setdefault(cid, []).append((bgr, a))
    return out


def paste_one(bg: np.ndarray, obj: Tuple[np.ndarray, np.ndarray],
              roi: Tuple[float, float, float, float],
              scale_range: Tuple[float, float] = (0.06, 0.20)) -> Tuple[int, float, float, float, float]:
    """Bir nesneyi bg uzerine yapistirir, YOLO (cls disinda) kutu doner."""
    H, W = bg.shape[:2]
    bgr, alpha = obj
    oh, ow = bgr.shape[:2]
    # hedef genislik: bg genisliginin rastgele orani
    tw = int(W * random.uniform(*scale_range))
    th = max(1, int(tw * oh / ow))
    bgr_r = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
    # ROI icinde rastgele konum
    rx1, ry1, rx2, ry2 = roi
    x1 = int(random.uniform(rx1 * W, max(rx1 * W, rx2 * W - tw)))
    y1 = int(random.uniform(ry1 * H, max(ry1 * H, ry2 * H - th)))
    x1 = max(0, min(x1, W - tw)); y1 = max(0, min(y1, H - th))
    region = bg[y1:y1 + th, x1:x1 + tw]
    if alpha is not None:
        a = cv2.resize(alpha, (tw, th)).astype(np.float32)[..., None] / 255.0
        bg[y1:y1 + th, x1:x1 + tw] = (bgr_r * a + region * (1 - a)).astype(np.uint8)
    else:
        mask = np.full((th, tw), 255, np.uint8)
        center = (x1 + tw // 2, y1 + th // 2)
        try:
            bg[:] = cv2.seamlessClone(bgr_r, bg, mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            bg[y1:y1 + th, x1:x1 + tw] = bgr_r  # fallback duz yapistir
    xc = (x1 + tw / 2) / W; yc = (y1 + th / 2) / H
    return None, xc, yc, tw / W, th / H


def main() -> int:
    ap = argparse.ArgumentParser(description="Copy-paste nesne augmentasyonu")
    ap.add_argument("--objects", required=True, help="sinif alt klasorlu nesne dizini")
    ap.add_argument("--backgrounds", required=True, help="sahne kareleri dizini")
    ap.add_argument("--data-yaml", default="training/data.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bg", type=int, default=3, help="arka plan basina uretim")
    ap.add_argument("--max-objs", type=int, default=2, help="kare basina max nesne")
    ap.add_argument("--roi", default="0,0,1,1", help="yapistirma bolgesi: x1,y1,x2,y2 (norm)")
    ap.add_argument("--scale-min", type=float, default=0.06, help="nesne genisligi / bg (alt)")
    ap.add_argument("--scale-max", type=float, default=0.20, help="nesne genisligi / bg (ust)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    name2id = load_names(args.data_yaml)
    objs = load_objects(Path(args.objects), name2id)
    if not objs:
        print("Nesne bulunamadi (alt klasor adlari data.yaml sinifi olmali).")
        return 1
    roi = tuple(float(v) for v in args.roi.split(","))
    bgs = list_imgs(Path(args.backgrounds))
    out_img = Path(args.out) / "images"; out_lbl = Path(args.out) / "labels"
    out_img.mkdir(parents=True, exist_ok=True); out_lbl.mkdir(parents=True, exist_ok=True)

    all_cids = list(objs)
    made = 0
    for bgp in bgs:
        base = cv2.imread(str(bgp))
        if base is None:
            continue
        for i in range(args.per_bg):
            canvas = base.copy()
            lines = []
            for _ in range(random.randint(1, args.max_objs)):
                cid = random.choice(all_cids)
                obj = random.choice(objs[cid])
                _, xc, yc, w, h = paste_one(canvas, obj, roi,
                                            scale_range=(args.scale_min, args.scale_max))
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            stem = f"{bgp.stem}_cp{i}"
            cv2.imwrite(str(out_img / (stem + ".jpg")), canvas)
            (out_lbl / (stem + ".txt")).write_text("\n".join(lines))
            made += 1
    print(f"[copy_paste] {made} sentetik kare -> {args.out}  (siniflar: "
          f"{[k for k in name2id if name2id[k] in objs]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
