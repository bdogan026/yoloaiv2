"""pseudo_label_committee.py — Komite videolarini (gercek KARANLIK 4K domain)
best_model.pt ile 9-sinif pseudo-label'lar -> domain adaptasyon egitim seti.
image'a GIRMEZ.

Neden: model bu domaini yalnizca yuksek cozunurlukte (imgsz=1920) + CLAHE ile
dogru tespit ediyor (640'ta kayboluyor). Yuksek-guvenli tespitleri RAW kare +
YOLO etiketi olarak kaydedip egitime ekleyince model gercek domaini DUSUK
cozunurlukte de ogrenir (genelleme + kucuk imgsz'de calisma).

ONEMLI:
  * Tespit CLAHE'li @imgsz ile yapilir (kucuk nesne icin sart), ama RAW kare
    kaydedilir (egitim domaini tutarli; CLAHE inference'ta zaten uygulaniyor).
  * Sinif id'leri zaten 9-sinif semasinda (best_model.pt.names sirasi) -> REMAP YOK.
  * conf YUKSEK tutulur (gurultu/yanlis etiket girmesin); self-training kalitesi
    icin kritik.

Kullanim:
  python training/pseudo_label_committee.py \
      --videos ~/Downloads/teknocan/video_2.mp4 ~/Downloads/teknocan/video_3.mp4 \
      --weights weights/best_model.pt --out datasets/committee_pseudo \
      --fps 5 --conf 0.45 --imgsz 1920
"""
from __future__ import annotations

import argparse
import collections
import os

import cv2

NAMES = ['arac', 'plaka', 'teknocan', 'bilgisayar', 'telefon',
         'sigara', 'sise', 'emniyet_kemeri', 'kisi']


def clahe_bgr(fr):
    lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L = cv2.createCLAHE(2.5, (8, 8)).apply(L)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)


def run(videos, out, weights, fps, conf, imgsz):
    from ultralytics import YOLO
    m = YOLO(weights)
    img_dir = f"{out}/images/train"
    lbl_dir = f"{out}/labels/train"
    viz_dir = f"{out}/viz"
    for d in (img_dir, lbl_dir, viz_dir):
        os.makedirs(d, exist_ok=True)

    cnt = collections.Counter()
    n_img = n_box = nviz = 0
    for vp in videos:
        vp = os.path.expanduser(vp)
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"  [atla] acilamadi: {vp}")
            continue
        src = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(round(src / max(0.1, fps))))
        stem = os.path.splitext(os.path.basename(vp))[0]
        fi = kept = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if fi % step != 0:
                fi += 1
                continue
            fi += 1
            r = m.predict(clahe_bgr(fr), conf=conf, imgsz=imgsz, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xywhn = r.boxes.xywhn.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                     for c, (x, y, w, h) in zip(cls, xywhn)]
            base = f"{stem}_{fi:05d}"
            cv2.imwrite(f"{img_dir}/{base}.jpg", fr)            # RAW kare
            with open(f"{lbl_dir}/{base}.txt", "w") as f:
                f.write("\n".join(lines) + "\n")
            for c in cls:
                cnt[int(c)] += 1
            n_img += 1
            n_box += len(lines)
            kept += 1
            if nviz < 10:
                cv2.imwrite(f"{viz_dir}/{base}.jpg", r.plot())
                nviz += 1
        cap.release()
        print(f"  {stem}: {kept} etiketli kare")

    with open(f"{out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(out)}\ntrain: images/train\n"
                f"val: images/train\nnc: 9\nnames: {NAMES}\n")
    print(f"\nBITTI -> {out}: {n_img} kare, {n_box} kutu")
    print("=== sinif basina kutu ===")
    for i, nm in enumerate(NAMES):
        if cnt[i]:
            print(f"  {i} {nm:16} {cnt[i]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Komite videosu 9-sinif pseudo-label")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--weights", default="weights/best_model.pt")
    ap.add_argument("--out", default="datasets/committee_pseudo")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=1920)
    args = ap.parse_args()
    return run(args.videos, args.out, args.weights, args.fps, args.conf, args.imgsz)


if __name__ == "__main__":
    raise SystemExit(main())
