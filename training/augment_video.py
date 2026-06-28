"""augment_video.py — ZAMANSAL TUTARLI video augmentasyonu — image'a GIRMEZ.

Neden ayri? YOLO dedektoru kare bazli egitilir; orada goruntu augmentasyonu
(augment.py) yeterlidir. Ama bu sistemin ZAMANSAL modulleri var (slalom
yorungesi, etrafa_bakinma salinim penceresi, olay debounce). Video augment
ederken her kareye BAGIMSIZ rastgele parametre uygulanirsa sis/parlaklik kare
kare titrer (flicker) ve bu zamansal sinyaller bozulur.

Cozum: augmentasyon parametreleri KLIP basina BIR KEZ sabitlenir; istenirse
yavas DRIFT (sin tabanli) ile dogal degisim (orn. sis yavasca yogunlasir).
Yagmur damlalari her karede yeniden uretilir (dogal hareket) ama aci/siddet
sabit kalir.

Kullanimlar:
  * Gurbuzluk/demo: temiz videodan sisli/yagmurlu/dusuk-isik klipler uret.
  * Zamansal dogrulama seti: ayni klibin zorlu kosul varyantlari.
  * Cikarim ablasyonu: ayni videoyu farkli kosullarda calistir, sonuc karsilastir.

NOT: Donusumler fotometrik/bulaniklik tabanlidir; bbox koordinatlari degismez.
Videodan kare cikarip egitim kullanacaksan ayni YOLO etiketleri gecerlidir.

Ornek:
  python training/augment_video.py --input demo.mp4 --out demo_fog.mp4 \
      --transforms fog,light --drift
  python training/augment_video.py --input-dir clips/ --out-dir clips_aug/ \
      --transforms random --per-video 2
"""
from __future__ import annotations

import argparse
import math
import os
import random
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from augment import TRANSFORMS, sample_params

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


class ClipAugmentor:
    """Bir klip icin sabit parametreli (opsiyonel drift'li) augmentasyon."""

    def __init__(self, names: List[str], h: int, w: int,
                 drift: bool = False, drift_period_s: float = 8.0,
                 fps: float = 25.0) -> None:
        self.names = names
        self.drift = drift
        self.drift_period = max(1.0, drift_period_s)
        self.fps = max(1e-6, fps)
        # KLIP basina bir kez parametre ornekle (zamansal tutarlilik)
        self.params: Dict[str, Dict[str, Any]] = {
            n: sample_params(n, h, w) for n in names
        }

    def _drift_factor(self, frame_idx: int) -> float:
        """0.7..1.3 arasi yavas modulasyon (drift kapaliysa 1.0)."""
        if not self.drift:
            return 1.0
        t = frame_idx / self.fps
        return 1.0 + 0.3 * math.sin(2 * math.pi * t / self.drift_period)

    def apply(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        f = self._drift_factor(frame_idx)
        out = frame
        for n in self.names:
            p = dict(self.params[n])  # kopya (drift uygula)
            if self.drift:
                if n == "fog" and "beta" in p:
                    p["beta"] = max(0.1, p["beta"] * f)        # sis yogunlugu drift
                elif n == "light" and "beta" in p:
                    p["beta"] = p["beta"] * f                   # parlaklik drift
            out = TRANSFORMS[n](out, **p)
        return out


def _resolve_transforms(spec: str, k: int) -> List[str]:
    if spec == "random":
        return random.sample(list(TRANSFORMS.keys()), k=min(k, len(TRANSFORMS)))
    names = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in TRANSFORMS]
    if bad:
        raise SystemExit(f"Bilinmeyen donusum(ler): {bad}. "
                         f"Secenekler: {list(TRANSFORMS.keys())}")
    return names


def augment_one(input_path: str, out_path: str, transforms_spec: str,
                k: int = 2, drift: bool = False,
                fourcc: str = "mp4v") -> Optional[str]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[video] acilamadi: {input_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w == 0 or h == 0:
        cap.release()
        print(f"[video] gecersiz boyut: {input_path}")
        return None

    names = _resolve_transforms(transforms_spec, k)
    aug = ClipAugmentor(names, h, w, drift=drift, fps=fps)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc),
                             fps, (w, h))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(aug.apply(frame, idx))
        idx += 1
    cap.release()
    writer.release()
    print(f"[video] {input_path} -> {out_path} "
          f"({idx} kare, donusumler={'+'.join(names)}, drift={drift})")
    return out_path


def augment_dir(input_dir: str, out_dir: str, transforms_spec: str,
                per_video: int = 1, k: int = 2, drift: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    vids = [f for f in os.listdir(input_dir)
            if f.lower().endswith(VIDEO_EXTS)]
    print(f"[video] {len(vids)} video, her birinden {per_video} varyant.")
    for v in vids:
        stem, ext = os.path.splitext(v)
        for i in range(per_video):
            names = _resolve_transforms(transforms_spec, k)
            tag = "_".join(names)
            out = os.path.join(out_dir, f"{stem}_syn{i}_{tag}{ext}")
            augment_one(os.path.join(input_dir, v), out,
                        ",".join(names), k=k, drift=drift)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Zamansal tutarli video augmentasyonu")
    ap.add_argument("--input", help="tek video yolu")
    ap.add_argument("--out", help="cikti video yolu (--input ile)")
    ap.add_argument("--input-dir", help="video klasoru (toplu mod)")
    ap.add_argument("--out-dir", help="cikti klasoru (--input-dir ile)")
    ap.add_argument("--transforms", default="random",
                    help="virgulle: fog,rain,light,motion,lowres,jpeg | random")
    ap.add_argument("--k", type=int, default=2,
                    help="random secimde donusum sayisi")
    ap.add_argument("--per-video", type=int, default=1,
                    help="toplu modda her videodan kac varyant")
    ap.add_argument("--drift", action="store_true",
                    help="zamansal yavas drift (sis/parlaklik)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.input and args.out:
        augment_one(args.input, args.out, args.transforms, k=args.k,
                    drift=args.drift)
    elif args.input_dir and args.out_dir:
        augment_dir(args.input_dir, args.out_dir, args.transforms,
                    per_video=args.per_video, k=args.k, drift=args.drift)
    else:
        ap.error("Ya (--input & --out) ya da (--input-dir & --out-dir) verin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
