"""augment.py — SENTETIK veri uretimi (gurbuzluk icin) — image'a GIRMEZ.

Gercek algoritmalarla zorlu kosul sentezi (OpenCV/numpy, ekstra bagimlilik yok):
  * sis (atmosferik sacilim modeli: I = J*t + A*(1-t))
  * yagmur (hareket-bulanik gurultu cizgileri)
  * parlaklik/kontrast/gamma
  * hareket bulanikligi (motion blur)
  * cozunurluk dususu (low-res simulasyonu)
  * JPEG bozulmasi

Tum donusumler FOTOMETRIK/bulaniklik tabanlidir; bbox koordinatlarini
DEGISTIRMEZ -> YOLO etiketleri (.txt) oldugu gibi kopyalanir. Bu sayede
"sentetikli vs sentetiksiz" ablasyonu (Gelistirme notu) dogrudan yapilabilir.

Her donusum acik PARAMETRE alabilir (None -> rastgele orneklenir). Bu sayede:
  * Goruntu modu: her ornekte rastgele parametre (bu dosya).
  * Video modu: KLIP basina sabit parametre + zamansal tutarlilik
    (bkz. augment_video.py) -> flicker yok, zamansal sinyaller bozulmaz.

Kullanim (goruntu seti):
    python training/augment.py --images datasets/raw/images \
        --labels datasets/raw/labels --out datasets/synth --per-image 2
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ----------------------------------------------------------------------------
# Yardimci: radyal derinlik haritasi (sis icin; video'da bir kez hesaplanir)
# ----------------------------------------------------------------------------
def radial_depth(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return (d / d.max()).astype(np.float32)


# ----------------------------------------------------------------------------
# Tekil donusumler (parametreler None ise rastgele orneklenir)
# ----------------------------------------------------------------------------
def add_fog(img: np.ndarray, beta: Optional[float] = None,
            A: Optional[float] = None,
            depth: Optional[np.ndarray] = None) -> np.ndarray:
    """Atmosferik sacilim: I = J*t + A*(1-t), t = exp(-beta*d)."""
    if beta is None:
        beta = random.uniform(0.6, 1.6)
    if A is None:
        A = random.uniform(180, 235)
    if depth is None:
        depth = radial_depth(*img.shape[:2])
    t = np.exp(-beta * (0.3 + depth))[..., None]
    out = img.astype(np.float32) * t + A * (1.0 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_rain(img: np.ndarray, amount: Optional[float] = None,
             length: Optional[int] = None,
             angle: Optional[float] = None) -> np.ndarray:
    """Hareket-bulanik gurultu cizgileriyle yagmur.

    NOT: damla konumlari HER cagrida yeniden uretilir (video'da dogal hareket);
    amount/length/angle sabit tutulunca yagmurun siddeti/yonu klip boyunca tutarli."""
    if amount is None:
        amount = random.uniform(0.002, 0.01)
    if length is None:
        length = random.randint(10, 22)
    if angle is None:
        angle = random.uniform(-20, 20)
    h, w = img.shape[:2]
    rain = np.zeros((h, w), np.uint8)
    n = int(amount * h * w)
    xs = np.random.randint(0, w, n)
    ys = np.random.randint(0, h, n)
    rain[ys, xs] = np.random.randint(160, 255, n).astype(np.uint8)
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    k /= k.sum() + 1e-6
    rain = cv2.filter2D(rain, -1, k)
    rain3 = cv2.cvtColor(rain, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(img, 0.8, rain3, 0.5, 0)
    return cv2.GaussianBlur(out, (3, 3), 0)


def adjust_light(img: np.ndarray, alpha: Optional[float] = None,
                 beta: Optional[float] = None,
                 gamma: Optional[float] = None) -> np.ndarray:
    """Parlaklik/kontrast/gamma."""
    if alpha is None:
        alpha = random.uniform(0.6, 1.4)
    if beta is None:
        beta = random.uniform(-40, 40)
    if gamma is None:
        gamma = random.uniform(0.6, 1.6)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                    for i in range(256)]).astype(np.uint8)
    return cv2.LUT(out, lut)


def motion_blur(img: np.ndarray, size: Optional[int] = None,
                horizontal: Optional[bool] = None) -> np.ndarray:
    if size is None:
        size = random.choice([5, 7, 9, 11])
    if horizontal is None:
        horizontal = random.random() < 0.5
    k = np.zeros((size, size), np.float32)
    if horizontal:
        k[size // 2, :] = 1.0
    else:
        k[:, size // 2] = 1.0
    k /= size
    return cv2.filter2D(img, -1, k)


def downscale(img: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
    """Dusuk cozunurluk simulasyonu (kucult-buyut)."""
    if factor is None:
        factor = random.uniform(0.3, 0.6)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def jpeg_corrupt(img: np.ndarray, quality: Optional[int] = None) -> np.ndarray:
    if quality is None:
        quality = random.randint(15, 45)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


TRANSFORMS: Dict[str, Callable[..., np.ndarray]] = {
    "fog": add_fog,
    "rain": add_rain,
    "light": adjust_light,
    "motion": motion_blur,
    "lowres": downscale,
    "jpeg": jpeg_corrupt,
}


# ----------------------------------------------------------------------------
# Parametre orneklayicilari (klip/goruntu basina sabit param uretmek icin)
# ----------------------------------------------------------------------------
def sample_params(name: str, h: int, w: int) -> Dict[str, Any]:
    """Bir donusum icin tek seferlik parametre seti uretir."""
    if name == "fog":
        return {"beta": random.uniform(0.6, 1.6),
                "A": random.uniform(180, 235),
                "depth": radial_depth(h, w)}
    if name == "rain":
        return {"amount": random.uniform(0.002, 0.01),
                "length": random.randint(10, 22),
                "angle": random.uniform(-20, 20)}
    if name == "light":
        return {"alpha": random.uniform(0.6, 1.4),
                "beta": random.uniform(-40, 40),
                "gamma": random.uniform(0.6, 1.6)}
    if name == "motion":
        return {"size": random.choice([5, 7, 9, 11]),
                "horizontal": random.random() < 0.5}
    if name == "lowres":
        return {"factor": random.uniform(0.3, 0.6)}
    if name == "jpeg":
        return {"quality": random.randint(15, 45)}
    return {}


def random_pipeline(img: np.ndarray, k: int = 2) -> Tuple[np.ndarray, List[str]]:
    """Goruntu modu: k rastgele donusum, her biri rastgele parametreyle."""
    names = random.sample(list(TRANSFORMS.keys()), k=min(k, len(TRANSFORMS)))
    out = img
    for n in names:
        out = TRANSFORMS[n](out)  # parametreler None -> rastgele
    return out, names


# ----------------------------------------------------------------------------
# Veri seti uzerinde toplu uretim (GORUNTU; etiketler kopyalanir)
# ----------------------------------------------------------------------------
def process_dataset(images_dir: str, labels_dir: str, out_dir: str,
                    per_image: int = 2, transforms_per: int = 2) -> None:
    out_img = os.path.join(out_dir, "images")
    out_lbl = os.path.join(out_dir, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    files = [f for f in os.listdir(images_dir)
             if f.lower().endswith(IMG_EXTS)]
    print(f"[augment] {len(files)} goruntu, her birinden {per_image} sentetik.")
    made = 0
    for fname in files:
        img = cv2.imread(os.path.join(images_dir, fname))
        if img is None:
            continue
        stem, ext = os.path.splitext(fname)
        label_src = os.path.join(labels_dir, stem + ".txt")
        for i in range(per_image):
            aug, names = random_pipeline(img, k=transforms_per)
            tag = "_".join(names)
            new_stem = f"{stem}_syn{i}_{tag}"
            cv2.imwrite(os.path.join(out_img, new_stem + ext), aug)
            if os.path.exists(label_src):
                shutil.copy(label_src, os.path.join(out_lbl, new_stem + ".txt"))
            made += 1
    print(f"[augment] {made} sentetik ornek uretildi -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Sentetik GORUNTU verisi uretimi "
                                             "(video icin: augment_video.py)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-image", type=int, default=2)
    ap.add_argument("--transforms-per", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    process_dataset(args.images, args.labels, args.out,
                    args.per_image, args.transforms_per)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
