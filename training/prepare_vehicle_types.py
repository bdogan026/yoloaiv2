"""prepare_vehicle_types.py — govde-tipi veri setini FTR 7 tipine esler — image'a GIRMEZ.

prepare_vcor.py'nin TIP karsiligi. Ön/yan cekimli bir govde-tipi siniflandirma
seti (Stanford Cars, Roboflow "vehicle type", BoxCars vb.) alir; klasor adlarini
bizim 7 kanonik tipe (schema.ARAC_TIPLERI) esler. Cikti datasets/tip/<tip>/
altinda toplanir -> train_tip.py ile egitilir.

DETRAC tepeden cekim tip icin COKMUSTU (bkz. negatif sonuc); bu yuzden tip icin
ON/YAN cekim seti SART. MIO-TCD zaten pickup/minibus/panelvan/kamyon'u
karsiliyor; ASIL BOSLUK sedan/suv/hatchback -> bu script onlari kapatir.

Iki mod:
  --mode keyword (varsayilan): klasor ADINDA gecen anahtar kelimeye gore esler
     (Stanford Cars: "BMW X5 SUV 2012" -> 'suv', "Audi A4 Sedan" -> 'sedan').
  --mode exact: klasor adi DOGRUDAN kanonik tip ise birebir (Roboflow body-type).

Kullanim:
  # Stanford Cars (Kaggle by-classes-folder): car_data/train/<196 sinif>/
  python training/prepare_vehicle_types.py --src ~/Downloads/stanford_cars --out datasets/tip
  # Roboflow (klasor adlari zaten sedan/suv/...):
  python training/prepare_vehicle_types.py --src ~/Downloads/rf_set --out datasets/tip --mode exact
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.schema import ARAC_TIPLERI  # noqa: E402

CANON = list(ARAC_TIPLERI)  # sedan,suv,hatchback,pickup,minibus,panelvan,kamyon
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# keyword modu: SIRA ÖNEMLI (genel/uzun ozel kelime once; minivan, van'dan once).
# Burayi DEGISTIREBILIRSIN. Eslenmeyen klasorler ATILIR (rapor edilir).
# Bilerek atilan (FTR'de net karsiligi yok): coupe, convertible, wagon.
VTYPE_RULES: List[Tuple[str, str]] = [
    ("minivan", "minibus"),
    ("minibus", "minibus"),
    ("hatchback", "hatchback"),
    ("suv", "suv"),
    ("crew cab", "pickup"),
    ("extended cab", "pickup"),
    ("regular cab", "pickup"),
    ("supercab", "pickup"),
    ("super cab", "pickup"),
    ("pickup", "pickup"),
    ("cab", "pickup"),
    ("panel van", "panelvan"),
    ("cargo van", "panelvan"),
    ("van", "panelvan"),
    ("sedan", "sedan"),
    ("truck", "kamyon"),
    ("bus", "minibus"),
]


def keyword_to_canon(folder_name: str) -> Optional[str]:
    n = folder_name.lower()
    for kw, canon in VTYPE_RULES:
        if kw in n:
            return canon
    return None


def _leaf_class_dirs(src: str):
    """src altinda gorsel iceren YAPRAK klasorleri (sinif klasorleri) bul."""
    for root, dirs, files in os.walk(src):
        has_img = any(f.lower().endswith(IMG_EXTS) for f in files)
        is_leaf = not any(os.path.isdir(os.path.join(root, d)) for d in dirs)
        if has_img and is_leaf and os.path.abspath(root) != os.path.abspath(src):
            yield root


def prepare(src: str, out: str, mode: str, copy: bool,
            limit_per_class: int) -> int:
    if not os.path.isdir(src):
        print(f"HATA: kaynak yok: {src}", file=sys.stderr)
        return 1
    counts: Dict[str, int] = defaultdict(int)
    dropped: Dict[str, int] = defaultdict(int)

    for cdir in _leaf_class_dirs(src):
        cname = os.path.basename(cdir)
        if mode == "exact":
            canon = cname.lower() if cname.lower() in CANON else None
        else:
            canon = keyword_to_canon(cname)
        imgs = [f for f in sorted(os.listdir(cdir)) if f.lower().endswith(IMG_EXTS)]
        if canon is None:
            dropped[cname] += len(imgs)
            continue
        target = os.path.join(out, canon)
        os.makedirs(target, exist_ok=True)
        parent = os.path.basename(os.path.dirname(cdir))  # train/test ayrimi
        for fn in imgs:
            if limit_per_class and counts[canon] >= limit_per_class:
                break
            srcp = os.path.join(cdir, fn)
            dstp = os.path.join(target, f"{parent}__{cname}__{fn}".replace(" ", "_"))
            if os.path.exists(dstp):
                continue
            if copy:
                shutil.copy2(srcp, dstp)
            else:
                os.symlink(os.path.abspath(srcp), dstp)
            counts[canon] += 1

    print("=" * 56)
    print(f"Govde-tipi -> FTR 7 tip  (mode={mode}, out={out})")
    print("-" * 56)
    total = 0
    for c in CANON:
        n = counts.get(c, 0); total += n
        flag = "" if n else "  <-- BOS (bu tip kaynakta yok)"
        print(f"  {c:11}: {n:6d}{flag}")
    print("-" * 56)
    print(f"  TOPLAM     : {total:6d}")
    if dropped:
        print("\nATILAN klasorler (eslenemeyen — coupe/convertible/wagon vb.):")
        for k, v in sorted(dropped.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {k[:34]:34}: {v:5d}")
    print(f"\nSonraki: python training/train_tip.py --data {out} --epochs 40")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="govde-tipi seti -> FTR 7 tip eslestirici")
    ap.add_argument("--src", required=True, help="kaynak kok (sinif alt klasorlu)")
    ap.add_argument("--out", default="datasets/tip", help="cikti tip dataset koku")
    ap.add_argument("--mode", choices=["keyword", "exact"], default="keyword")
    ap.add_argument("--copy", action="store_true", help="kopyala (varsayilan: symlink)")
    ap.add_argument("--limit-per-class", type=int, default=0)
    args = ap.parse_args()
    return prepare(args.src, args.out, args.mode, args.copy, args.limit_per_class)


if __name__ == "__main__":
    raise SystemExit(main())
