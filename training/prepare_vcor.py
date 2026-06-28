"""prepare_vcor.py — VCoR (Vehicle Color Recognition) -> FTR 9 renk klasoru
eslestirici — image'a GIRMEZ (yalnizca egitim veri hazirligi).

train_color.py, --data altinda TAM olarak schema.ARAC_RENKLERI isimli 9 alt
klasor bekler (beyaz/, siyah/, ...). CANON disi her klasor egitimde sinif-0'a
(beyaz) dusup etiketi bozar. Bu script VCoR'un renk klasorlerini bizim 9'a
esler, FTR'de OLMAYAN renkleri (pembe/mor/bej/altin...) ATAR.

VCoR genelde train/val/test altinda renk alt klasorludur; bu script tum bu
agaclari tarar, eslenen renkleri tek bir cikti agacinda toplar.

Kullanim:
    python training/prepare_vcor.py --src ~/Downloads/VCoR --out datasets/colors
    python training/prepare_vcor.py --src ~/Downloads/VCoR --out datasets/colors \
        --mode symlink --limit-per-class 1500
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.schema import ARAC_RENKLERI  # noqa: E402

CANON = list(ARAC_RENKLERI)  # ['beyaz','siyah','gri','kirmizi','mavi','sari','yesil','turuncu','kahverengi']

# VCoR (ve benzeri) renk adi -> bizim kanonik renk. Burada DEGISTIREBILIRSIN.
# Esleme disinda kalan VCoR renkleri (pink/purple/beige/gold/cyan...) ATILIR.
VCOR_MAP: Dict[str, str] = {
    "white": "beyaz",
    "black": "siyah",
    "grey": "gri",
    "gray": "gri",
    "silver": "gri",     # gumus govde rengi pratikte gri sayilir
    "red": "kirmizi",
    "blue": "mavi",
    "yellow": "sari",
    "green": "yesil",
    "orange": "turuncu",
    "brown": "kahverengi",
    "tan": "kahverengi",  # tan = acik kahve
    # --- bilerek ATILANLAR (FTR 9'da yok; etiket gurultusu yapmasin) ---
    # "beige", "gold", "pink", "purple", "cyan"
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _iter_color_dirs(src: str):
    """src altinda, basename'i VCOR_MAP'te olan TUM klasorleri bul (train/val/test dahil)."""
    for root, dirs, _files in os.walk(src):
        base = os.path.basename(root).lower()
        if base in VCOR_MAP:
            yield root, VCOR_MAP[base]


def prepare(src: str, out: str, mode: str = "copy",
            limit_per_class: int = 0) -> int:
    if not os.path.isdir(src):
        print(f"HATA: kaynak yok: {src}", file=sys.stderr)
        return 1

    # cikti: her kanonik renk icin klasor
    counts: Dict[str, int] = defaultdict(int)
    dropped: Dict[str, int] = defaultdict(int)
    seen_canon: List[str] = []

    # once ATILAN renkleri de raporlamak icin tum renk klasorlerini tara
    for root, _dirs, _files in os.walk(src):
        base = os.path.basename(root).lower()
        if base not in VCOR_MAP and any(
                f.lower().endswith(IMG_EXTS) for f in os.listdir(root)) \
                and base not in ("", ) and root != src:
            # sadece "renk gibi gorunen" yaprak klasorleri say (gurultu olmasin)
            n = sum(1 for f in os.listdir(root) if f.lower().endswith(IMG_EXTS))
            # train/val/test gibi ust klasorleri atlamak icin: alt klasoru yoksa yaprak
            if not any(os.path.isdir(os.path.join(root, d)) for d in _dirs):
                if base not in VCOR_MAP:
                    dropped[base] += n

    for color_dir, canon in _iter_color_dirs(src):
        target = os.path.join(out, canon)
        os.makedirs(target, exist_ok=True)
        if canon not in seen_canon:
            seen_canon.append(canon)
        src_tag = os.path.basename(os.path.dirname(color_dir))  # train/val/test ayirimi
        for fn in sorted(os.listdir(color_dir)):
            if not fn.lower().endswith(IMG_EXTS):
                continue
            if limit_per_class and counts[canon] >= limit_per_class:
                break
            src_path = os.path.join(color_dir, fn)
            # cakismayi onle: kaynak-split + orijinal ad
            dst_name = f"{src_tag}__{os.path.basename(color_dir)}__{fn}"
            dst_path = os.path.join(target, dst_name)
            if os.path.exists(dst_path):
                continue
            if mode == "symlink":
                os.symlink(os.path.abspath(src_path), dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            counts[canon] += 1

    # rapor
    print("=" * 56)
    print(f"VCoR -> FTR 9 renk  (mode={mode}, out={out})")
    print("-" * 56)
    total = 0
    for c in CANON:
        n = counts.get(c, 0)
        total += n
        flag = "" if n > 0 else "  <-- BOS! (VCoR'da bu renk yok/eslesemedi)"
        print(f"  {c:11} : {n:6d}{flag}")
    print("-" * 56)
    print(f"  TOPLAM      : {total:6d}")
    if dropped:
        print("\nATILAN renkler (FTR 9'da yok):")
        for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"  {k:11} : {v:6d}")
    empty = [c for c in CANON if counts.get(c, 0) == 0]
    if empty:
        print(f"\nUYARI: bos sinif(lar) {empty} -> bu renkler icin VCoR'da veri "
              f"yok; ek crop ekle yoksa model bu rengi ogrenemez.", file=sys.stderr)
    print("\nSonraki adim:")
    print(f"  python training/train_color.py --data {out} --epochs 30")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="VCoR -> FTR 9 renk klasoru eslestirici")
    ap.add_argument("--src", required=True, help="VCoR kok dizini (train/val/test icerebilir)")
    ap.add_argument("--out", default="datasets/colors", help="cikti renk-klasorlu dizin")
    ap.add_argument("--mode", choices=["copy", "symlink"], default="copy",
                    help="copy (guvenli) | symlink (yer tasarrufu)")
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="renk basina maks gorsel (dengeleme; 0=sinirsiz)")
    args = ap.parse_args()
    return prepare(args.src, args.out, mode=args.mode,
                   limit_per_class=args.limit_per_class)


if __name__ == "__main__":
    raise SystemExit(main())
