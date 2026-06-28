"""prepare_dataset.py — kaynak datasetleri 15-sinifli birlesik YOLO setine cevirir.

Image'a GIRMEZ (sadece gelistirme makinesinde calisir).

Ne yapar:
  - Her kaynagi (format: yolo | coco) okur, siniflarini HEDEF sema adina remap eder
    (data.yaml'daki 15 sinif tek otorite; isimle eslersin, id ezberlemezsin).
  - Eslenmeyen siniflari ATAR. Hicbir hedef kutu kalmayan goruntuyu (varsayilan) atar.
  - Hepsini birlestirip train/val/test'e boler (varsayilan 0.70/0.15/0.15).
  - `val_only: true` kaynaklar (komite videolari) tamamen VAL'e gider — gercek-domain capa.
  - Cikti: <out>/images/{train,val,test} + <out>/labels/{train,val,test}  (data.yaml uyumlu)
  - Sinif histogramini basar, eksik/bos sinif uyarir.

Kullanim:
  python training/prepare_dataset.py \
      --sources training/sources.yaml \
      --data-yaml training/data.yaml \
      --out ../datasets/rapid_response \
      --split 0.70 0.15 0.15 --seed 42

Kaynak formati icin: training/sources.example.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# (image_path, [(cls_id, xc, yc, w, h), ...])  -- hepsi YOLO normalize [0,1]
Item = Tuple[Path, List[Tuple[int, float, float, float, float]]]


def load_target_names(data_yaml: str) -> Dict[str, int]:
    """data.yaml'daki names -> {sinif_adi: id} (tek otorite)."""
    with open(data_yaml, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    names = d["names"]
    if isinstance(names, dict):
        return {str(v): int(k) for k, v in names.items()}
    return {str(n): i for i, n in enumerate(names)}


def _resolve_labels_dir(images_dir: Path, labels_dir: Optional[str]) -> Path:
    if labels_dir:
        return Path(labels_dir)
    # YOLO konvansiyonu: .../images/... -> .../labels/...
    p = str(images_dir)
    if os.sep + "images" in p:
        return Path(p.replace(os.sep + "images", os.sep + "labels"))
    return images_dir.parent / "labels"


def read_yolo_source(src: dict, name2id: Dict[str, int], cap: int = 0) -> List[Item]:
    """YOLO formatli kaynak: images/ + labels/*.txt. `names` (kaynak id sirasi)
    ve `map` (kaynak_adi -> hedef_adi) ile remap eder."""
    images_dir = Path(src["images"])
    labels_dir = _resolve_labels_dir(images_dir, src.get("labels"))
    src_names: List[str] = src.get("names") or []
    cmap: Dict[str, str] = src.get("map", {})
    # kaynak sinif id -> hedef id (None = at)
    id_to_target: Dict[int, Optional[int]] = {}
    for i, sn in enumerate(src_names):
        tgt = cmap.get(sn)
        id_to_target[i] = name2id[tgt] if tgt in name2id else None

    # HIZ: etiketleri okumadan ONCE goruntu listesini cap'le -> Drive okumayi azaltir
    imgs = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if cap and len(imgs) > cap:
        imgs = random.sample(imgs, cap)
    items: List[Item] = []
    for img in imgs:
        lbl = labels_dir / (img.stem + ".txt")
        boxes: List[Tuple[int, float, float, float, float]] = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                sid = int(float(parts[0]))
                tgt = id_to_target.get(sid)
                if tgt is None:
                    continue
                xc, yc, w, h = map(float, parts[1:5])
                boxes.append((tgt, xc, yc, w, h))
        items.append((img, boxes))
    return items


def read_coco_source(src: dict, name2id: Dict[str, int], cap: int = 0) -> List[Item]:
    """COCO instances.json kaynagi. `map` (coco_kategori_adi -> hedef_adi)."""
    images_dir = Path(src["images"])
    with open(src["ann"], "r", encoding="utf-8") as f:
        coco = json.load(f)
    cmap: Dict[str, str] = src.get("map", {})
    cat_id_to_target: Dict[int, Optional[int]] = {}
    for c in coco["categories"]:
        tgt = cmap.get(c["name"])
        cat_id_to_target[c["id"]] = name2id[tgt] if tgt in name2id else None

    img_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_img: Dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    items: List[Item] = []
    for img_id, im in img_by_id.items():
        W, H = float(im["width"]), float(im["height"])
        boxes: List[Tuple[int, float, float, float, float]] = []
        for a in anns_by_img.get(img_id, []):
            tgt = cat_id_to_target.get(a["category_id"])
            if tgt is None or W <= 0 or H <= 0:
                continue
            x, y, w, h = a["bbox"]  # COCO: sol-ust x,y + w,h (mutlak px)
            xc, yc = (x + w / 2) / W, (y + h / 2) / H
            boxes.append((tgt, xc, yc, w / W, h / H))
        path = images_dir / im["file_name"]
        if path.exists():
            items.append((path, boxes))
    if cap and len(items) > cap:
        items = random.sample(items, cap)
    return items


def collect_source(src: dict, name2id: Dict[str, int], cap: int = 0) -> List[Item]:
    fmt = src.get("format", "yolo").lower()
    if fmt == "yolo":
        return read_yolo_source(src, name2id, cap)
    if fmt == "coco":
        return read_coco_source(src, name2id, cap)
    raise ValueError(f"Bilinmeyen format: {fmt} (yolo|coco)")


def write_split(items: List[Item], split: str, out: Path, src_name: str) -> None:
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img, boxes in items:
        stem = f"{src_name}_{img.stem}"
        shutil.copy(img, img_dir / (stem + img.suffix.lower()))
        lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in boxes]
        (lbl_dir / (stem + ".txt")).write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="15-sinif birlesik YOLO seti hazirla")
    ap.add_argument("--sources", default="training/sources.yaml")
    ap.add_argument("--data-yaml", default="training/data.yaml")
    ap.add_argument("--out", default="../datasets/rapid_response")
    ap.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-background", action="store_true",
                    help="Hedef kutusu olmayan goruntuleri de tut (negatif ornek).")
    ap.add_argument("--clean", action="store_true", help="Cikis klasorunu once temizle.")
    ap.add_argument("--max-per-source", type=int, default=0,
                    help="Kaynak basina max goruntu (0=sinirsiz). Buyuk Drive setlerini kisar.")
    args = ap.parse_args()

    name2id = load_target_names(args.data_yaml)
    id2name = {v: k for k, v in name2id.items()}
    with open(args.sources, "r", encoding="utf-8") as f:
        sources = yaml.safe_load(f) or []

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    random.seed(args.seed)
    tr, va, te = args.split

    hist = Counter()          # hedef id -> kutu sayisi
    per_split = Counter()     # split -> goruntu sayisi
    per_source = {}

    for src in sources:
        sname = src["name"]
        try:
            items = collect_source(src, name2id, args.max_per_source)
        except Exception as e:
            print(f"[ATLA] {sname}: {type(e).__name__}: {e}")
            per_source[sname] = (0, "atlandi")
            continue
        if not args.keep_background:
            items = [it for it in items if it[1]]  # bos kutuluyu at
        for _, boxes in items:
            for b in boxes:
                hist[b[0]] += 1

        if src.get("val_only"):
            write_split(items, "val", out, sname)
            per_split["val"] += len(items)
            per_source[sname] = (len(items), "val_only")
            continue

        random.shuffle(items)
        n = len(items)
        n_tr = int(n * tr)
        n_va = int(n * va)
        splits = {
            "train": items[:n_tr],
            "val": items[n_tr:n_tr + n_va],
            "test": items[n_tr + n_va:],
        }
        for split, chunk in splits.items():
            write_split(chunk, split, out, sname)
            per_split[split] += len(chunk)
        per_source[sname] = (n, f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")

    # --- Ozet ---
    print("\n=== KAYNAK OZETI (goruntu) ===")
    for s, (n, sp) in per_source.items():
        print(f"  {s:28s} {n:6d}  ({sp})")
    print("\n=== SPLIT (goruntu) ===")
    for s in ("train", "val", "test"):
        print(f"  {s:6s} {per_split[s]}")
    print("\n=== SINIF HISTOGRAMI (kutu) ===")
    for cid in range(len(name2id)):
        nm = id2name[cid]
        c = hist.get(cid, 0)
        flag = "  <-- BOS!" if c == 0 else ""
        print(f"  {cid:2d} {nm:18s} {c:7d}{flag}")
    print(f"\nCikti: {out}  (data.yaml: {args.data_yaml})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
