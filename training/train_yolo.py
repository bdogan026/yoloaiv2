"""train_yolo.py — YOLO11 fine-tune (transfer learning) — image'a GIRMEZ.

Ultralytics ile pretrained agirliktan transfer learning yapar; en iyi agirligi
weights/best_model.pt olarak kaydeder. Tum parametreler config.yaml'dan okunur
(hardcode yok). Degerlendirme: mAP50, mAP50-95, precision, recall, confusion
matrix (Ultralytics val ciktilari).

Kullanim:
    python training/train_yolo.py --config config/config.yaml
    python training/train_yolo.py --no-synthetic   # ablasyon: sentetik veri yok

NOT: Bu betik gelistirme makinesinde calistirilir (internet ile pretrained
indirilebilir). FTR image'inda yer ALMAZ.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import yaml


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Rapid Response YOLO11 fine-tune")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--no-synthetic", action="store_true",
                    help="Ablasyon: sentetik (augment.py) verisi olmadan egit.")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    t = cfg.get("training", {}) or {}

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics kurulu degil: pip install ultralytics", file=sys.stderr)
        return 1

    pretrained = t.get("pretrained", "yolo11s.pt")
    data_yaml = t.get("data_yaml", "training/data.yaml")
    print(f"[train] pretrained={pretrained} data={data_yaml} "
          f"synthetic={'OFF' if args.no_synthetic else 'ON'}")

    model = YOLO(pretrained)

    results = model.train(
        data=data_yaml,
        epochs=int(t.get("epochs", 100)),
        imgsz=int(t.get("imgsz", 640)),
        batch=int(t.get("batch", 16)),
        lr0=float(t.get("lr0", 0.01)),
        lrf=float(t.get("lrf", 0.01)),
        patience=int(t.get("patience", 25)),
        optimizer=t.get("optimizer", "auto"),
        device=t.get("device", 0),
        project=t.get("project", "runs"),
        name=t.get("name", "rapid_response_yolo11s"),
        # Ultralytics dahili augmentasyon
        hsv_h=float(t.get("hsv_h", 0.015)),
        hsv_s=float(t.get("hsv_s", 0.7)),
        hsv_v=float(t.get("hsv_v", 0.4)),
        degrees=float(t.get("degrees", 5.0)),
        translate=float(t.get("translate", 0.1)),
        scale=float(t.get("scale", 0.5)),
        fliplr=float(t.get("fliplr", 0.5)),
        mosaic=float(t.get("mosaic", 1.0)),
        mixup=float(t.get("mixup", 0.1)),
        verbose=True,
    )

    # Dogrulama metrikleri (mAP50, mAP50-95, P, R, confusion matrix dosyalari)
    metrics = model.val()
    try:
        print("\n=== Dogrulama Metrikleri ===")
        print(f"mAP50     : {metrics.box.map50:.4f}")
        print(f"mAP50-95  : {metrics.box.map:.4f}")
        print(f"precision : {metrics.box.mp:.4f}")
        print(f"recall    : {metrics.box.mr:.4f}")
        print("(confusion matrix + PR egrileri runs/.../ altinda kaydedildi)")
    except Exception as exc:
        print(f"Metrik ozeti alinamadi: {exc}")

    # En iyi agirligi cikarim yoluna kopyala
    best = getattr(getattr(model, "trainer", None), "best", None)
    export_to = t.get("export_best_to", "weights/best_model.pt")
    if best and os.path.exists(best):
        os.makedirs(os.path.dirname(export_to), exist_ok=True)
        shutil.copy(best, export_to)
        print(f"[train] En iyi agirlik kopyalandi: {best} -> {export_to}")
    else:
        print("[train] Uyari: best.pt bulunamadi; export atlandi.")

    # Opsiyonel hizlandirma (gomulu, offline)
    fmt = (t.get("export_format") or "").strip()
    if fmt:
        try:
            print(f"[train] Export: {fmt} (int8={t.get('export_int8', False)})")
            model.export(format=fmt, int8=bool(t.get("export_int8", False)),
                         imgsz=int(t.get("imgsz", 640)))
        except Exception as exc:
            print(f"[train] Export hatasi: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
