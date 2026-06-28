"""fetch_assets.py — Offline model varliklarini weights/ altina indirir.

Bu betik GELISTIRME makinesinde (internet ACIK) BIR KEZ calistirilir; indirdigi
modeller image'a gomulur ve calisma aninda (internet KAPALI) yalnizca lokal
yuklenir. (best_model.pt'yi train_yolo.py uretir; bu betik diger gomulu
varliklari indirir.)

Indirilenler:
  * weights/face_landmarker.task   (MediaPipe Face Landmarker)
  * weights/ocr/                   (EasyOCR en modelleri: detection + recognition)

Kullanim:
    python training/fetch_assets.py --weights-dir weights
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task")
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task")


def fetch_face(weights_dir: str) -> None:
    dst = os.path.join(weights_dir, "face_landmarker.task")
    if os.path.exists(dst):
        print(f"[assets] zaten var: {dst}")
        return
    print(f"[assets] indiriliyor: {FACE_URL}")
    urllib.request.urlretrieve(FACE_URL, dst)
    print(f"[assets] kaydedildi: {dst}")


def fetch_hand(weights_dir: str) -> None:
    dst = os.path.join(weights_dir, "hand_landmarker.task")
    if os.path.exists(dst):
        print(f"[assets] zaten var: {dst}")
        return
    print(f"[assets] indiriliyor: {HAND_URL}")
    urllib.request.urlretrieve(HAND_URL, dst)
    print(f"[assets] kaydedildi: {dst}")


def fetch_fast_plate(weights_dir: str, model: str = "cct-xs-v1-global-model") -> None:
    """fast-plate-ocr ONNX modelini build aninda indirir; ~/.cache'teki tum cache'i
    weights/fast_plate_ocr/ altina kopyalar (calisma aninda offline geri-yuklenir)."""
    out_dir = os.path.join(weights_dir, "fast_plate_ocr")
    try:
        import shutil
        from fast_plate_ocr import LicensePlateRecognizer  # tetikle -> ~/.cache'e iner
        LicensePlateRecognizer(model)
        cache = os.path.expanduser("~/.cache/fast-plate-ocr")
        if os.path.isdir(cache):
            shutil.copytree(cache, out_dir, dirs_exist_ok=True)
            print(f"[assets] fast-plate-ocr cache -> {out_dir}")
        else:
            print(f"[assets] fast-plate-ocr cache yok: {cache}", file=sys.stderr)
    except Exception as exc:
        print(f"[assets] fast-plate-ocr indirilemedi: {exc}", file=sys.stderr)


def fetch_ocr(weights_dir: str, languages) -> None:
    ocr_dir = os.path.join(weights_dir, "ocr")
    os.makedirs(ocr_dir, exist_ok=True)
    try:
        import easyocr
    except ImportError:
        print("easyocr kurulu degil; OCR modelleri indirilemedi.", file=sys.stderr)
        return
    # download_enabled=True ile bir kez indir; modeller ocr_dir'e yazilir.
    print(f"[assets] EasyOCR modelleri indiriliyor -> {ocr_dir}")
    easyocr.Reader(list(languages), model_storage_directory=ocr_dir,
                   download_enabled=True, verbose=False)
    print(f"[assets] EasyOCR modelleri hazir: {ocr_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline model varliklarini indir")
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--languages", nargs="+", default=["en"])
    ap.add_argument("--skip-face", action="store_true")
    ap.add_argument("--skip-ocr", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.weights_dir, exist_ok=True)
    if not args.skip_face:
        try:
            fetch_face(args.weights_dir)
            fetch_hand(args.weights_dir)
        except Exception as exc:
            print(f"[assets] face/hand_landmarker indirilemedi: {exc}", file=sys.stderr)
        fetch_fast_plate(args.weights_dir)
    if not args.skip_ocr:
        fetch_ocr(args.weights_dir, args.languages)
    print("\n[assets] Tamamlandi. best_model.pt icin: python training/train_yolo.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
