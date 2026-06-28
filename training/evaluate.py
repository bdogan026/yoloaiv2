"""evaluate.py — Resmi metrik harness'i — image'a GIRMEZ.

Gelistirme notunun #1 kirmizi maddesi ("gercek olculmus metrikler", 20 puan)
ve QoD acik/kapali Δ tablosu (yarismanin %40'i) icin uctan uca olcum.

Modlar:
  detection : Ultralytics val -> mAP50, mAP50-95, precision, recall, FPS
              (kare bazli dedektor metrikleri; data.yaml test/val seti gerekir).
  events    : Tahmin results.json'lari ile GROUND-TRUTH results.json'lari
              karsilastirir -> olay bazli P/R/F1 (zaman toleransli) + plaka
              karakter dogrulugu (CER) + tip/renk dogrulugu.
  qod       : Ayni test setini "QoD kapali" (hafif yol) ve "QoD acik" (agir yol)
              profilleriyle calistirip GT'ye gore Δ tablosu uretir.

GROUND-TRUTH nasil hazirlanir (gercek veriyi nasil olceriz):
  * Dedektor metrikleri icin: test goruntulerini YOLO formatinda etiketle
    (data.yaml test split). Ultralytics mAP/P/R'yi otomatik hesaplar.
  * Olay/uctan-uca metrikler icin: birkac test videosunu ELLE etiketle; her video
    icin bir results.json (zaman_saniye + kategori + etiket + plaka/tip/renk)
    olustur (--gt-dir). Bu, "altin standart"tir; tahminle karsilastirilir.

Kullanim:
  python training/evaluate.py --mode detection --weights weights/best_model.pt \
      --data training/data.yaml --split test
  python training/evaluate.py --mode events --gt-dir gt/ --videos-dir test_videos/
  python training/evaluate.py --mode events --gt-dir gt/ --pred-dir preds/
  python training/evaluate.py --mode qod --gt-dir gt/ --videos-dir test_videos/
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config  # noqa: E402


# ============================================================================
# Yardimcilar
# ============================================================================
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_by_video_id(folder: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in glob.glob(os.path.join(folder, "*.json")):
        try:
            data = load_json(p)
            vid = data.get("video_id") or os.path.splitext(os.path.basename(p))[0]
            out[vid] = data
        except Exception as exc:
            print(f"UYARI: {p} okunamadi: {exc}")
    return out


# ============================================================================
# Olay bazli metrikler (zaman toleransli eslesme)
# ============================================================================
def match_events(gt: List[Dict[str, Any]], pred: List[Dict[str, Any]],
                 tol: float) -> Dict[Tuple[str, str], Dict[str, int]]:
    res: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0})
    gt_by: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    pr_by: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for e in gt:
        gt_by[(e.get("kategori"), e.get("etiket"))].append(float(e.get("zaman_saniye", 0)))
    for e in pred:
        pr_by[(e.get("kategori"), e.get("etiket"))].append(float(e.get("zaman_saniye", 0)))
    for k in set(gt_by) | set(pr_by):
        g = sorted(gt_by[k])
        p = sorted(pr_by[k])
        used = [False] * len(g)
        for pt in p:
            best, bd = -1, tol + 1e-9
            for i, gt_t in enumerate(g):
                if used[i]:
                    continue
                d = abs(gt_t - pt)
                if d <= tol and d < bd:
                    bd, best = d, i
            if best >= 0:
                used[best] = True
                res[k]["tp"] += 1
            else:
                res[k]["fp"] += 1
        res[k]["fn"] += used.count(False)
    return res


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate_events(gt_map: Dict[str, Dict[str, Any]],
                    pred_map: Dict[str, Dict[str, Any]],
                    tol: float) -> Dict[str, Any]:
    agg: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0})
    plate_exact, plate_total, plate_char_acc = 0, 0, []
    tip_ok = tip_total = renk_ok = renk_total = 0

    for vid, gt in gt_map.items():
        pred = pred_map.get(vid, {})
        m = match_events(gt.get("tespitler", []), pred.get("tespitler", []), tol)
        for k, v in m.items():
            for key in ("tp", "fp", "fn"):
                agg[k][key] += v[key]
        # plaka / tip / renk
        g_ai = gt.get("arac_bilgisi", {}) or {}
        p_ai = pred.get("arac_bilgisi", {}) or {}
        g_plate = (g_ai.get("plaka") or "").upper()
        if g_plate:
            plate_total += 1
            p_plate = (p_ai.get("plaka") or "").upper()
            if p_plate == g_plate:
                plate_exact += 1
            cer = levenshtein(p_plate, g_plate) / max(1, len(g_plate))
            plate_char_acc.append(max(0.0, 1.0 - cer))
        if g_ai.get("tip"):
            tip_total += 1
            tip_ok += int(p_ai.get("tip") == g_ai.get("tip"))
        if g_ai.get("renk"):
            renk_total += 1
            renk_ok += int(p_ai.get("renk") == g_ai.get("renk"))

    per_label = {}
    micro = {"tp": 0, "fp": 0, "fn": 0}
    f_list = []
    for k, v in sorted(agg.items()):
        p, r, f = prf(v["tp"], v["fp"], v["fn"])
        per_label[f"{k[0]}/{k[1]}"] = {"P": round(p, 4), "R": round(r, 4),
                                       "F1": round(f, 4), **v}
        for key in micro:
            micro[key] += v[key]
        f_list.append(f)
    mp, mr, mf = prf(micro["tp"], micro["fp"], micro["fn"])
    return {
        "per_label": per_label,
        "micro": {"P": round(mp, 4), "R": round(mr, 4), "F1": round(mf, 4)},
        "macro_F1": round(sum(f_list) / len(f_list), 4) if f_list else 0.0,
        "plate": {
            "exact_match_rate": round(plate_exact / plate_total, 4) if plate_total else None,
            "char_accuracy": round(sum(plate_char_acc) / len(plate_char_acc), 4) if plate_char_acc else None,
            "n": plate_total,
        },
        "tip_accuracy": round(tip_ok / tip_total, 4) if tip_total else None,
        "renk_accuracy": round(renk_ok / renk_total, 4) if renk_total else None,
    }


# ============================================================================
# Pipeline calistirma (events/qod modlari icin)
# ============================================================================
def run_pipeline_dir(videos_dir: str, cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    from src.pipeline import Pipeline  # torch/cv2 gerekir
    preds: Dict[str, Dict[str, Any]] = {}
    vids = [f for f in os.listdir(videos_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    for v in vids:
        path = os.path.join(videos_dir, v)
        # Her video icin TAZE Pipeline (durum kirlenmesini onlemek icin)
        pipe = Pipeline(copy.deepcopy(cfg))
        preds[v] = pipe.run(path, video_id=v)
        print(f"[eval] islendi: {v} ({len(preds[v].get('tespitler', []))} tespit)")
    return preds


# ============================================================================
# Modlar
# ============================================================================
def mode_detection(args) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics gerekli.", file=sys.stderr)
        return 1
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz,
                        device=args.device, verbose=False)
    speed = getattr(metrics, "speed", {}) or {}
    total_ms = sum(float(v) for v in speed.values()) if speed else 0.0
    fps = (1000.0 / total_ms) if total_ms > 0 else None
    print("\n=== DEDEKTOR METRIKLERI ===")
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")
    print(f"precision : {metrics.box.mp:.4f}")
    print(f"recall    : {metrics.box.mr:.4f}")
    if fps:
        print(f"FPS       : {fps:.1f}  (kare basi {total_ms:.1f} ms)")
    print("(confusion matrix + PR egrileri runs/.../ altinda)")
    return 0


def mode_events(args) -> int:
    cfg = load_config()
    tol = float((cfg.get("evaluate", {}) or {}).get("event_time_tolerance_seconds", 1.0))
    gt_map = index_by_video_id(args.gt_dir)
    if args.pred_dir:
        pred_map = index_by_video_id(args.pred_dir)
    elif args.videos_dir:
        pred_map = run_pipeline_dir(args.videos_dir, cfg)
    else:
        print("events: --pred-dir veya --videos-dir gerekli.", file=sys.stderr)
        return 1
    report = evaluate_events(gt_map, pred_map, tol)
    print("\n=== OLAY BAZLI METRIKLER (tol=%.1fs) ===" % tol)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


def mode_qod(args) -> int:
    if not args.videos_dir:
        print("qod: --videos-dir gerekli.", file=sys.stderr)
        return 1
    cfg = load_config()
    ev = cfg.get("evaluate", {}) or {}
    tol = float(ev.get("event_time_tolerance_seconds", 1.0))
    gt_map = index_by_video_id(args.gt_dir)

    cfg_off = deep_merge(cfg, ev.get("qod_off_profile", {}))
    cfg_on = deep_merge(cfg, ev.get("qod_on_profile", {}))
    print("[qod] QoD KAPALI (hafif yol) calisiyor...")
    rep_off = evaluate_events(gt_map, run_pipeline_dir(args.videos_dir, cfg_off), tol)
    print("[qod] QoD ACIK (agir yol) calisiyor...")
    rep_on = evaluate_events(gt_map, run_pipeline_dir(args.videos_dir, cfg_on), tol)

    def delta(a, b):
        if a is None or b is None:
            return None
        return round(b - a, 4)

    table = {
        "qod_off": rep_off,
        "qod_on": rep_on,
        "delta_on_minus_off": {
            "micro_F1": delta(rep_off["micro"]["F1"], rep_on["micro"]["F1"]),
            "macro_F1": delta(rep_off["macro_F1"], rep_on["macro_F1"]),
            "plate_char_accuracy": delta(rep_off["plate"]["char_accuracy"],
                                         rep_on["plate"]["char_accuracy"]),
            "plate_exact_match_rate": delta(rep_off["plate"]["exact_match_rate"],
                                            rep_on["plate"]["exact_match_rate"]),
            "tip_accuracy": delta(rep_off["tip_accuracy"], rep_on["tip_accuracy"]),
            "renk_accuracy": delta(rep_off["renk_accuracy"], rep_on["renk_accuracy"]),
        },
    }
    print("\n=== QoD ACIK vs KAPALI Δ (kanit tablosu) ===")
    print(json.dumps(table, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(table, f, ensure_ascii=False, indent=2)
    return 0


def mode_speed(args) -> int:
    """GERCEK HIZ tahmini hata metrigi (MAE/RMSE km/h). Sartname §5 (%40 YZ)."""
    if not args.gt_speeds or not args.videos_dir:
        print("speed: --gt-speeds ve --videos-dir gerekli.", file=sys.stderr)
        return 1
    from live.speed import estimate_main_vehicle_speed
    from src.detection.detector import YOLODetector
    cfg = load_config()
    with open(args.gt_speeds, "r", encoding="utf-8") as f:
        gt = json.load(f)
    abs_errs, rows = [], []
    for vid, gt_kmh in gt.items():
        path = os.path.join(args.videos_dir, vid)
        if not os.path.exists(path):
            continue
        det = YOLODetector(cfg, args.weights)  # taze (BoTSORT durumu karismasin)
        est = estimate_main_vehicle_speed(path, cfg, detector=det)
        if est is None:
            print(f"  {vid}: tahmin yok")
            continue
        err = abs(est - float(gt_kmh))
        abs_errs.append(err)
        rows.append({"video": vid, "gt_kmh": float(gt_kmh),
                     "pred_kmh": round(est, 2), "abs_err": round(err, 2)})
    report = {"rows": rows}
    if abs_errs:
        report["MAE_kmh"] = round(sum(abs_errs) / len(abs_errs), 3)
        report["RMSE_kmh"] = round((sum(e*e for e in abs_errs) / len(abs_errs)) ** 0.5, 3)
        report["reference_MAE_kmh"] = 0.58  # Gelistirme notu (Gajdoš IPM)
    print("\n=== HIZ HATA METRIGI (km/h) ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Rapid Response metrik harness'i")
    ap.add_argument("--mode", required=True,
                    choices=["detection", "events", "qod", "speed"])
    ap.add_argument("--weights", default="weights/best_model.pt")
    ap.add_argument("--data", default="training/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0)
    ap.add_argument("--gt-dir")
    ap.add_argument("--pred-dir")
    ap.add_argument("--videos-dir")
    ap.add_argument("--gt-speeds")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.mode == "detection":
        return mode_detection(args)
    if args.mode == "events":
        if not args.gt_dir:
            print("events: --gt-dir gerekli.", file=sys.stderr)
            return 1
        return mode_events(args)
    if args.mode == "qod":
        if not args.gt_dir:
            print("qod: --gt-dir gerekli.", file=sys.stderr)
            return 1
        return mode_qod(args)
    if args.mode == "speed":
        return mode_speed(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
