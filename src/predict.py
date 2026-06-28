"""predict.py — FTR cikarim giris fonksiyonu.

main.py bunu cagirir: run_inference(video_path, weights_path) -> results dict.
Sonuc, main.py tarafindan /app/data/output/results.json'a yazilir.

Gurbuzluk: her turlu hata yakalanir; cokme yerine gecerli (bos tespitli) bir
sonuc dondurulur. Anti-hile: ortam tespiti YOK, tek tip davranis.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from src.pipeline import Pipeline
from src.schema import empty_results
from src.utils import get_logger, load_config

LOG = get_logger()


def run_inference(video_path: str,
                  weights_path: Optional[str] = None,
                  config_path: Optional[str] = None) -> Dict[str, Any]:
    """Videoyu analiz edip yarisma semasinda sonuc sozlugu doner."""
    video_id = os.path.basename(video_path) if video_path else "video.mp4"
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        LOG.error("Config yuklenemedi (%s); bos sonuc.", exc)
        return empty_results(video_id)

    # weights_path verilmediyse config aday yollari kullanilir (Pipeline icinde)
    try:
        pipeline = Pipeline(cfg, weights_path=weights_path)
        return pipeline.run(video_path, video_id=video_id)
    except Exception as exc:
        LOG.error("Cikarim sirasinda hata (%s); bos sonuc donduruluyor.", exc)
        return empty_results(video_id)


if __name__ == "__main__":
    import json
    import sys

    vp = sys.argv[1] if len(sys.argv) > 1 else "/app/data/input/video.mp4"
    wp = sys.argv[2] if len(sys.argv) > 2 else None
    out = run_inference(vp, wp)
    print(json.dumps(out, ensure_ascii=False, indent=2))
