"""main.py — FTR giris noktasi.

docker run ile otomatik baslar (Dockerfile CMD). Akis:
  /app/data/input/video.mp4  ->  run_inference  ->  /app/data/output/results.json

KRITIK kurallar:
  * Cikti JSON anahtarlari/etiketleri BIREBIR (schema.py garanti eder).
  * json.dump(..., ensure_ascii=False, indent=2).
  * Cokme yok: girdi yoksa/bozuksa bile gecerli (bos tespitli) results.json uretilir.
  * Anti-hile: ortam degiskeni/hostname/IP/dosya tabanli "degerlendirme ortami mi?"
    tespiti YOK; tek tip davranis.
"""
import json
import os
import sys

# src/ paketine erisim
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.predict import run_inference  # noqa: E402
from src.schema import empty_results  # noqa: E402
from src.utils import get_logger, load_config  # noqa: E402

LOG = get_logger()

# Yarisma sabit yollari (config okunamazsa yedek olarak kullanilir)
DEFAULT_INPUT = "/app/data/input/video.mp4"
DEFAULT_OUTPUT = "/app/data/output/results.json"


def _paths():
    """Yollari config'den oku; basarisizsa yarisma varsayilanlarina dus."""
    try:
        cfg = load_config()
        p = cfg.get("paths", {}) or {}
        return (p.get("input_video", DEFAULT_INPUT),
                p.get("output_json", DEFAULT_OUTPUT))
    except Exception as exc:
        LOG.warning("Config okunamadi (%s); varsayilan yollar kullanilacak.", exc)
        return DEFAULT_INPUT, DEFAULT_OUTPUT


def _write_json(output_path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> int:
    input_path, output_path = _paths()
    video_id = os.path.basename(input_path) or "video.mp4"

    LOG.info("Yol Guvenligi Yapay Zeka Cikarim Islemi Baslatildi...")
    LOG.info("Girdi: %s | Cikti: %s", input_path, output_path)

    # Girdi yoksa: cokme yerine gecerli bos results.json uret (kural geregi).
    if not os.path.exists(input_path):
        LOG.error("Girdi videosu bulunamadi -> %s. Bos results.json yaziliyor.",
                  input_path)
        try:
            _write_json(output_path, empty_results(video_id))
        except Exception as exc:
            LOG.error("Bos cikti yazilamadi: %s", exc)
        return 0

    try:
        output_data = run_inference(input_path)
        _write_json(output_path, output_data)
        LOG.info("Islem basariyla tamamlandi. Cikti kaydedildi: %s", output_path)
        return 0
    except Exception as exc:
        # Son savunma hatti: yine de gecerli bos cikti birak, cokme yok.
        LOG.error("Model calistirilirken hata: %s. Bos results.json yaziliyor.", exc)
        try:
            _write_json(output_path, empty_results(video_id))
        except Exception as exc2:
            LOG.error("Bos cikti da yazilamadi: %s", exc2)
        return 0


if __name__ == "__main__":
    sys.exit(main())
