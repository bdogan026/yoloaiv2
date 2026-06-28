"""schema.py — Izinli etiket/kategori sabitleri + cikti dogrulama.

Bu modul, otomatik puanlama scripti ile BIREBIR eslesmeyi garanti eder.
FTR Teslim Dokumantasyonu (Tablo 1/2 ve JSON ornekleri) tek otorite kaynaktir.

KRITIK kurallar (ihlali teslimi gecersiz kilar):
  * JSON anahtarlari birebir: video_id, arac_bilgisi, tespitler, tip, plaka,
    renk, confidence_score, zaman_saniye, kategori, etiket.
  * Kategori degerleri: "sofor_eylemi", "nesneler", "yolcular"
    (Tablo'daki "Surucu eylemi" DEGIL -> JSON ornegindeki "sofor_eylemi").
  * Tum etiketler ASCII + kucuk harf (Turkce karaktersiz).

Bu modul disariya yalnizca sabitleri ve dogrulama fonksiyonlarini acar; hicbir
agir bagimlilik (torch/cv2) import etmez, bu sayede tek basina test edilebilir.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Tuple

# --- Kategori degerleri (JSON ornegindeki birebir degerler) ------------------
KATEGORI_SOFOR_EYLEMI: str = "sofor_eylemi"
KATEGORI_NESNELER: str = "nesneler"
KATEGORI_YOLCULAR: str = "yolcular"

# --- Tablo 2: kategori -> gecerli etiket kumeleri ----------------------------
SOFOR_EYLEMI_ETIKETLERI: Tuple[str, ...] = (
    "arkaya_bakma",
    "esneme",
    "sigara_icme",
    "su_icme",
    "telefonla_konusma",
    "slalom",
    "etrafa_bakinma",
    "emniyet_kemeri_ihlali",
)
NESNELER_ETIKETLERI: Tuple[str, ...] = (
    "teknocan",
    "bilgisayar",
)
YOLCULAR_ETIKETLERI: Tuple[str, ...] = (
    "arka_koltuk_1",
    "arka_koltuk_2",
    "on_koltuk",
)

ALLOWED_LABELS: Dict[str, Tuple[str, ...]] = {
    KATEGORI_SOFOR_EYLEMI: SOFOR_EYLEMI_ETIKETLERI,
    KATEGORI_NESNELER: NESNELER_ETIKETLERI,
    KATEGORI_YOLCULAR: YOLCULAR_ETIKETLERI,
}

# --- Tablo 1: arac bilgisi izinli degerleri ----------------------------------
ARAC_TIPLERI: Tuple[str, ...] = (
    "sedan",
    "suv",
    "hatchback",
    "pickup",
    "minibus",
    "panelvan",
    "kamyon",
)
ARAC_RENKLERI: Tuple[str, ...] = (
    "beyaz",
    "siyah",
    "gri",
    "kirmizi",
    "mavi",
    "sari",
    "yesil",
    "turuncu",
    "kahverengi",
)

# Turkce -> ASCII donusum tablosu (etiket/plaka normalize icin)
_TR_ASCII = str.maketrans({
    "ç": "c", "Ç": "c",
    "ğ": "g", "Ğ": "g",
    "ı": "i", "İ": "i",
    "ö": "o", "Ö": "o",
    "ş": "s", "Ş": "s",
    "ü": "u", "Ü": "u",
})


def to_ascii_lower(text: str) -> str:
    """Turkce karakterleri ASCII'ye cevirip kucuk harfe dusurur."""
    if text is None:
        return ""
    text = str(text).translate(_TR_ASCII)
    # kalan aksanlari da temizle (ornn. é -> e)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.strip().lower()


def is_valid_detection(kategori: str, etiket: str) -> bool:
    """Bir (kategori, etiket) ciftinin izinli olup olmadigini doner."""
    return kategori in ALLOWED_LABELS and etiket in ALLOWED_LABELS[kategori]


def normalize_plate(raw: str, regex: str) -> str:
    """Plaka metnini normalize edip resmi regex ile dogrular.

    Adimlar: ASCII'ye cevir -> bosluk/ozel karakter temizle -> buyuk harf ->
    regex kontrolu. Regex disindaki sonuc "tespit edilemedi" sayilir -> "".
    """
    if not raw:
        return ""
    cleaned = to_ascii_lower(raw).upper()
    # Sadece harf ve rakam birak (OCR'in urettigi - . boslugu vb. at)
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
    if not cleaned:
        return ""
    # Resmi regex harf araliklari kucuk/buyuk; emniyet icin IGNORECASE ile dogrula
    if re.match(regex, cleaned, flags=re.IGNORECASE):
        return cleaned
    return ""


def coerce_label(kategori: str, etiket: str) -> str:
    """Etiketi ASCII/kucuk harfe normalize eder (kategori bazli degil; saf metin)."""
    return to_ascii_lower(etiket)


def validate_results(
    results: Dict[str, Any],
    *,
    drop_invalid: bool = True,
    round_confidence: int = 4,
    round_time: int = 2,
) -> Dict[str, Any]:
    """Sonuc sozlugunu sema kurallarina gore dogrular ve temizler.

    - tespitler icindeki her olay (kategori, etiket) izinli mi kontrol edilir;
      izinli degilse normalize edilip tekrar denenir, yine olmazsa
      drop_invalid'e gore atilir.
    - arac_bilgisi.tip/renk izinli kumeye zorlanir; plaka regex'i schema
      disindaysa cagiran tarafindan zaten "" gelir.
    - confidence_score [0,1] araligina kelepcelenir ve yuvarlanir.
    Anahtar isimleri bu fonksiyonda DEGISTIRILMEZ; girdi zaten dogru
    anahtarlarla gelmelidir (formatter sorumlulugu).
    """
    out: Dict[str, Any] = {}
    out["video_id"] = str(results.get("video_id", "video.mp4"))

    # --- arac_bilgisi ---
    arac = results.get("arac_bilgisi")
    if isinstance(arac, dict):
        tip = to_ascii_lower(arac.get("tip", ""))
        if tip not in ARAC_TIPLERI:
            tip = ""  # gecersiz tip -> bos (puanlama "tespit edilemedi")
        renk = to_ascii_lower(arac.get("renk", ""))
        if renk not in ARAC_RENKLERI:
            renk = ""
        # Plaka zaten upstream'de regex ile dogrulanip normalize edilir; burada
        # son bir savunma: ASCII'ye cevir, harf/rakam disini at, buyuk harf.
        plaka = re.sub(r"[^A-Z0-9]", "", to_ascii_lower(arac.get("plaka", "")).upper())
        conf = _clamp01(arac.get("confidence_score", 0.0))
        out["arac_bilgisi"] = {
            "tip": tip,
            "plaka": plaka,
            "renk": renk,
            "confidence_score": round(conf, round_confidence),
        }

    # --- tespitler ---
    tespitler: List[Dict[str, Any]] = []
    for det in results.get("tespitler", []) or []:
        if not isinstance(det, dict):
            continue
        kategori = str(det.get("kategori", ""))
        etiket = str(det.get("etiket", ""))
        if not is_valid_detection(kategori, etiket):
            # normalize edip tekrar dene
            etiket_n = coerce_label(kategori, etiket)
            if is_valid_detection(kategori, etiket_n):
                etiket = etiket_n
            elif drop_invalid:
                continue
            else:
                etiket = etiket_n
        tespitler.append({
            "zaman_saniye": round(float(det.get("zaman_saniye", 0.0)), round_time),
            "kategori": kategori,
            "etiket": etiket,
            "confidence_score": round(_clamp01(det.get("confidence_score", 0.0)),
                                      round_confidence),
        })
    out["tespitler"] = tespitler
    return out


def empty_results(video_id: str = "video.mp4") -> Dict[str, Any]:
    """Cokme durumunda bile gecerli (bos tespitli) bir sonuc iskeleti."""
    return {
        "video_id": video_id,
        "arac_bilgisi": {
            "tip": "",
            "plaka": "",
            "renk": "",
            "confidence_score": 0.0,
        },
        "tespitler": [],
    }


def _clamp01(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


if __name__ == "__main__":
    # Tek basina hizli dogrulama testi
    import json

    sample = {
        "video_id": "video.mp4",
        "arac_bilgisi": {"tip": "Sedan", "plaka": "34 ABC 123", "renk": "Kırmızı",
                          "confidence_score": 1.4},
        "tespitler": [
            {"zaman_saniye": 14.5, "kategori": "sofor_eylemi",
             "etiket": "telefonla_konusma", "confidence_score": 0.89},
            {"zaman_saniye": 1.0, "kategori": "sofor_eylemi",
             "etiket": "GECERSIZ", "confidence_score": 0.5},
        ],
    }
    print(json.dumps(validate_results(sample), ensure_ascii=False, indent=2))
    print("plaka:", normalize_plate("34 abc 123",
          r'^(0[1-9]|[1-7][0-9]|8[01])((\s?[a-zA-Z]\s?)(\d{4,5})|(\s?[a-zA-Z]{2}\s?)(\d{3,4})|(\s?[a-zA-Z]{3}\s?)(\d{2,3}))$'))
