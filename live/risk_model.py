"""risk_model.py — Turk Trafik Kanunu'na dayali NESNEL risk skoru + QoD ihtiyac
koprusu (image'a GIRMEZ).

Iki AYRI eksen (Gelistirme notu: "risk(0-100) <-> guven(0-1) koprusunu acikca
tanimlayin"):
  * R  (risk severity) ∈ [0,100] : olayin SONUC agirligi (KTK madde agirligi x
    yakinlik faktoru). "Ne kadar tehlikeli."
  * c  (confidence)    ∈ [0,1]   : modelin o tespitten ne kadar EMIN oldugu.
                                    "Ne kadar eminiz."

Bu ikisi DIK (orthogonal). QoD karari risk BUYUKLUGUNE degil, BILGI IHTIYACINA
baglidir:
      need_i = severity_potential_i * (1 - c_i)
      need   = max_i(need_i)
Yani "potansiyel agir ama belirsiz" olaylar en yuksek QoD ihtiyacini uretir
(en belirsiz -> en yuksek destek). Bu, eski "risk<50 VE risk>85 kapat" celiskili
kuralinin yerine gecen tek tutarli ilkedir (bkz. qod_client.QoDController).

Profil siniflari (sliding-window, ~3 sn): 0-24 Guvenli, 25-49 Dikkatli,
50-74 Riskli, 75-100 Tehlikeli.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

# KTK/TCK madde agirliklari (OTR ile tutarli; gerektiginde config'e tasinabilir).
# NOT: madde numaralari OTR'deki iddialarla ayni tutulmustur.
LAW_WEIGHTS: Dict[str, Dict[str, Any]] = {
    "esneme":                {"weight": 35, "ref": "TCK m.47 (yorgunluk)"},
    "telefonla_konusma":     {"weight": 40, "ref": "TCK m.65"},
    "emniyet_kemeri_ihlali": {"weight": 30, "ref": "TCK m.79"},
    "sigara_icme":           {"weight": 15, "ref": "gorus/dikkat engeli"},
    "su_icme":               {"weight": 10, "ref": "dikkat dagilmasi"},
    "slalom":                {"weight": 35, "ref": "tehlikeli manevra"},
    "arkaya_bakma":          {"weight": 20, "ref": "dikkat dagilmasi"},
    "etrafa_bakinma":        {"weight": 15, "ref": "dikkat dagilmasi"},
}
_MAX_WEIGHT = max(v["weight"] for v in LAW_WEIGHTS.values())  # normalizasyon icin


@dataclass
class RiskState:
    risk: float          # 0..100
    profile: str         # Guvenli | Dikkatli | Riskli | Tehlikeli
    need: float          # 0..1  (QoD bilgi ihtiyaci)
    top_event: str       # en cok katki yapan etiket


class RiskModel:
    """Tek arac icin sliding-window risk + QoD ihtiyac sinyali."""

    def __init__(self, window_frames: int = 90) -> None:
        self.window = window_frames
        self._scores: Deque[float] = deque(maxlen=window_frames)

    @staticmethod
    def proximity_factor(box: Tuple[float, float, float, float],
                         frame_wh: Tuple[int, int]) -> float:
        """bbox/frame alan oranindan yakinlik carpani [1.0 .. ~2.0]."""
        w, h = frame_wh
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        ratio = area / float(max(1, w * h))
        return 1.0 + min(1.0, ratio * 6.0)  # yakin arac -> daha yuksek risk

    def update(self, events: List[Dict[str, Any]],
               proximity: float = 1.0) -> RiskState:
        """Bu karenin olaylarindan anlik risk + QoD ihtiyacini gunceller.

        events: [{etiket, confidence_score, ...}, ...] (sofor_eylemi olaylari)
        """
        frame_score = 0.0
        need = 0.0
        top_event = ""
        for e in events:
            etiket = e.get("etiket", "")
            info = LAW_WEIGHTS.get(etiket)
            if not info:
                continue
            c = float(e.get("confidence_score", 0.0))
            w = float(info["weight"])
            # risk katkisi: agirlik x guven x yakinlik
            contrib = w * c * proximity
            frame_score += contrib
            # QoD ihtiyaci: potansiyel agirlik (normalize) x belirsizlik
            sev_pot = w / _MAX_WEIGHT
            need_i = sev_pot * (1.0 - c)
            if need_i > need:
                need = need_i
                top_event = etiket

        self._scores.append(min(100.0, frame_score))
        risk = sum(self._scores) / max(1, len(self._scores))
        risk = float(min(100.0, risk))
        return RiskState(risk=risk, profile=self._profile(risk),
                         need=float(need), top_event=top_event)

    def add_plate_need(self, base: RiskState, plate_unreadable: bool,
                       vehicle_present: bool) -> RiskState:
        """Plaka okunamiyorsa (ilgili araç varken) QoD ihtiyacina sabit katki."""
        if plate_unreadable and vehicle_present:
            base.need = max(base.need, 0.6)  # plaka okuma ihtiyaci -> QoD adayi
        return base

    @staticmethod
    def _profile(risk: float) -> str:
        if risk < 25:
            return "Guvenli"
        if risk < 50:
            return "Dikkatli"
        if risk < 75:
            return "Riskli"
        return "Tehlikeli"
