"""temporal.py — Zamansal mantik: slalom (yorunge) + olay debounce + zaman damgalama.

Iki parca:
  1) SlalomDetector: ana aracin bbox merkez-x'inin zaman penceresi uzerindeki
     yanal salinimindan slalom uretir (tek karede degil, pencerede).
  2) EventAggregator: tum ham olaylari (kategori, etiket, conf, zaman_saniye)
     toplar; ayni etiketin yakin zamanli tekrarlarini TEK olaya indirger
     (gurultu bastirma) ve her olaya kendi confidence_score'unu verir.

Tum olaylar: zaman_saniye = kare_index / fps (cagiran taraf t verir).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from src.schema import KATEGORI_SOFOR_EYLEMI


# =============================================================================
# Slalom — yanal yorunge salinimi (PER-TRACK)
# =============================================================================
class SlalomDetector:
    """Her arac track'i icin AYRI yorunge tutar; takip ID'si yörüngeyi dogru
    aracta tutar (kareler arasi ana arac degisse bile karismaz)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        s = (cfg.get("temporal", {}) or {}).get("slalom", {}) or {}
        self.window_s: float = float(s.get("window_seconds", 3.0))
        self.min_cross: int = int(s.get("min_lane_crossings", 3))
        self.amp_ratio: float = float(s.get("lateral_amplitude_ratio", 0.04))
        self.min_conf: float = float(s.get("min_confidence", 0.5))
        self.max_stale: float = float(
            (cfg.get("tracking", {}) or {}).get("max_stale_seconds", 2.0))
        # track_id -> deque[(t, cx_norm)]
        self._bufs: Dict[int, Deque[Tuple[float, float]]] = {}

    def update(self, track_id: Optional[int],
               box: Optional[Tuple[float, float, float, float]],
               frame_w: int, t_seconds: float) -> Optional[Dict[str, Any]]:
        if box is None or frame_w <= 0:
            return None
        key = track_id if track_id is not None else -1
        cx = 0.5 * (box[0] + box[2]) / float(frame_w)  # 0..1
        buf = self._bufs.setdefault(key, deque())
        buf.append((t_seconds, cx))
        while buf and (t_seconds - buf[0][0]) > self.window_s:
            buf.popleft()

        self._cleanup(t_seconds)

        if len(buf) < 5:
            return None
        xs = np.array([c for _, c in buf], dtype=np.float32)
        amp = float(xs.max() - xs.min())
        crossings = self._zero_crossings(xs - xs.mean())
        if crossings >= self.min_cross and amp >= self.amp_ratio:
            conf = min(1.0, self.min_conf + amp)
            return {
                "kategori": KATEGORI_SOFOR_EYLEMI,
                "etiket": "slalom",
                "confidence_score": float(conf),
                "zaman_saniye": float(t_seconds),
            }
        return None

    def _cleanup(self, t_now: float) -> None:
        """Uzun suredir gorunmeyen track yorungelerini temizle."""
        stale = [k for k, b in self._bufs.items()
                 if not b or (t_now - b[-1][0]) > self.max_stale]
        for k in stale:
            del self._bufs[k]

    @staticmethod
    def _zero_crossings(x: np.ndarray) -> int:
        s = np.sign(x)
        s[s == 0] = 1
        return int(np.sum(s[1:] != s[:-1]))


# =============================================================================
# Olay debounce / birlestirme
# =============================================================================
@dataclass
class _Cluster:
    kategori: str
    etiket: str
    count: int = 0
    best_conf: float = 0.0
    first_t: float = 0.0
    peak_t: float = 0.0
    last_t: float = 0.0


class EventAggregator:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        t = cfg.get("temporal", {}) or {}
        self.gap: float = float(t.get("debounce_gap_seconds", 1.5))
        self.min_frames: int = int(t.get("min_event_frames", 2))
        self.policy: str = str(t.get("timestamp_policy", "peak"))
        self._raw: List[Dict[str, Any]] = []

    def add(self, event: Dict[str, Any]) -> None:
        if event:
            self._raw.append(event)

    def add_many(self, events: List[Dict[str, Any]]) -> None:
        for e in events or []:
            self.add(e)

    def finalize(self) -> List[Dict[str, Any]]:
        """Ham olaylari debounce edip sema-hazir tespit listesi doner."""
        # (kategori, etiket) bazinda zamana gore grupla
        by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for e in self._raw:
            key = (e.get("kategori", ""), e.get("etiket", ""))
            by_key.setdefault(key, []).append(e)

        out: List[Dict[str, Any]] = []
        for (kategori, etiket), evs in by_key.items():
            evs.sort(key=lambda x: x.get("zaman_saniye", 0.0))
            clusters: List[_Cluster] = []
            cur: Optional[_Cluster] = None
            for e in evs:
                t = float(e.get("zaman_saniye", 0.0))
                conf = float(e.get("confidence_score", 0.0))
                if cur is None or (t - cur.last_t) > self.gap:
                    cur = _Cluster(kategori, etiket, 0, 0.0, t, t, t)
                    clusters.append(cur)
                cur.count += 1
                cur.last_t = t
                if conf > cur.best_conf:
                    cur.best_conf = conf
                    cur.peak_t = t

            for c in clusters:
                if c.count < self.min_frames:
                    continue
                t = c.peak_t if self.policy == "peak" else c.first_t
                out.append({
                    "zaman_saniye": float(t),
                    "kategori": c.kategori,
                    "etiket": c.etiket,
                    "confidence_score": float(c.best_conf),
                })

        out.sort(key=lambda x: x["zaman_saniye"])
        return out
