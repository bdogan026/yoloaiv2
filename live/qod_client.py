"""qod_client.py — CAMARA / Turkcell Open Gateway Quality on Demand (QoD)
istemcisi + tetikleme denetleyicisi (image'a GIRMEZ).

Iki parca:
  1) QoDClient      : CAMARA QoD API (/sessions) ile gecici yuksek kalite talebi
                      (create) ve serbest birakma (delete). PCF->SMF->UPF zinciri
                      saglayicidir; biz yalnizca QoD oturumunu yonetiriz.
  2) QoDController  : "yalnizca ihtiyac varken" QoD tetikleme mantigi. Risk
                      buyuklugune DEGIL, BILGI IHTIYACINA (need) baglidir.

CELISKI DUZELTMESI (Gelistirme notu):
  Eski kural: "<50 ve >85 kapat" -> risk buyuklugunu QoD ihtiyaciyla karistiriyor
  ve 50-85 arasinda suni bir pencere yaratiyordu.
  Yeni tutarli kural (histerezisli):
      AC  : need >= open_threshold,  open_frames kare ust uste
      KAPAT: need <  close_threshold, close_frames kare ust uste (cooldown)
  Burada need = severity_potential * (1 - confidence) (bkz. risk_model).
  Dogal sonuc:
    - Dusuk risk + emin  -> need~0 -> KAPALI (eski "risk<50 kapat")
    - Yuksek risk + emin -> need~0 -> KAPALI (eski "risk>85 kapat"; karar verildi)
    - Belirsiz (orta)    -> need yuksek -> ACIK (en belirsiz -> en yuksek destek)
  Boylece eski iki kapatma kosulu, TEK ilkenin dogal sonucu olur; suni
  sureksizlik ve celiski ortadan kalkar.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from live.camara_auth import CamaraAuthClient, TokenError, device_login_hint

LOG = logging.getLogger("rapid_response.qod")


# =============================================================================
# QoD API istemcisi (CAMARA Quality-on-Demand)
# =============================================================================
class QoDClient:
    def __init__(self,
                 base_url: Optional[str] = None,
                 access_token: Optional[str] = None,
                 qos_profile: str = "QOS_E",
                 *,
                 auth: Optional[CamaraAuthClient] = None,
                 scope: Optional[str] = None,
                 timeout: float = 10.0) -> None:
        # Endpoint env'den; GIZLI kimlik bilgileri CamaraAuthClient ile env'den.
        self.base_url = base_url or os.environ.get(
            "QOD_BASE_URL", "https://opengateway.turkcell.com.tr/qod/v0")
        # Statik token (varsa) auth akisini atlar — test/gelistirme icin.
        self.static_token = access_token or os.environ.get("QOD_ACCESS_TOKEN", "")
        self.auth = auth if auth is not None else CamaraAuthClient()
        self.scope = scope or os.environ.get("QOD_SCOPE", "qod")
        self.qos_profile = qos_profile
        self.timeout = timeout
        self._session_id: Optional[str] = None
        self._device: Optional[Dict[str, Any]] = None

    def _bearer(self, device: Optional[Dict[str, Any]]) -> str:
        """Gecerli access_token (statik > CIBA(cihaz) > client_credentials)."""
        if self.static_token:
            return self.static_token
        if self.auth and self.auth.configured:
            hint = device_login_hint(device)
            if hint:
                return self.auth.get_ciba_token(self.scope, hint)
            return self.auth.get_client_credentials_token(self.scope)
        return ""

    def _headers(self, token: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def create_session(self, device: Dict[str, Any],
                       app_server: Dict[str, Any],
                       duration_s: int = 60) -> Optional[str]:
        """QoD oturumu ac (gecici yuksek kalite). sessionId doner."""
        import requests  # lazy (live bagimliligi)
        try:
            token = self._bearer(device)
        except TokenError as exc:
            LOG.error("QoD token alinamadi: %s", exc)
            return None
        if not token:
            LOG.error("QoD access_token yok (CAMARA_CLIENT_ID/SECRET veya "
                      "QOD_ACCESS_TOKEN ayarlayin).")
            return None
        payload = {
            "qosProfile": self.qos_profile,
            "device": device,            # orn. {"phoneNumber": "+90..."} / ipv4Address
            "applicationServer": app_server,  # MEC/AS adresi
            "duration": duration_s,
        }
        try:
            r = requests.post(f"{self.base_url}/sessions",
                              json=payload, headers=self._headers(token),
                              timeout=self.timeout)
            r.raise_for_status()
            self._session_id = r.json().get("sessionId")
            self._device = device
            LOG.info("QoD oturumu acildi: %s (profil=%s, %ss)",
                     self._session_id, self.qos_profile, duration_s)
            return self._session_id
        except Exception as exc:
            LOG.error("QoD oturumu acilamadi: %s", exc)
            return None

    def delete_session(self) -> bool:
        """Aktif QoD oturumunu serbest birak (durum bitince)."""
        if not self._session_id:
            return True
        import requests
        try:
            token = self._bearer(self._device)
        except TokenError as exc:
            LOG.error("QoD kapatma token hatasi: %s", exc)
            return False
        if not token:
            LOG.error("QoD oturumu kapatilamadi: access_token yok.")
            return False
        try:
            r = requests.delete(f"{self.base_url}/sessions/{self._session_id}",
                                headers=self._headers(token), timeout=self.timeout)
            r.raise_for_status()
            LOG.info("QoD oturumu kapatildi: %s", self._session_id)
            self._session_id = None
            self._device = None
            return True
        except Exception as exc:
            LOG.error("QoD oturumu kapatilamadi: %s", exc)
            return False

    @property
    def active(self) -> bool:
        return self._session_id is not None


# =============================================================================
# QoD tetikleme denetleyicisi (histerezisli, ihtiyac-tabanli)
# =============================================================================
@dataclass
class QoDController:
    open_threshold: float = 0.45      # need bu esigi asarsa ACMA adayi
    close_threshold: float = 0.20     # need bu esigin altinda KAPATMA adayi
    open_frames: int = 3              # AC: ust uste kare
    close_frames: int = 8             # KAPAT: ust uste kare (cooldown)
    max_session_seconds: float = 120  # emniyet: maksimum acik kalma

    _active: bool = field(default=False, init=False)
    _above: int = field(default=0, init=False)
    _below: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)

    def update(self, need: float) -> bool:
        """need [0..1] -> QoD aktif olmali mi? (histerezis + emniyet zaman asimi)"""
        if need >= self.open_threshold:
            self._above += 1
            self._below = 0
        elif need < self.close_threshold:
            self._below += 1
            self._above = 0
        else:
            # belirsiz bant: sayaclari tasima (kararliligi koru)
            self._above = max(0, self._above - 1)
            self._below = max(0, self._below - 1)

        if not self._active and self._above >= self.open_frames:
            self._active = True
            self._opened_at = time.time()
            LOG.info("QoD TETIKLENDI (need=%.2f) - bilgi ihtiyaci yuksek.", need)
        elif self._active:
            too_long = (time.time() - self._opened_at) >= self.max_session_seconds
            if self._below >= self.close_frames or too_long:
                self._active = False
                LOG.info("QoD SERBEST (need=%.2f, sure_asimi=%s) - durum cozuldu.",
                         need, too_long)
        return self._active

    @property
    def active(self) -> bool:
        return self._active
