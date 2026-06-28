"""camara_auth.py — CAMARA / Turkcell Open Gateway OAuth2 token saglayici
(image'a GIRMEZ; yalnizca canli final entegrasyonu).

NV ve QoD client'lari artik access_token'i env'den BEKLEMEZ; bu modul tokeni
otomatik alir, onbellekler ve suresi dolmadan yeniler. Iki akis desteklenir:

  1) CIBA (3-legged, Client-Initiated Backchannel Authentication) — SESSIZ,
     SEBEKE-TABANLI. Number Verification ve cihaza-bagli QoD icin DOGRU akis:
        a) POST /bc-authorize  (login_hint=tel:+90..., scope=...) -> auth_req_id
        b) POST /token         (grant_type=...:ciba, auth_req_id) -> access_token
           ("authorization_pending"/"slow_down" pollanir; histerezisli bekleme)
  2) client_credentials (2-legged) — cihaz kimligi gerektirmeyen cagrilar icin.

GIZLI kimlik bilgileri yalnizca ENV'den gelir (config.yaml'a SECRET YAZILMAZ):
    CAMARA_CLIENT_ID, CAMARA_CLIENT_SECRET
Endpoint/scope oncelik sirasi: explicit arg > env > config[auth] > varsayilan.
Sandbox kimlik bilgileri KOMITEDEN saglanir (henuz verilmedi).
"""
from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

LOG = logging.getLogger("rapid_response.camara_auth")

# CIBA token grant'i (CAMARA / OpenID CIBA)
CIBA_GRANT = "urn:openid:params:grant-type:ciba"

_DEFAULT_AUTH_BASE = "https://opengateway.turkcell.com.tr"
_FORM = "application/x-www-form-urlencoded"


class TokenError(RuntimeError):
    """Token alinamadiginda yukseltilir (cagiran tarafindan yakalanir)."""


@dataclass
class _TokenEntry:
    access_token: str
    expires_at: float  # time.monotonic() referansli son kullanma


# =============================================================================
# login_hint yardimcilari
# =============================================================================
def phone_login_hint(phone: str) -> str:
    """Telefon numarasini CIBA login_hint formatina cevirir (tel:+90...)."""
    p = (phone or "").strip()
    if not p:
        return ""
    if ":" in p:  # zaten tel:/ipport: vb. on ekli verilmis
        return p
    return "tel:" + p


def device_login_hint(device: Optional[Dict[str, Any]]) -> str:
    """CAMARA device sozlugunden login_hint uretir (phoneNumber > ipv4)."""
    if not device:
        return ""
    ph = device.get("phoneNumber")
    if ph:
        return phone_login_hint(ph)
    ip = device.get("ipv4Address")
    if isinstance(ip, dict):
        addr = ip.get("publicAddress")
        if addr and addr != "0.0.0.0":
            return "ipport:" + str(addr)
    return ""


def _mask(hint: str) -> str:
    """login_hint'i loglarken maskeler (gizlilik)."""
    return hint if len(hint) <= 6 else hint[:5] + "***" + hint[-2:]


# =============================================================================
# CAMARA OAuth2 token saglayici
# =============================================================================
class CamaraAuthClient:
    def __init__(self, *,
                 config: Optional[Dict[str, Any]] = None,
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 token_url: Optional[str] = None,
                 bc_authorize_url: Optional[str] = None,
                 timeout: float = 10.0) -> None:
        a = ((config or {}).get("auth") or {})
        auth_base = (os.environ.get("CAMARA_AUTH_BASE")
                     or str(a.get("auth_base") or "").strip()
                     or _DEFAULT_AUTH_BASE).rstrip("/")

        # GIZLI degerler: yalnizca explicit arg veya env (config'e yazilmaz)
        self.client_id = client_id or os.environ.get("CAMARA_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("CAMARA_CLIENT_SECRET", "")

        # Endpoint'ler: explicit > env > config > {auth_base}/...
        self.token_url = (token_url or os.environ.get("CAMARA_TOKEN_URL")
                          or str(a.get("token_url") or "").strip()
                          or f"{auth_base}/token")
        self.bc_authorize_url = (bc_authorize_url
                                 or os.environ.get("CAMARA_BC_AUTHORIZE_URL")
                                 or str(a.get("bc_authorize_url") or "").strip()
                                 or f"{auth_base}/bc-authorize")

        self.poll_interval = float(a.get("poll_interval_seconds", 5))
        self.poll_timeout = float(a.get("poll_timeout_seconds", 30))
        self.leeway = float(a.get("token_leeway_seconds", 30))
        self.timeout = timeout
        self._cache: Dict[Tuple[str, ...], _TokenEntry] = {}

    # ------------------------------------------------------------------
    @property
    def configured(self) -> bool:
        """client_id + client_secret mevcut mu? (yoksa token alinamaz)."""
        return bool(self.client_id and self.client_secret)

    def _basic_auth(self) -> str:
        raw = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def _form_headers(self) -> Dict[str, str]:
        return {
            "Authorization": self._basic_auth(),
            "Content-Type": _FORM,
            "Accept": "application/json",
        }

    def _cached(self, key: Tuple[str, ...]) -> Optional[str]:
        e = self._cache.get(key)
        if e and time.monotonic() < (e.expires_at - self.leeway):
            return e.access_token
        return None

    def _store(self, key: Tuple[str, ...], access_token: str,
               expires_in: Any) -> None:
        ttl = float(expires_in) if expires_in else 3600.0
        self._cache[key] = _TokenEntry(access_token, time.monotonic() + ttl)

    def invalidate(self) -> None:
        """Onbellegi temizle (orn. 401 sonrasi yeniden token almak icin)."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # 2-legged: client_credentials
    # ------------------------------------------------------------------
    def get_client_credentials_token(self, scope: str) -> str:
        if not self.configured:
            raise TokenError(
                "CAMARA client_id/secret yok (env: CAMARA_CLIENT_ID / CAMARA_CLIENT_SECRET).")
        key = ("cc", scope)
        tok = self._cached(key)
        if tok:
            return tok
        import requests  # lazy (live bagimliligi)
        data: Dict[str, str] = {"grant_type": "client_credentials"}
        if scope:
            data["scope"] = scope
        r = requests.post(self.token_url, data=data,
                          headers=self._form_headers(), timeout=self.timeout)
        if r.status_code != 200:
            raise TokenError(
                f"client_credentials token hatasi {r.status_code}: {r.text[:200]}")
        j = r.json()
        at = j.get("access_token")
        if not at:
            raise TokenError("access_token yok (client_credentials yaniti).")
        self._store(key, at, j.get("expires_in"))
        LOG.info("client_credentials token alindi (scope=%s).", scope)
        return at

    # ------------------------------------------------------------------
    # 3-legged: CIBA (backchannel) — sessiz, sebeke-tabanli
    # ------------------------------------------------------------------
    def get_ciba_token(self, scope: str, login_hint: str) -> str:
        if not self.configured:
            raise TokenError(
                "CAMARA client_id/secret yok (env: CAMARA_CLIENT_ID / CAMARA_CLIENT_SECRET).")
        if not login_hint:
            raise TokenError("CIBA icin login_hint gerekli (telefon/cihaz kimligi).")
        key = ("ciba", scope, login_hint)
        tok = self._cached(key)
        if tok:
            return tok
        auth_req_id, interval = self._bc_authorize(scope, login_hint)
        at, expires_in = self._poll_token(auth_req_id, interval)
        self._store(key, at, expires_in)
        LOG.info("CIBA token alindi (scope=%s, hint=%s).", scope, _mask(login_hint))
        return at

    def _bc_authorize(self, scope: str, login_hint: str) -> Tuple[str, float]:
        import requests
        data: Dict[str, str] = {"login_hint": login_hint}
        if scope:
            data["scope"] = scope
        r = requests.post(self.bc_authorize_url, data=data,
                          headers=self._form_headers(), timeout=self.timeout)
        if r.status_code not in (200, 201):
            raise TokenError(f"bc-authorize hatasi {r.status_code}: {r.text[:200]}")
        j = r.json()
        auth_req_id = j.get("auth_req_id")
        if not auth_req_id:
            raise TokenError("auth_req_id yok (bc-authorize yaniti).")
        interval = float(j.get("interval", self.poll_interval) or self.poll_interval)
        return auth_req_id, interval

    def _poll_token(self, auth_req_id: str, interval: float) -> Tuple[str, Any]:
        import requests
        deadline = time.monotonic() + self.poll_timeout
        wait = max(0.0, interval)
        while True:
            data = {"grant_type": CIBA_GRANT, "auth_req_id": auth_req_id}
            r = requests.post(self.token_url, data=data,
                              headers=self._form_headers(), timeout=self.timeout)
            if r.status_code == 200:
                j = r.json()
                at = j.get("access_token")
                if not at:
                    raise TokenError("access_token yok (ciba token yaniti).")
                return at, j.get("expires_in")

            # 400 + error kodu: bekle / hizlan / hata
            try:
                err = str(r.json().get("error", ""))
            except Exception:
                err = ""
            if err == "authorization_pending":
                pass
            elif err == "slow_down":
                wait += 5.0
            else:
                raise TokenError(
                    f"ciba token hatasi {r.status_code}/{err or '-'}: {r.text[:200]}")

            if time.monotonic() + wait > deadline:
                raise TokenError(
                    f"CIBA token zaman asimi ({self.poll_timeout:.0f}s; son durum="
                    f"{err or r.status_code}).")
            time.sleep(wait)
