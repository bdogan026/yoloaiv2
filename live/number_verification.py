"""number_verification.py — CAMARA / Turkcell Open Gateway Number Verification
istemcisi (image'a GIRMEZ).

Ilk adimda kullanici/arac sistemini SMS'SIZ, SESSIZ, SEBEKE-TABANLI dogrular.
Mobil veri uzerinden operator, cihazin telefon numarasini sebeke seviyesinde
teyit eder (CAMARA Number Verification). SMS OTP yok -> dusuk surtunme + yuksek
guvenlik.

Akis:
  1) CAMARA auth (CIBA, 3-legged) ile access_token OTOMATIK alinir
     (scope: number-verification:verify) -> bkz. camara_auth.CamaraAuthClient.
  2) POST /verify {phoneNumber} -> {devicePhoneNumberVerified: true/false}
  veya /device-phone-number ile maskelenmis numara dogrulamasi.

Token kaynagi onceligi:
  - NV_ACCESS_TOKEN (statik, env)  ->  varsa dogrudan kullanilir (test/bypass)
  - aksi halde CamaraAuthClient (CAMARA_CLIENT_ID/SECRET ile CIBA akisi)
Sandbox kimlik bilgileri komiteden saglanir.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from live.camara_auth import CamaraAuthClient, TokenError, phone_login_hint

LOG = logging.getLogger("rapid_response.nv")


class NumberVerificationClient:
    def __init__(self,
                 base_url: Optional[str] = None,
                 access_token: Optional[str] = None,
                 *,
                 auth: Optional[CamaraAuthClient] = None,
                 scope: Optional[str] = None,
                 timeout: float = 10.0) -> None:
        self.base_url = base_url or os.environ.get(
            "NV_BASE_URL",
            "https://opengateway.turkcell.com.tr/number-verification/v0")
        # Statik token (varsa) auth akisini atlar — test/gelistirme icin.
        self.static_token = access_token or os.environ.get("NV_ACCESS_TOKEN", "")
        self.auth = auth if auth is not None else CamaraAuthClient()
        self.scope = scope or os.environ.get("NV_SCOPE", "number-verification:verify")
        self.timeout = timeout

    def _bearer(self, login_hint: str) -> str:
        """Gecerli access_token'i dondurur (statik > CIBA). Yoksa "" doner."""
        if self.static_token:
            return self.static_token
        if self.auth and self.auth.configured:
            return self.auth.get_ciba_token(self.scope, login_hint)
        return ""

    def _headers(self, token: str) -> dict:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def verify(self, phone_number: str) -> bool:
        """Sebeke tabanli sessiz dogrulama. True/False doner."""
        import requests  # lazy
        try:
            token = self._bearer(phone_login_hint(phone_number))
        except TokenError as exc:
            LOG.error("NV token alinamadi: %s", exc)
            return False
        if not token:
            LOG.error("NV access_token yok (CAMARA_CLIENT_ID/SECRET veya "
                      "NV_ACCESS_TOKEN ayarlayin).")
            return False
        try:
            r = requests.post(f"{self.base_url}/verify",
                              json={"phoneNumber": phone_number},
                              headers=self._headers(token), timeout=self.timeout)
            r.raise_for_status()
            verified = bool(r.json().get("devicePhoneNumberVerified", False))
            LOG.info("Number Verification: %s -> %s", phone_number, verified)
            return verified
        except Exception as exc:
            LOG.error("Number Verification hatasi: %s", exc)
            return False

    def device_phone_number(self, login_hint: Optional[str] = None) -> Optional[str]:
        """Cihazin sebekedeki (maskelenmis) numarasini doner (opsiyonel).

        Numara onceden bilinmediginden token; verilen login_hint ile CIBA,
        yoksa client_credentials uzerinden alinir.
        """
        import requests
        try:
            if self.static_token:
                token = self.static_token
            elif self.auth and self.auth.configured:
                token = (self.auth.get_ciba_token(self.scope, login_hint)
                         if login_hint
                         else self.auth.get_client_credentials_token(self.scope))
            else:
                token = ""
            if not token:
                LOG.error("device-phone-number: access_token yok.")
                return None
            r = requests.get(f"{self.base_url}/device-phone-number",
                             headers=self._headers(token), timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("devicePhoneNumber")
        except Exception as exc:
            LOG.error("device-phone-number hatasi: %s", exc)
            return None
