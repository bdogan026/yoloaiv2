import {AppConfig} from '../config';

/**
 * Number Verification (sessiz, sebeke-tabanli) dogrulamasi.
 * Gercek dogrulama backend uzerinden Turkcell NV API ile yapilir; bu istemci
 * yalnizca backend'e telefon numarasini iletir ve sonucu alir.
 * Sartname §4.1: SMS/OTP yok; dogrulama mobil sebeke uzerinden arka planda.
 */
export async function verifyNumber(phoneNumber: string): Promise<boolean> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const resp = await fetch(`${AppConfig.httpBase}/auth/number-verification`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({phoneNumber}),
      signal: controller.signal,
    });
    if (resp.status === 200) {
      const data = await resp.json();
      return data?.verified === true;
    }
    return false;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}
