"""server.py — Mobil uygulama <-> cikarim motoru koprusu (FastAPI + WebSocket).
image'a GIRMEZ (canli final; FTR submission'a dahil degil).

Mimari (Sartname §3/§4 ile uyumlu):
  React Native Mobil App  <--WebSocket-->  server.py  -->  StreamPipeline (YOLO+...)
                          <--REST-->       /auth/number-verification (NV)

Akis:
  * POST /auth/number-verification : sessiz, sebeke-tabanli NV (Turkcell API
    client'i uzerinden). Telefon/araç sistemi dogrulamasi (ilk adim).
  * WS  /ws/stream                 : kamera akisini StreamPipeline ile isler;
    her kare icin {tespitler, arac_bilgisi(+hiz), risk, qod, jpeg} yayinlar.
  * GET /qod/status                : guncel QoD durumu (panel).

Cikarim (cv2 dongusu) BLOKLAYICI oldugundan ayri THREAD'de calisir; sonuclar
thread-safe kuyruga konur, async WS kuyruktan okuyup gonderir. WS koptugunda
should_stop ile dongu durdurulur.

Calistirma (gelistirme makinesi):
  pip install -r live/requirements-live.txt
  uvicorn live.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import base64
import os
import queue
import threading
from typing import Any, Dict, Optional

import cv2

from live.camara_auth import CamaraAuthClient
from live.number_verification import NumberVerificationClient
from src.utils import get_logger, load_config

LOG = get_logger("rapid_response.server")

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("FastAPI gerekli: pip install -r live/requirements-live.txt "
                     f"({exc})")

app = FastAPI(title="Rapid Response Live API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_CFG = load_config()
_JPEG_QUALITY = 70
# Paylasimli CAMARA token saglayici (token onbellegi istekler arasinda korunur)
_AUTH = CamaraAuthClient(config=_CFG)
# Gelistirme: NV kimlik bilgileri (sandbox) henuz yoksa girisi acmak icin
_NV_DEV_BYPASS = os.environ.get("NV_DEV_BYPASS", "0") == "1"


class NVRequest(BaseModel):
    phoneNumber: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/number-verification")
def number_verification(req: NVRequest) -> Dict[str, Any]:
    """Sessiz, sebeke-tabanli dogrulama (Turkcell NV API client'i)."""
    if _NV_DEV_BYPASS:
        LOG.warning("NV_DEV_BYPASS aktif: dogrulama atlandi (yalnizca gelistirme).")
        return {"verified": True, "dev_bypass": True}
    client = NumberVerificationClient(
        auth=_AUTH, scope=(_CFG.get("auth") or {}).get("nv_scope"))
    verified = client.verify(req.phoneNumber)
    return {"verified": verified}


@app.get("/qod/status")
def qod_status() -> Dict[str, Any]:
    return {"info": "QoD durumu WS akisindaki her kare mesajinda 'qod' alaninda."}


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    """Kamera akisini isleyip her kareyi mobil uygulamaya yayinlar.

    Sorgu parametreleri:
      source : video yolu / rtsp url / kamera index (varsayilan 0)
      phone  : (ops.) NV icin telefon numarasi
    """
    await ws.accept()
    params = ws.query_params
    source = params.get("source", "0")
    source = int(source) if source.isdigit() else source
    phone = params.get("phone")

    out_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=4)
    stop_event = threading.Event()

    def on_frame(result: Dict[str, Any], frame) -> None:
        # Kareyi JPEG+base64 olarak mesaja ekle (kutular orijinal cozunurlukte)
        ok, enc = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if ok:
            result = {**result,
                      "image_jpeg_b64": base64.b64encode(enc.tobytes()).decode("ascii")}
        # Kuyruk doluysa en eskiyi at (gecikme birikmesin)
        try:
            out_q.put_nowait(result)
        except queue.Full:
            try:
                out_q.get_nowait()
            except queue.Empty:
                pass
            try:
                out_q.put_nowait(result)
            except queue.Full:
                pass

    def worker() -> None:
        # StreamPipeline agir importlari (torch) burada, baglanti aninda yuklenir
        from live.stream_pipeline import StreamPipeline
        pipe = StreamPipeline(load_config(), phone_number=phone)
        pipe.run(source, on_frame=on_frame, should_stop=stop_event.is_set)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    loop = asyncio.get_event_loop()
    try:
        while not stop_event.is_set():
            try:
                msg = await loop.run_in_executor(None, out_q.get, True, 1.0)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue
            await ws.send_json(msg)
    except WebSocketDisconnect:
        LOG.info("WS istemci ayrildi.")
    except Exception as exc:
        LOG.error("WS akis hatasi: %s", exc)
    finally:
        stop_event.set()
        thread.join(timeout=3.0)
        LOG.info("WS akis kapatildi.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
