# Rapid Response — Mobil Uygulama (React Native 0.73 / TypeScript)

OTR'de taahhüt edilen framework: **React Native 0.73 / TypeScript**. Bu uygulama,
güvenlik görevlisine yönelik sade arayüzde canlı video + bounding-box overlay,
risk profili, QoD durumu ve araç bilgisini gösterir. Backend ile köprü
`live/server.py` (FastAPI + WebSocket) üzerinden kurulur.

## Mimari

```
Telefon (React Native)  --REST-->  POST /auth/number-verification   (Number Verification)
                        --WS---->   /ws/stream?source=&phone=        (canli kare + tespitler)
                                         |
                                    live/server.py  →  StreamPipeline (YOLO+BoTSORT, hiz, risk, QoD)
```

- **Number Verification:** giriş ekranı telefon numarasını backend'e iletir; gerçek
  doğrulama Turkcell NV API'si ile backend'de yapılır (SMS/OTP yok — Şartname §4.1).
- **Canlı akış:** WebSocket'ten her kare JSON olarak gelir
  (`arac_bilgisi`, `detections`, `tespitler`, `risk`, `qod`, `image_jpeg_b64`).
- **QoD:** ihtiyaç (need) yüksekken backend QoD'u aktive eder; uygulama `qod.active`
  durumunu görsel olarak gösterir (Şartname %40 — yalnızca ihtiyaç varken).

## Kurulum

```bash
cd mobile
npm install        # veya: yarn

# iOS (yalnizca macOS):
cd ios && pod install && cd ..
npm run ios

# Android:
npm run android
```

> İlk çalıştırmadan önce `src/config.ts` içindeki `backendHost` değerini,
> `live/server.py`'nin çalıştığı bilgisayarın **telefonla aynı ağ / 5G**
> üzerindeki IP adresine göre düzenleyin.

## Backend'i başlatma

```bash
# rapid_response/ kökünde
pip install fastapi "uvicorn[standard]"
uvicorn live.server:app --host 0.0.0.0 --port 8000
# Gelistirmede NV'yi atlamak icin (yalnizca test): NV_DEV_BYPASS=1
```

## Dosya yapısı

```
mobile/
  App.tsx                      # Navigasyon (Login -> Live)
  index.js                     # AppRegistry giris noktasi
  src/
    config.ts                  # backendHost/port, WS URL uretici
    models/frameResult.ts      # WS kare semasi + guvenli parse
    services/
      authService.ts           # POST /auth/number-verification
      streamService.ts         # WebSocket /ws/stream sarmalayici
    screens/
      LoginScreen.tsx          # Number Verification girisi
      LiveScreen.tsx           # Canli video + tespit duzeni
    components/
      DetectionOverlay.tsx     # Bounding box overlay (react-native-svg)
      EventsList.tsx           # Surucu eylem/tespit listesi
      RiskBadge.tsx            # Risk profili rozeti
      QodIndicator.tsx         # QoD aktif/pasif gostergesi
      VehicleCard.tsx          # Arac bilgisi (tip/plaka/renk/hiz)
```

> Not: Bu modül FTR offline imajına **girmez** (`.dockerignore` ile hariç),
> yalnızca canlı final demosu içindir.
