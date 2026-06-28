# Proje Durumu — Rapid Response (TEKNOFEST 2026 5G & YZ Akıllı Yol Güvenliği)

> **Bu dosya nedir?** Takıma hızlı oryantasyon / devir notu. Mimari detay için
> [`README.md`](README.md)'ye bak. Bağlayıcı çıktı kuralları için
> [`README.md` §0](README.md) + bu dosyadaki "Önce oku" bölümü.
>
> Son güncelleme: 2026-06-24

---

## 0. ⚠️ ÖNCE OKU (yanlış anlamayı önler)

1. **Belge öncelik sırası (çelişince):** `FTR > Şartname > Geliştirme > OTR`.
   - FTR = otomatik puanlanan offline çıktı kuralları → **ihlali teslimi geçersiz kılar**.
   - Bir şey çelişiyorsa üstteki kazanır.

2. **Mobil framework = React Native 0.73 / TypeScript** (OTR'de resmen taahhüt edilen).
   - Bir ara Flutter ile yazılmıştı, **silindi**; `mobile/` artık React Native.
   - **Flutter'a geri dönme.** (Geliştirme.pdf'teki tek satırlık "Flutter" notu bağlayıcı değil.)

3. **Number Verification (NV) ve Quality on Demand (QoD) API'lerini biz YAZMIYORUZ.**
   - Bunları **Turkcell** Open Gateway platformunda sağlıyor; biz yalnızca **client** olarak çağırıyoruz.
   - Şebeke tarafı (PCF→SMF→UPF, MEC) Turkcell'in; sandbox erişimi/kimlik bilgileri komiteden gelecek.

4. **İki ayrı çalışma ortamı var, karıştırma:**
   - **FTR (offline):** Docker imajı, internet KAPALI, `video.mp4 → results.json`. Puanın bir kısmı.
   - **Canlı final:** `live/` backend + `mobile/` uygulama + gerçek 5G API'leri. Puanın diğer kısmı.
   - **Hız tahmini canlı/rapor tarafında**; FTR `results.json` şemasına hız **EKLENMEZ** (şema sabit).

---

## 1. ✅ Yapılanlar (Done)

### FTR offline çıkarım (puanın temeli)
- `main.py` → `/app/data/input/video.mp4` → `run_inference` → `/app/data/output/results.json`.
- `src/`: ön işleme, YOLO11 tespit, plaka OCR, araç analizi, sürücü eylemleri, şema.
- `src/schema.py`: FTR'nin **birebir** JSON anahtarları + ASCII/küçük-harf etiket kuralları.
- `Dockerfile`: T4 / CUDA 12.1 base, <8GB, seçici COPY (training/live/mobile imaja **girmez**),
  `/app/models → /app/weights` symlink (ağırlık yolu çelişkisi çözülü).
- Çökme yok: girdi bozuk/yok olsa bile geçerli (boş tespitli) `results.json` üretilir.

### Ölçüm / değerlendirme
- `training/evaluate.py`: tespit/plaka metrikleri + **hız MAE/RMSE modu** (`--mode speed`).

### Hız tahmini (Şartname %40 YZ — gerçek hız)
- `live/speed.py`: IPM/homografi ile araç yer-temas noktasından km/h (regression/endpoints/median).
- `training/calibrate_speed.py`: 3 mod — `extract` (videodan referans kare + şerit adayları),
  `homography` (ölçülen metre noktalarından H), `validate` (GT ile MAE, Gajdoš ~0.58 km/h referansı).
- Sentetik test: 54 km/h → 54.0 ✓ (üç modda), hareketsiz→0, homografi yok→güvenli (çökme yok).

### Canlı final mimarisi (`live/`)
- `live/server.py`: FastAPI + WebSocket köprüsü (telefon ↔ çıkarım motoru). Her kare için
  `{arac_bilgisi(+hiz), detections, tespitler, risk, qod, jpeg}` yayınlar.
- `live/stream_pipeline.py`: canlı pipeline (QoD-gated ön işleme, ihtiyaç-tabanlı QoD, hız, risk).
- `live/risk_model.py`, `live/qod_client.py` (QoDController: histerezisli, ihtiyaç-tabanlı tetikleme).

### Mobil uygulama (`mobile/` — React Native 0.73 / TypeScript)
- NV ile sessiz giriş ekranı → WS canlı akış ekranı (video + bounding-box overlay + risk/QoD/araç kartı).
- Yapı: `App.tsx`, `src/{config,models,services,screens,components}`. Detay: [`mobile/README.md`](mobile/README.md).

### 5G token akışı (CAMARA OAuth2 / CIBA) — **bu oturumda eklendi**
- `live/camara_auth.py`: token otomatik alınır/önbelleklenir/yenilenir.
  - **CIBA** (3-legged, sessiz/şebeke-tabanlı): NV ve cihaz-bağlı QoD.
  - **client_credentials** (2-legged): cihaz kimliği yoksa.
- NV & QoD client'ları artık token'ı env'den beklemiyor; otomatik alıyor (statik token bypass'ı korunur).
- Mock'lu testlerle doğrulandı (6/6 geçti). Tüm `.py` derleniyor (38 dosya).

---

## 2. 🔧 Düzenlenecekler / dikkat (çalıştırmadan önce)

| Ne | Nerede | Neden |
|---|---|---|
| **Backend IP** | `mobile/src/config.ts` → `backendHost` | Telefonun, `server.py`'nin koştuğu makineyle aynı ağdaki IP'sine ayarla |
| **CAMARA kimlik bilgileri** | env: `CAMARA_CLIENT_ID`, `CAMARA_CLIENT_SECRET` | **Komiteden gelecek** (sandbox). config.yaml'a SECRET YAZILMAZ |
| **NV/QoD scope + endpoint** | `config/config.yaml` → `auth:` | Komite kesin scope/endpoint verince güncelle (şimdilik makul varsayılan) |
| **Hız homografisi** | `training/calibrate_speed.py` + config `speed.homography` | Gerçek **metre** ölçüleri gerekli; pikselden uydurulmaz |
| **best_model.pt** | `weights/` (şu an YOK) | FTR imajı bu ağırlık olmadan boş çıktı verir — bkz. Yapılacaklar |

---

## 3. 📋 Yapılacaklar (To do) — öncelik sırasına göre

1. **[BLOCKER] YOLO11 eğit + `weights/best_model.pt` üret ve yerleştir.**
   `training/train_yolo.py` (+ `data.yaml`). Bu olmadan FTR imajı gerçek tespit yapmaz.
2. **[BLOCKER] Offline varlıkları indir:** `training/fetch_assets.py`
   (face_landmarker.task + EasyOCR) → `weights/` altına; imaja gömülür (runtime internet KAPALI).
3. **Docker build + T4 testi:** imaj <8GB ve video başına ≤10 dk sınırını ölç.
4. **Gerçek 5G API testi:** sandbox kimlik bilgileri gelince NV girişi + QoD oturumu uçtan uca dene
   (şu an mock'lu test geçiyor; gerçek ağ denenmedi).
5. **Mobil derleme:** `cd mobile && npm install && npm run tsc`, sonra cihazda `npm run ios/android`.
6. **Rapor (Geliştirme.pdf kırmızı maddeleri):** ölçülmüş bileşen metrikleri (mAP/P/R/F1/FPS),
   **QoD açık vs kapalı ölçülmüş Δ** (yarışmanın büyük kısmı), veri seti metodolojisi, ablasyon.
7. **Hız kalibrasyonu:** sahne işaretleriyle homografi çıkar + GT hızla doğrula (MAE raporla).

---

## 4. ▶️ Çalıştırma

### FTR (offline / Docker)
```bash
docker build -t rapid-response .
docker run --rm --gpus all \
  -v $(pwd)/data/input:/app/data/input \
  -v $(pwd)/data/output:/app/data/output \
  rapid-response
# Çıktı: data/output/results.json
```

### Canlı backend
```bash
pip install -r live/requirements-live.txt
# Geliştirmede NV'yi atlamak için (sandbox creds yokken giriş açılır):
NV_DEV_BYPASS=1 uvicorn live.server:app --host 0.0.0.0 --port 8000
# Gerçek akış için: CAMARA_CLIENT_ID=... CAMARA_CLIENT_SECRET=... uvicorn ...
```

### Mobil (React Native)
```bash
cd mobile
npm install
npm run ios      # veya: npm run android
# Önce mobile/src/config.ts -> backendHost'u backend IP'sine ayarla
```

---

## 5. 🗂️ Dosya haritası (özet)

```
rapid_response/
  main.py                  # FTR giriş noktası (video.mp4 -> results.json)
  Dockerfile, .dockerignore
  requirements.txt
  config/config.yaml       # tüm ayarlar (detection, qod, speed, auth, training...)
  src/                     # FTR offline çıkarım çekirdeği (imaja girer)
  training/                # eğitim + kalibrasyon + evaluate (imaja GİRMEZ)
    train_yolo.py, evaluate.py, calibrate_speed.py, fetch_assets.py ...
  live/                    # canlı final (imaja GİRMEZ)
    server.py              # FastAPI + WebSocket köprüsü
    stream_pipeline.py     # canlı pipeline
    camara_auth.py         # CAMARA OAuth2 / CIBA token sağlayıcı  ← YENİ
    number_verification.py # NV client
    qod_client.py          # QoD client + QoDController
    speed.py               # IPM hız tahmini
    risk_model.py
  mobile/                  # React Native 0.73 / TS uygulaması (imaja GİRMEZ)
  weights/                 # best_model.pt + offline varlıklar (git'e KONMAZ; eksik!)
```

---

## 6. Belge kaynakları (bu repoda YOK — ayrı paylaşılır)

> Yarışma PDF'leri repo köküne **konmadı** (telif + boyut). Takım ortak alanından edinin.

- Şartname: `2026_..._SARTNAMESI_TR.1_Lb6.pdf`
- FTR teslim dokümanı: `5G_..._FTR_Aşaması_Teslim_D.pdf`
- OTR: `OTR_RapidResponse_v4_updated.pdf`
- Geliştirme notları: `Geliştirme.pdf` (iç beyin fırtınası — bağlayıcı değil)
