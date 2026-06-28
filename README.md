# Rapid Response — 5G & Yapay Zekâ ile Akıllı Yol Güvenliği

Uçtan uca, **modüler** ve **yeniden eğitilebilir** (fine-tune) yol güvenliği
yapay zekâ sistemi. İki teslim hedefi **ortak bir çekirdeği** (`src/`) paylaşır:

1. **FTR teslimi:** İzole Docker konteynerinde **offline** video çıkarımı
   (`/app/data/input/video.mp4` → `/app/data/output/results.json`),
   NVIDIA Tesla T4, çalışma anında internet **kapalı**, video başına **maks. 10 dk**.
2. **Canlı final:** Mobil uygulama + 5G API'leri (Number Verification + Quality
   on Demand). "Kritik durum" tespitinde QoD ile ağ kalitesi yükseltilip daha
   yüksek başarımlı çıkarım yolu (DCP + SAHI + OCR) devreye alınır.

---

## 0. Tasarım ilkeleri (FTR kuralları ile birebir)

- **JSON anahtar/değerleri birebir:** `video_id`, `arac_bilgisi`, `tespitler`,
  `tip`, `plaka`, `renk`, `confidence_score`, `zaman_saniye`, `kategori`,
  `etiket`. Kategori değerleri: `sofor_eylemi`, `nesneler`, `yolcular`.
  Tüm etiketler **ASCII + küçük harf**. Bunlar `src/schema.py` ile yazımdan
  **önce doğrulanır** (otomatik puanlama scriptiyle birebir eşleşme garantisi).
- **İnternet kapalı:** Tüm ağırlıklar (YOLO `.pt`, MediaPipe `.task`, EasyOCR)
  image'a gömülür ve **lokal yoldan** yüklenir. Auto-download kapalıdır
  (`runtime.offline_env` + Dockerfile `ENV` + EasyOCR `download_enabled=False`).
- **Anti-hile:** Ortam değişkeni / hostname / IP / dosya varlığıyla "değerlendirme
  ortamında mıyım?" tespiti yapıp davranış değiştiren kod **yoktur**. Tek tip
  davranılır. (Model dosyasının yerini bulmak için yapılan yol kontrolü ortam
  tespiti değildir; her koşulda aynıdır.)
- **Çökme yok:** Girdi yoksa/bozuksa try/except ile yakalanır ve yine de geçerli
  (boş tespitli) `results.json` üretilir; süreç çökmez.
- **Hiçbir parametre hardcode değildir:** her şey `config/config.yaml`'dan okunur.

---

## 1. Proje yapısı

```
rapid_response/
├── Dockerfile              # FTR image (root level)
├── README.md
├── requirements.txt
├── main.py                 # FTR giriş noktası (video.mp4 → results.json)
├── config/config.yaml      # TÜM ayarlar (tek kaynak)
├── src/                    # ÇEKİRDEK (image'a girer)
│   ├── predict.py          # run_inference(video, weights)
│   ├── pipeline.py         # kare döngüsü + orkestrasyon
│   ├── preprocessing.py    # CLAHE (her zaman) + DCP (koşullu) + sis geçidi
│   ├── actions.py          # telefon/sigara/su (nesne + kafa yakınlığı)
│   ├── temporal.py         # slalom + olay debounce + zaman damgalama
│   ├── schema.py           # izinli etiket sabitleri + çıktı doğrulama
│   ├── utils.py            # config, offline env, cihaz, geometri, zaman
│   ├── detection/
│   │   ├── detector.py     # paylaşılan YOLO (tek best_model.pt, FP16, batch)
│   │   ├── vehicle.py      # araç tipi + renk + ana araç konsolidasyonu
│   │   ├── plate.py        # plaka kırp → AYRI OCR → regex normalize
│   │   ├── objects.py      # teknocan, bilgisayar
│   │   ├── passengers.py   # on_koltuk / arka_koltuk_1 / arka_koltuk_2
│   │   ├── seatbelt.py     # emniyet_kemeri_ihlali
│   │   └── sliced.py       # SAHI dilimli çıkarım (bölgesel/seçici)
│   └── face/driver_face.py # MediaPipe Face: baş pozu, esneme, bakış
├── weights/                # GÖMÜLÜ ağırlıklar (bkz. weights/README.md)
├── training/               # fine-tune (image'a GİRMEZ)
│   ├── train_yolo.py · data.yaml · augment.py
└── live/                   # canlı final (image'a GİRMEZ)
    ├── number_verification.py · qod_client.py · risk_model.py · stream_pipeline.py
```

`training/` ve `live/` image'a **kopyalanmaz** (seçici COPY + `.dockerignore`),
böylece image <8GB korunur.

---

## 2. Çekirdek çıkarım akışı

Kare başına:
`kare → ön işleme → YOLO tespit → (kırpılan bölgelerde) yüz/OCR/renk →
zamansal mantık → olay üretimi`. Sonunda tüm video için tek `results.json`.

### Ön işleme (`preprocessing.py`) — problem → çözüm
| Zorluk (değerlendirme videoları) | Çözüm |
|---|---|
| Düşük ışık / kontrast | **CLAHE** (her karede, düşük gecikme; ana iyileştirme) |
| Sis / pus | **DCP dehazing** (dark channel → atmospheric light → transmission → guided filter), **adaptif** sis-skoru geçidi + **histerezis** |
| Hareket bulanıklığı / cam parlaması / oklüzyon | eğitimde sentetik augmentasyon (`training/augment.py`) + zamansal debounce |

DCP modu `config`'ten: `adaptive` (FTR) · `qod_gated` (canlı) · `always` · `off`.
Her aşama bağımsız açılıp kapatılabilir (ablation). Pipeline her karede hangi
adımların çalıştığını ve toplam ms'yi loglar.

### Tespit katmanı (sınıf → yöntem)
- **YOLO11 (fine-tune, tek `best_model.pt`, T4'te FP16, batch):** araç tipi,
  plaka bölgesi, `teknocan`/`bilgisayar`, telefon/sigara/şişe, emniyet kemeri,
  kişi. **Tek YOLO geçişi** tüm yorumlayıcı modüllere dağıtılır (10 dk bütçesi).
- **BoTSORT takibi (`tracking.enabled`):** kalıcı track ID → her aracın
  yörüngesi doğru izlenir. `slalom` artık **track bazlı** (ana araç kareler arası
  değişse bile yörünge karışmaz); `arac_bilgisi` **baskın track**'ten konsolide
  edilir (çoklu araç geçişinde tip/plaka/renk karışmaz). Takip **stateful** olduğu
  için batch yerine **ardışık** çalışır (trade-off: biraz yavaş, doğru ID).
- **SAHI (opsiyonel, bölgesel/seçici):** küçük/uzak nesneler için. Aynı
  `best_model.pt`; yeniden eğitim yok. **Tüm kareye değil**, yalnızca araç/plaka
  bölgesine ve `trigger` (`plate_unreadable` vb.) ile + bütçe throttle ile.
- **Plaka OCR (AYRI aşama):** kırp → EasyOCR (offline) → ASCII/boşluk normalize →
  **resmi regex** doğrulama. Regex dışı sonuç "tespit edilemedi" → boş.
  `plate_ocr.enabled=false` ile kapatılabilir (canlı hafif yol / ablasyon).
- **Araç rengi (`color.method`):** `auto` → `color_cnn.pt` varsa **CNN** (offline),
  yoksa/düşük güvende **HSV** merkez yöntemine düşer. `cnn` / `hsv_centroid` ile
  zorlanabilir. İzinli 9 renk.
- **MediaPipe Face Landmarker:** `arkaya_bakma` (baş yaw), `etrafa_bakinma` (yaw
  salınımı), `esneme` (jawOpen). `.task` gömülü, lokal yüklenir.
- **Yorgunluk yedek CNN'i (`face.fallback_cnn`):** yol kenarı/uzak/açılı/cam ardı
  yüzde MediaPipe başarısız olunca `esneme`/yorgunluğu crop'tan doğrudan kestiren
  offline CNN devreye girer (Geliştirme sarı-1 çözümü). Yoksa MediaPipe'a düşülür.
- **Birleşik eylemler (`actions.py`):** telefon/sigara/şişe nesnesi + sürücü
  kafa/ağız yakınlığı → `telefonla_konusma`/`sigara_icme`/`su_icme`.
- **Zamansal (`temporal.py`):** `slalom` (yanal yörünge salınımı), olay debounce,
  `zaman_saniye = kare_index / fps`, her olaya kendi `confidence_score`'u.

### Çıktı (`schema.py`)
`/app/data/output/results.json`, `json.dump(..., ensure_ascii=False, indent=2)`:
```json
{
  "video_id": "video.mp4",
  "arac_bilgisi": { "tip": "...", "plaka": "...", "renk": "...", "confidence_score": 0.0 },
  "tespitler": [
    { "zaman_saniye": 0.0, "kategori": "sofor_eylemi", "etiket": "...", "confidence_score": 0.0 }
  ]
}
```

---

## 3. FTR'yi çalıştırma (Docker)

Ağırlıkları `weights/` altına koyun (bkz. [weights/README.md](weights/README.md)).

```bash
# 1) Image'ı oluştur (build sırasında internet VAR)
docker build -t teknofest/rapid_response:latest .

# 2) Tesla T4 üzerinde çalıştır (runtime'da internet YOK)
docker run --rm --gpus all \
  -v /yol/video.mp4:/app/data/input/video.mp4 \
  -v /yol/cikti_klasoru:/app/data/output \
  teknofest/rapid_response:latest
```
Çıktı: `/app/data/output/results.json`. T4 yoksa CPU'ya düşer (hedef cuda).

---

## 4. Fine-tune (eğitim)

```bash
# (opsiyonel) sentetik GÖRÜNTÜ verisi üret — gürbüzlük + ablasyon kanıtı
python training/augment.py --images datasets/raw/images \
  --labels datasets/raw/labels --out datasets/synth --per-image 2

# (opsiyonel) ZAMANSAL TUTARLI video augment — sisli/yağmurlu demo/test klipleri
python training/augment_video.py --input demo.mp4 --out demo_fog.mp4 \
  --transforms fog,light --drift
python training/augment_video.py --input-dir clips/ --out-dir clips_aug/ \
  --transforms random --per-video 2

# Offline varlıkları indir (face_landmarker.task + EasyOCR) — internet açıkken bir kez
python training/fetch_assets.py --weights-dir weights

# YOLO11 transfer learning (parametreler config.yaml > training:)
python training/train_yolo.py --config config/config.yaml
python training/train_yolo.py --no-synthetic        # ablasyon (sentetiksiz)

# (opsiyonel) renk CNN'i ve yorgunluk yedek CNN'i — offline gömülür
python training/train_color.py --data datasets/colors --epochs 30
python training/train_drowsiness.py --data datasets/drowsiness --epochs 30
```
En iyi ağırlık `weights/best_model.pt`'ye kopyalanır.

### Ölçüm / değerlendirme (`training/evaluate.py`) — Geliştirme #1 kırmızı madde
Gerçek veriyi nasıl ölçeriz:
- **Dedektör metrikleri** (mAP50/mAP50-95/P/R/FPS): test görüntülerini YOLO
  formatında etiketle (`data.yaml` test split), sonra:
  `python training/evaluate.py --mode detection --weights weights/best_model.pt --data training/data.yaml --split test`
- **Olay/uçtan-uca metrikler** (P/R/F1 + plaka karakter doğruluğu + tip/renk):
  birkaç test videosunu **elle etiketle** → her video için bir ground-truth
  `results.json` (`gt/` klasörü). Sonra:
  `python training/evaluate.py --mode events --gt-dir gt/ --videos-dir test_videos/`
- **QoD açık vs kapalı Δ** (yarışmanın %40'ı): aynı seti iki profille çalıştırıp
  GT'ye göre farkı üretir:
  `python training/evaluate.py --mode qod --gt-dir gt/ --videos-dir test_videos/`
- **Gerçek hız hatası** (MAE/RMSE km/h — Şartname §5, %40 YZ): komiteden gelen
  GT hızlarıyla karşılaştırır:
  `python training/evaluate.py --mode speed --gt-speeds gt_speeds.json --videos-dir test_videos/`

### Hız kalibrasyonu (`training/calibrate_speed.py`) — IPM/homografi
Gerçek hız, FTR `results.json`'a **yazılmaz** (FTR şemasında hız alanı yok); yalnızca
`live/` + ölçüm tarafındadır. Akış:
- `--mode extract`: örnek videolardan **temiz referans kare** (medyan arka plan) +
  şerit/sahne işareti adayları (Canny+Hough) çıkarır (annotasyon için).
- `--mode homography`: seçtiğin ≥4 görüntü noktası + **gerçek ölçülmüş** dünya
  koordinatlarından (metre) homografiyi üretir → `config.speed.homography`.
- `--mode validate`: GT hızlarıyla MAE/RMSE raporlar + Geliştirme referansı
  (Gajdoš IPM ~0.58 km/h) ile kıyaslar.
> Not: metrik ölçek **gerçek mesafe ölçümünden** gelir; pikselden metre uydurulmaz.

### Geometri kalibrasyonu (`training/calibrate_geometry.py`)
Sürücü/koltuk eşikleri **tahmin değil, veriden** çıkarılır: örnek videolarda kişi
konumlarını kümeler, `front_y_max` / `back_left_x_max` / `driver_selection` önerir
(config'e elle yapıştırılır):
`python training/calibrate_geometry.py --videos-dir sample_videos/ --weights weights/best_model.pt`
İsteğe bağlı TensorRT/ONNX export (`training.export_format`).

### Veri seti metodolojisi (özet)
- **Kaynaklar:** COCO (laptop/phone/bottle/person), UA-DETRAC + BDD100K (araç),
  açık Türk plaka setleri, açık seatbelt/sigara setleri. `teknocan` **yarışmaya
  özel** → kendi toplanan + **sentetik** örnekler (`augment.py`).
- **Bölme:** train/val/test = 0.70 / 0.15 / 0.15, sınıf bazlı dengeli.
- **Dengeleme:** nadir sınıflar (teknocan, plaka) için oversampling + sentetik.
- **Augmentasyon:** sis/yağmur/parlaklık/gamma/motion blur/çözünürlük/JPEG
  (foto-metrik → bbox değişmez, etiketler korunur). Kare bazlı dedektör için
  görüntü augmentasyonu ([augment.py](training/augment.py)) yeterlidir.
- **Video augmentasyonu** ([augment_video.py](training/augment_video.py)): zamansal
  modüller (slalom, etrafa_bakinma, debounce) için **klip-tutarlı** parametre +
  opsiyonel yavaş drift. Her kareye bağımsız rastgele augment **uygulanmaz**
  (flicker → zamansal sinyalleri bozar). Demo/test ve uçtan uca ablasyon için.

---

## 5. Canlı final mimarisi (`live/`)

Puanlamanın **%40'ı QoD'u yalnızca ihtiyaç varken kullanmaya** bağlıdır; mimari
bunu **kanıtlanabilir** biçimde gösterir.

- **`number_verification.py`:** ilk adımda SMS'siz, sessiz, **şebeke tabanlı**
  doğrulama (CAMARA Number Verification).
- **`risk_model.py`:** Türk Trafik Kanunu'na dayalı **nesnel** risk (0–100) +
  profil (Güvenli/Dikkatli/Riskli/Tehlikeli). **risk(0–100) ↔ güven(0–1)
  köprüsü:** bunlar **dik** eksenlerdir; QoD ihtiyacı
  `need = severity_potential × (1 − confidence)`.
- **`qod_client.py`:** CAMARA QoD `/sessions` (aç/serbest bırak) +
  **`QoDController`** (histerezisli, ihtiyaç tabanlı tetikleme).
  **Çelişki düzeltmesi:** eski "risk<50 ve risk>85 kapat" kuralı yerine tek
  tutarlı ilke — *AÇ: need≥eşik (k kare); KAPAT: need<eşik (m kare, cooldown)*.
  Düşük risk+emin → need≈0 → kapalı; yüksek risk+emin (karar verildi) → need≈0
  → kapalı; **belirsiz orta bölge → need yüksek → açık** ("en belirsiz → en
  yüksek destek"). İki eski kapatma koşulu, bu tek ilkenin doğal sonucudur.
- **`speed.py`:** **gerçek hız** (Şartname §5, %40 YZ) — IPM/homografi ile araç
  yer-temas noktası yer düzlemine yansıtılır, BoTSORT track'inin dünya konumu
  zaman üzerinde regresyonla km/h'ye çevrilir. **FTR `results.json`'a girmez**
  (FTR > Şartname; şemada hız alanı yok); sadece canlı UI + ölçüm.
- **`stream_pipeline.py`:** İKİ ÇIKARIM YOLU — *Hafif* (CLAHE + tek YOLO) ve
  *Ağır* (DCP + SAHI + OCR, QoD aktifken). **Aynı `best_model.pt`** — kazanç
  QoD+SAHI+OCR'den gelir, büyük modelden değil. QoD **açık vs kapalı** başarım
  farkı (`evidence_report`) ile ölçülür/raporlanır (puanlama kanıtı). Hız
  tahmini de burada üretilip UI'a yayınlanır.

---

## 6. Performans bütçesi (T4 / 10 dk)

- Kare örnekleme (`analiz_fps`), YOLO batch, FP16.
- Ağır işler (OCR, MediaPipe, DCP, SAHI) **yalnızca gerektiğinde** tetiklenir.
- Toplam süre ölçülür/loglanır; bütçe aşımına karşı **adaptif örnekleme**
  (`max_runtime_seconds`, `adaptive_min_fps`).

---

## 7. Config açıklaması (`config/config.yaml`)

| Bölüm | İşlev |
|---|---|
| `paths` | giriş/çıkış, ağırlık aday yolları, OCR/face model yolları |
| `runtime` | cihaz (auto/cuda/cpu), FP16, offline env |
| `performance` | analiz_fps, batch, imgsz, süre bütçesi, adaptif örnekleme |
| `tracking` | **BoTSORT** aç/kapa, tracker, stale temizleme |
| `preprocessing` | CLAHE, sis skoru, DCP (mod + histerezis) |
| `detection` | eşikler + **model sınıf adı → şema** eşlemeleri |
| `color` | renk yöntemi (auto/cnn/hsv) + HSV eşikleri |
| `plate_ocr` | OCR aç/kapa, motoru, eşik, **resmi regex** |
| `sahi` | dilimli çıkarım (mod/tetikleyici/slice) |
| `face` | MediaPipe eşikleri (yaw/esneme/bakınma) + yedek CNN |
| `actions` | yakınlık + min_frames |
| `evaluate` | olay eşleşme toleransı + QoD açık/kapalı profilleri |
| `speed` | IPM/homografi, kontak noktası, pencere, yumuşatma (live/ölçüm) |
| `passengers` | koltuk bölge eşikleri |
| `temporal` | slalom + debounce + zaman damgalama politikası |
| `vehicle_summary` | tip/plaka/renk oylama + birleşik güven |
| `output` | ASCII/indent, geçersiz etiket düşürme, yuvarlama |
| `training` | fine-tune parametreleri (image'a girmez) |

---

## 8. Varsayımlar (belirsizlikler makul biçimde dolduruldu)

- **OCR kütüphanesi:** EasyOCR (offline `model_storage_directory` +
  `download_enabled=False`). Türk plakaları latin harf+rakam → `en` yeterli.
- **`teknocan` veri kaynağı:** açık veri yok → kendi toplanan + sentetik
  (`augment.py`). `bilgisayar` için COCO "laptop".
- **Sürücü seçimi:** araç içindeki en büyük kişi; yoksa aracın üst ~%60'ı (kabin).
- **Ağırlık yolu çelişkisi** (`/app/weights` vs `/app/models`): config tek
  kaynak + Dockerfile symlink → her iki yol geçerli.
- **Resmi plaka regex'i** dokümandan birebir; dokümandaki `a-zAZ` OCR artefaktı
  `a-zA-Z` olarak düzeltildi.

---

## 9. Teslim öncesi kontrol listesi (otomasyon)

`python -m src.schema` çıktı JSON'ını sema kurallarına göre doğrular:
- [x] Dockerfile root level · base image `nvidia/cuda:12.1.0-base-ubuntu22.04`
- [x] GPU (cuda) + FP16; T4 yoksa CPU
- [x] Giriş `/app/data/input/video.mp4` · Çıkış `/app/data/output/results.json`
- [x] Etiketler ASCII + küçük harf · anahtarlar birebir · plaka regex'e uygun/boş
- [x] `docker run` ile otomatik başlar (CMD)
- [x] Çökme yok (boş results.json fallback) · anti-hile yok
