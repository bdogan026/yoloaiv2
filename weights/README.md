# weights/ — GOMULU model agirliklari (image'a girer, offline yuklenir)

Bu klasor Docker image'ina kopyalanir ve **calisma aninda internet KAPALI** iken
lokal yoldan yuklenir. Asagidaki dosyalari buraya koyun:

```
weights/
├── best_model.pt          # YOLO11 fine-tuned detektor (training/train_yolo.py uretir)
├── face_landmarker.task   # MediaPipe Face Landmarker modeli
├── color_cnn.pt           # (ops.) arac rengi CNN'i (training/train_color.py)
├── drowsiness_cnn.pt      # (ops.) yorgunluk/esneme yedek CNN (train_drowsiness.py)
└── ocr/                   # EasyOCR onceden indirilmis modeller (offline)
    ├── craft_mlt_25k.pth          # detection
    └── english_g2.pth             # recognition (en)
```

> `face_landmarker.task` ve `ocr/` icin: `python training/fetch_assets.py`
> (gelistirme makinesinde, internet acikken bir kez). `color_cnn.pt` ve
> `drowsiness_cnn.pt` OPSIYONELDIR: yoksa renk HSV'ye, yorgunluk MediaPipe'a
> duser (sistem yine calisir). Egitilirse dogruluk artar.

## best_model.pt
`training/train_yolo.py` ciktisidir (`export_best_to: weights/best_model.pt`).
`model.names`, `config/config.yaml` -> `detection.*` ve `training/data.yaml`
sinif adlariyla **birebir** olmalidir.

## face_landmarker.task
MediaPipe Face Landmarker bundle. Bir kez indirilip buraya konur (build
makinesinde), image'a gomulur. Kaynak:
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`

## ocr/
EasyOCR modelleri bir kez indirilip bu klasore konur; calisma aninda
`download_enabled=False` ile yalnizca buradan yuklenir (offline garanti).
Indirme (build makinesinde, bir kez):
```python
import easyocr
easyocr.Reader(["en"], model_storage_directory="weights/ocr", download_enabled=True)
```

> NOT: Agirlik yolu celiskisi (ornek main.py `/app/weights`, VM tablosu
> `/app/models`) Dockerfile'da `/app/models -> /app/weights` symlink'i ile
> cozulur; her iki yol da gecerlidir.
