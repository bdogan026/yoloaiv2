"""train_tip.py — Hafif arac GOVDE TIPI CNN egitimi — image'a GIRMEZ.

train_color.py'nin tip karsiligi. mobilenet_v3_small ile 7 sinifli tip
siniflandirici egitir; en iyi agirligi weights/tip_cnn.pt olarak kaydeder
(cikarimda src/detection/tip_cnn.py bunu offline yukler). Sinif sirasi
schema.ARAC_TIPLERI ile BIREBIR zorlanir.

Veri: --data altinda her tip icin bir alt klasor (sedan/, suv/, ... kamyon/).
Kaynak: DETRAC pseudo-label crop'larinin elle etiketlenmesi (training/
sample_crops_for_labeling.py) ve/veya BoxCars gibi govde-tipi veri setleri.

AUGMENT farki (renk vs tip): tipte RENK ONEMSIZ -> agresif ColorJitter; sekil
ONEMLI -> guclu geometrik (flip, affine, perspektif). Boylece model rengi
yok sayip govde siluetine odaklanir.

Kullanim:
    python training/train_tip.py --data datasets/tip --epochs 40
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schema import ARAC_TIPLERI  # noqa: E402

CANON = list(ARAC_TIPLERI)  # cikarimla birebir sinif sirasi


class _Remap:
    """ImageFolder etiketini CANON sirasina cevirir (picklable; spawn/worker uyumu)."""

    def __init__(self, mapping: dict) -> None:
        self.mapping = mapping

    def __call__(self, y: int) -> int:
        return self.mapping.get(y, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Arac govde tipi CNN egitimi")
    ap.add_argument("--data", required=True, help="tip alt klasorlu kok dizin")
    ap.add_argument("--out", default="weights/tip_cnn.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--imgsz", type=int, default=96)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader isci sayisi (macOS'ta 0 onerilir)")
    ap.add_argument("--no-balance", action="store_true",
                    help="sinif-agirlikli loss'u kapat (varsayilan: dengeli)")
    ap.add_argument("--scratch", action="store_true",
                    help="ImageNet pretrained YERINE sifirdan egit (onerilmez)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        from torchvision.models import mobilenet_v3_small
    except ImportError as exc:
        print(f"torch/torchvision gerekli: {exc}", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # EGITIM: tip = SEKIL tabanli -> guclu GEOMETRIK augment; renk ONEMSIZ ->
    # agresif ColorJitter (model rengi yok saysin, siluete odaklansin).
    train_tf = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=12, translate=(0.08, 0.08),
                                scale=(0.85, 1.15), shear=6),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        norm,
    ])
    # DOGRULAMA: augment YOK -> durust val_acc
    eval_tf = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        norm,
    ])

    ds_train = ImageFolder(args.data, transform=train_tf)
    ds_eval = ImageFolder(args.data, transform=eval_tf)

    # Sinif sirasini CANON ile birebir hizala (egitim sirasi = cikarim sirasi)
    unknown = [c for c in ds_train.classes if c not in CANON]
    if unknown:
        print(f"UYARI: CANON disi klasorler atlanmali/duzeltilmeli: {unknown}",
              file=sys.stderr)
    remap = {ds_train.class_to_idx[c]: CANON.index(c)
             for c in ds_train.classes if c in CANON}
    ds_train.target_transform = _Remap(remap)
    ds_eval.target_transform = _Remap(remap)

    # Tutarli train/val ayrimi (ayni indeks havuzu; val'de augment yok)
    n = len(ds_train)
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    n_val = max(1, int(n * args.val_split))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr = Subset(ds_train, tr_idx)
    va = Subset(ds_eval, val_idx)
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers)
    dl_va = DataLoader(va, batch_size=args.batch, num_workers=args.workers)

    # Transfer learning: ImageNet on-egitimli backbone -> sifirdan COK daha iyi
    # (model sekilleri zaten bilir; biz sadece 7-tip son katmani ince-ayar yapariz).
    # Egitimde internet ACIK (backbone bir kez iner); sonuc offline gomulur.
    # --scratch ile eski (sifirdan) davranis. Inference tip_cnn.py degismez
    # (mimari ayni: mobilenet_v3_small + 7-sinif son katman).
    if args.scratch:
        model = mobilenet_v3_small(weights=None, num_classes=len(CANON))
    else:
        model = mobilenet_v3_small(weights="DEFAULT")
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_f, len(CANON))
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Sinif dengesizligine karsi ters-frekans agirligi (sedan baskinligini kir).
    # Cok az ornekli sinifin agirligi patlamasin diye [0.2, 6] araliginda kirpilir.
    import collections
    cnt = collections.Counter(remap.get(ds_train.samples[i][1], 0) for i in tr_idx)
    nC = len(CANON)
    if args.no_balance:
        weights = None
    else:
        w = [min(3.0, max(0.3, len(tr_idx) / (nC * cnt[i]))) if cnt.get(i, 0) else 0.0
             for i in range(nC)]
        weights = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"[train_tip] train sinif sayilari: "
          f"{ {CANON[i]: cnt.get(i, 0) for i in range(nC)} }")
    crit = torch.nn.CrossEntropyLoss(weight=weights)

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += int((pred == y).sum())
                total += int(y.numel())
        acc = correct / max(1, total)
        print(f"epoch {ep+1}/{args.epochs}  val_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.out)
    print(f"[train_tip] en iyi val_acc={best_acc:.4f} -> {args.out}")
    print(f"[train_tip] sinif sirasi: {CANON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
