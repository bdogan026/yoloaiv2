"""train_color.py — Hafif arac rengi CNN egitimi — image'a GIRMEZ.

mobilenet_v3_small ile 9 sinifli renk siniflandirici egitir; en iyi agirligi
weights/color_cnn.pt olarak kaydeder (cikarimda src/detection/color_cnn.py
bunu offline yukler). Sinif sirasi schema.ARAC_RENKLERI ile BIREBIR zorlanir.

Veri: --data altinda her renk icin bir alt klasor (beyaz/, siyah/, ... ).
Acik kaynak: VCoR (Vehicle Color Recognition) vb. + kendi crop'larin.

Kullanim:
    python training/train_color.py --data datasets/colors --epochs 30
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schema import ARAC_RENKLERI  # noqa: E402

CANON = list(ARAC_RENKLERI)  # cikarimla birebir sinif sirasi


class _Remap:
    """ImageFolder etiketini CANON sirasina cevirir. (lambda DEGIL; spawn/worker
    altinda picklable olmali — macOS/Windows DataLoader uyumu.)"""

    def __init__(self, mapping: dict) -> None:
        self.mapping = mapping

    def __call__(self, y: int) -> int:
        return self.mapping.get(y, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Arac rengi CNN egitimi")
    ap.add_argument("--data", required=True, help="renk alt klasorlu kok dizin")
    ap.add_argument("--out", default="weights/color_cnn.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--imgsz", type=int, default=64)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader isci sayisi (macOS'ta 0 onerilir)")
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
        device = "mps"            # Apple Silicon GPU (Mac'te hizli egitim)
    else:
        device = "cpu"

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # EGITIM: renk-GUVENLI augmentasyon (renk etiketini BOZMADAN gurbuzluk).
    #   flip/affine/blur -> poz, isik, cozunurluk cesitliligi
    #   ColorJitter hue KUCUK (0.02): kirmizi->kahverengi gibi kaymalar olmasin
    #   fog / agir isik YOK (renk siniflandirmada etiketi bozardi)
    train_tf = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
        transforms.GaussianBlur(3, sigma=(0.1, 1.2)),
        transforms.ToTensor(),
        norm,
    ])
    # DOGRULAMA: augmentasyon YOK -> durust val_acc
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

    # Tutarli train/val ayrimi (ayni indeks havuzu; val'de augmentasyon yok)
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

    model = mobilenet_v3_small(weights=None, num_classes=len(CANON)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        # val
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
    print(f"[train_color] en iyi val_acc={best_acc:.4f} -> {args.out}")
    print(f"[train_color] sinif sirasi: {CANON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
