"""train_drowsiness.py — Yorgunluk/esneme yedek CNN egitimi — image'a GIRMEZ.

mobilenet_v3_small ile 3 sinifli (normal / esneme / goz_kapali) siniflandirici
egitir; weights/drowsiness_cnn.pt olarak kaydeder (cikarimda
src/face/drowsiness_cnn.py offline yukler). Sinif sirasi DROWSY_CLASSES ile
birebir zorlanir.

Veri: --data altinda normal/ esneme/ goz_kapali/ alt klasorleri (surucu yuz/kafa
crop'lari). Acik kaynak: YawDD, NTHU-DDD, Roboflow "drowsiness/yawn" + kendi.

Kullanim:
    python training/train_drowsiness.py --data datasets/drowsiness --epochs 30
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Sinif sirasi cikarim moduluyle birebir (tek kaynak)
DROWSY_CLASSES = ["normal", "esneme", "goz_kapali"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Yorgunluk/esneme yedek CNN egitimi")
    ap.add_argument("--data", required=True, help="normal/esneme/goz_kapali alt klasorleri")
    ap.add_argument("--out", default="weights/drowsiness_cnn.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--imgsz", type=int, default=96)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, random_split
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        from torchvision.models import mobilenet_v3_small
    except ImportError as exc:
        print(f"torch/torchvision gerekli: {exc}", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = ImageFolder(args.data, transform=tf)
    unknown = [c for c in ds.classes if c not in DROWSY_CLASSES]
    if unknown:
        print(f"UYARI: beklenmeyen klasorler: {unknown}", file=sys.stderr)
    remap = {ds.class_to_idx[c]: DROWSY_CLASSES.index(c)
             for c in ds.classes if c in DROWSY_CLASSES}
    ds.target_transform = lambda y: remap.get(y, 0)

    n_val = max(1, int(len(ds) * args.val_split))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(args.seed))
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=2)
    dl_va = DataLoader(va, batch_size=args.batch, num_workers=2)

    model = mobilenet_v3_small(weights=None,
                               num_classes=len(DROWSY_CLASSES)).to(device)
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
    print(f"[train_drowsiness] en iyi val_acc={best_acc:.4f} -> {args.out}")
    print(f"[train_drowsiness] sinif sirasi: {DROWSY_CLASSES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
