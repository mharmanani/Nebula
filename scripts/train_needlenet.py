"""Train NeedleNet: MicroSegNet (TransUNet) started from the public prostate weights and fine-tuned on needle masks.

    python3 scripts/train_needlenet.py --fold 0 --out needlenet.pth [--nct2013 /path/to/nct2013]

Data: your OPTIMUM-format cores with an annotated needle track on the fire frame (`needle_mask.png`), split by patient with
the released folds; optionally NCT2013-format cores, whose needle region is one fixed template (more B-mode variety, no
localisation signal). 224 px, binary cross-entropy, AdamW 1e-4, 20 epochs, batch 8; the epoch with the best held-out Dice
is kept.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import skimage.transform
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from config import load_config
from data import NCT2013
from data.OPTIMUM import Cohort, CoreFrameDataset
from models.microsegnet import build_microsegnet

SIZE = 224


def to_tensors(item):
    """Frame scaled to [0, 1] and needle mask, both at 224 x 224."""
    img = skimage.transform.resize(item["image"] / 255.0, (SIZE, SIZE), anti_aliasing=True).astype(np.float32)
    mask = skimage.transform.resize(item["needle_mask"].astype(np.float32), (SIZE, SIZE), order=0) > 0.5
    return torch.from_numpy(img)[None], torch.from_numpy(mask.astype(np.float32))[None]


def dice(logits, mask) -> float:
    pred = logits > 0
    inter = (pred & (mask > 0.5)).sum((1, 2, 3)).float()
    total = pred.sum((1, 2, 3)) + (mask > 0.5).sum((1, 2, 3))
    return float(torch.where(total > 0, 2 * inter / total.clamp(min=1), torch.ones_like(inter)).mean())


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    scores = []
    for x, m in loader:
        x, m = x.to(device), m.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, m)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scores.append(dice(logits.detach(), m))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nct2013")
    ap.add_argument("--splits", default=str(ROOT / "splits" / "optimum_5fold_patient_splits.json"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config()
    device = cfg.get("device", "cuda")
    cohort = Cohort(cfg["cohort"], a.splits)
    train_ids, held_ids = cohort.split(a.fold)
    train, held = [CoreFrameDataset(cohort, train_ids, to_tensors)], [CoreFrameDataset(cohort, held_ids, to_tensors)]
    if a.nct2013:
        meta = NCT2013.read_metadata(a.nct2013)
        tr, he = NCT2013.split(meta, a.fold, exclude_benign_from_positive_patients=False)
        train.append(NCT2013.NCT2013Cores(a.nct2013, tr, to_tensors))
        held.append(NCT2013.NCT2013Cores(a.nct2013, he, to_tensors))
    train_loader = DataLoader(ConcatDataset(train), batch_size=a.batch_size, shuffle=True, num_workers=4)
    held_loader = DataLoader(ConcatDataset(held), batch_size=a.batch_size, num_workers=4)
    model = build_microsegnet().to(device)  # public prostate weights as the starting point
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    best = -1.0
    for epoch in range(a.epochs):
        tr, he = run_epoch(model, train_loader, device, optimizer), run_epoch(model, held_loader, device)
        if he > best:
            best = he
            torch.save(model.net.state_dict(), a.out)
        print(f"epoch {epoch}: train Dice {tr:.3f}, held-out Dice {he:.3f} (best {best:.3f})", flush=True)


if __name__ == "__main__":
    main()
