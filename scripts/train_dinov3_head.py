"""Train the DINOv3 cancer head (the second-opinion map) on NCT2013-format cores.

    python3 scripts/train_dinov3_head.py --nct2013 /path/to/nct2013 --fold 0 --out dinov3_head.pth

Recipe of the paper's model: DINOv3 ViT-L/16 (pretrained weights from the dinov3 package) with a prompt-free mask decoder,
384 px, ImageNet normalisation, frames flipped vertically to the OPTIMUM orientation, random translations; loss = binary
cross-entropy between the mean predicted probability inside the needle region and the prostate and the core's involvement,
plus 0.01 x the mean pixel entropy; AdamW (decoder 1e-5, backbone 1e-6, no weight decay), 20 epochs, batch 8; benign cores
of patients with csPCa left out of training; the epoch with the best held-out core AUROC is kept.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data.NCT2013 import nct2013_loaders
from models.dinov3_head import build_dinov3_head
from tools.dinov3 import MEAN, STD

SIZE, ENTROPY_WEIGHT, MAX_SHIFT = 384, 0.01, 0.1


class Preprocess:
    def __init__(self, train: bool):
        self.train = train

    def __call__(self, item):
        img, masks = item["image"].astype(np.float32), np.stack([item["prostate_mask"], item["needle_mask"]]).astype(np.float32)
        img, masks = np.flipud(img), masks[:, ::-1]  # NCT2013 frames flipped vertically to the OPTIMUM orientation
        x = torch.from_numpy(img.copy())[None, None].repeat(1, 3, 1, 1)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)[0]
        m = F.interpolate(torch.from_numpy(masks.copy())[None], size=(SIZE, SIZE), mode="nearest")[0]
        if self.train:  # random translation of the frame and its masks together
            dy, dx = (np.random.uniform(-MAX_SHIFT, MAX_SHIFT, 2) * SIZE).astype(int)
            x, m = torch.roll(x, (dy, dx), (1, 2)), torch.roll(m, (dy, dx), (1, 2))
        return ((x - MEAN[0]) / STD[0]).float(), m, torch.tensor(item["involvement"]).float(), torch.tensor(item["label"]).float()


def region_probs(logits, masks):
    m = F.interpolate(masks, size=logits.shape[-2:], mode="nearest")
    return [torch.sigmoid(logits[i, 0])[(m[i, 0] > 0.5) & (m[i, 1] > 0.5)] for i in range(len(logits))]


def mil_loss(logits, masks, involvement):
    loss = 0.0
    for p, inv in zip(region_probs(logits, masks), involvement):
        if not len(p):
            continue
        p = p.clamp(1e-7, 1 - 1e-7)
        q = p.mean()
        loss = (
            loss - inv * q.log() - (1 - inv) * (1 - q).log() + ENTROPY_WEIGHT * (-(p * p.log()) - (1 - p) * (1 - p).log()).mean()
        )
    return loss


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nct2013", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    train, held = nct2013_loaders(a.nct2013, a.fold, Preprocess(train=True), Preprocess(train=False))
    model = build_dinov3_head(pretrained_backbone=True).to(a.device)
    backbone = list(model.model.backbone.parameters())
    ids = {id(p) for p in backbone}
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": 1e-6}, {"params": [p for p in model.parameters() if id(p) not in ids], "lr": 1e-5}],
        weight_decay=0.0,
    )
    scaler, best = torch.amp.GradScaler(), -1.0
    for epoch in range(a.epochs):
        model.train()
        for x, m, inv, _ in train:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(bmode=x.to(a.device))["cancer_logits"]
            loss = mil_loss(logits.float(), m.to(a.device), inv.to(a.device))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for x, m, _, y in held:
                logits = model(bmode=x.to(a.device))["cancer_logits"].float()
                scores += [float(p.mean()) if len(p) else np.nan for p in region_probs(logits, m.to(a.device))]
                labels += y.tolist()
        scores, labels = np.array(scores), np.array(labels)
        ok = ~np.isnan(scores)
        auc = roc_auc_score(labels[ok], scores[ok])
        if auc > best:
            best = auc
            torch.save(model.state_dict(), a.out)
        print(f"epoch {epoch}: held-out core AUROC {auc:.3f} (best {best:.3f})", flush=True)


if __name__ == "__main__":
    main()
