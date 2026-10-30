"""NCT2013-format micro-ultrasound cores (one B-mode image per core), used to train the DINOv3 cancer head and, with the
fixed needle template, as extra B-mode variety for NeedleNet. No data ships with this repository.

Layout (all values are yours):
    <nct_root>/
    ├── metadata.csv      core_id, patient_id, center, bmode_path, prostate_mask_path, grade_group, involvement
    │                     (paths relative to <nct_root>; involvement = cancer fraction of the core, 0-1)
    └── needle_mask.npy   the needle region, one fixed template for every core (standardised probe geometry)

Infra first (metadata, folds, one core), then the torch dataset and dataloaders at the bottom.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

N_FOLDS = 5


def read_metadata(root) -> pd.DataFrame:
    return pd.read_csv(Path(root) / "metadata.csv", dtype={"core_id": str, "patient_id": str})


def patient_folds(meta: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = 0) -> dict[str, int]:
    """Patient-level folds, balanced for patients with csPCa: patients shuffled within each label, dealt round-robin."""
    y = meta.groupby("patient_id").grade_group.max().ge(2)
    rng = np.random.default_rng(seed)
    fold = {}
    for label in (True, False):
        ids = sorted(y.index[y == label])
        rng.shuffle(ids)
        fold.update({p: i % n_folds for i, p in enumerate(ids)})
    return fold


def split(meta: pd.DataFrame, fold: int, exclude_benign_from_positive_patients: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training cores, held-out cores) of a fold. As for the paper's DINOv3 head, benign cores of patients with csPCa can be
    left out of training (their tissue may still contain cancer that the core missed)."""
    f = meta.patient_id.map(patient_folds(meta))
    train, held_out = meta[f != fold], meta[f == fold]
    if exclude_benign_from_positive_patients:
        pos = set(train.patient_id[train.grade_group >= 2])
        train = train[~(train.patient_id.isin(pos) & (train.grade_group < 2))]
    return train.reset_index(drop=True), held_out.reset_index(drop=True)


def read_bmode(path) -> np.ndarray:
    """The core's B-mode image as (H, W) uint8 (the first frame if the file holds a cine)."""
    path = Path(path)
    if path.suffix == ".npy":
        a = np.load(path)
        a = a[..., 0] if a.ndim == 3 else a
        return (255 * (a - a.min()) / (a.max() - a.min() + 1e-8)).astype(np.uint8)
    return np.asarray(Image.open(path).convert("L"))


def read_core(root, row, needle_template: np.ndarray) -> dict:
    root = Path(root)
    img = read_bmode(root / row.bmode_path)
    shape = img.shape
    resize = lambda m: np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)) > 0
    prostate = (
        np.asarray(Image.open(root / row.prostate_mask_path).convert("L")) > 0
        if isinstance(row.prostate_mask_path, str)
        else np.ones(shape, bool)
    )
    return {
        "image": img,
        "prostate_mask": resize(prostate),
        "needle_mask": resize(needle_template > 0),
        "core_id": row.core_id,
        "label": float(row.grade_group >= 2),
        "grade_group": float(row.grade_group),
        "involvement": float(row.involvement or 0.0),
    }


# ── dataset and dataloaders ─────────────────────────────────────────────────────────────────────────────────────────


class NCT2013Cores:
    def __init__(self, root, table: pd.DataFrame, transform=None):
        self.root, self.table, self.transform = Path(root), table, transform
        self.needle_template = np.load(self.root / "needle_mask.npy")

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        item = read_core(self.root, self.table.iloc[i], self.needle_template)
        return self.transform(item) if self.transform else item


def nct2013_loaders(root, fold: int, train_transform, eval_transform, batch_size=8, num_workers=4):
    """(training loader, held-out loader) for one fold."""
    from torch.utils.data import DataLoader

    train, held_out = split(read_metadata(root), fold)
    return (
        DataLoader(
            NCT2013Cores(root, train, train_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        ),
        DataLoader(NCT2013Cores(root, held_out, eval_transform), batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
