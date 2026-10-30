"""OPTIMUM-format micro-ultrasound studies: one folder per patient with the rotational sweep, the biopsy cores and the
clinical values (layout in docs/data_format.md). No data ships with this repository.

Infra first (frames, `Core`, `Sweep`, `Case`, `Cohort`), then the torch datasets and dataloaders used to train the
specialists at the bottom.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CSPCA_MIN_GRADE_GROUP = 2


def read_frame(path) -> np.ndarray:
    """A B-mode frame as a (H, W) uint8 array."""
    return np.asarray(Image.open(path).convert("L"))


def read_mask(path, shape=None) -> np.ndarray:
    """A binary mask as a (H, W) bool array, resized (nearest) to `shape` if given."""
    m = Image.open(path).convert("L")
    if shape is not None:
        m = m.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(m) > 0


def _read_json(path) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


@dataclass
class Core:
    """One biopsy core: its cine (or a single frame), where the probe was and, for training / evaluation only, its labels."""

    core_id: str
    folder: Path
    fire_frame: int
    roll_deg: float
    width_mm: float
    depth_mm: float
    location: str | None = None
    labels: dict | None = None

    @classmethod
    def load(cls, folder) -> "Core":
        folder = Path(folder)
        meta = _read_json(folder / "meta.json")
        n = len(list((folder / "frames").glob("*.png")))
        return cls(
            core_id=folder.name,
            folder=folder,
            fire_frame=int(meta.get("fire_frame", n - 1)),
            roll_deg=float(meta["roll_deg"]),
            width_mm=float(meta["width_mm"]),
            depth_mm=float(meta["depth_mm"]),
            location=meta.get("location"),
            labels=_read_json(folder / "labels.json") or None,
        )

    @property
    def frame_paths(self) -> list[Path]:
        return sorted((self.folder / "frames").glob("*.png"))

    def frame(self, index: int | None = None) -> np.ndarray:
        """The frame at `index`; by default the last frame before the needle fires."""
        return read_frame(self.frame_paths[self.fire_frame if index is None else index])

    def needle_mask(self, shape=None) -> np.ndarray | None:
        """The annotated needle track on the fire frame (`needle_mask.png`, optional)."""
        p = self.folder / "needle_mask.png"
        return read_mask(p, shape) if p.exists() else None

    def mm_per_px(self, shape) -> tuple[float, float]:
        """(mm per pixel down, mm per pixel across) for a frame of this shape; pixels are not square."""
        return self.depth_mm / shape[0], self.width_mm / shape[1]

    @property
    def cspca(self) -> int | None:
        if not self.labels or self.labels.get("grade_group") is None:
            return None
        return int(self.labels["grade_group"] >= CSPCA_MIN_GRADE_GROUP)


@dataclass
class Sweep:
    """The rotational sweep: frames in acquisition order and the probe roll of each frame."""

    folder: Path
    roll: np.ndarray
    width_mm: float
    depth_mm: float

    @classmethod
    def load(cls, folder) -> "Sweep":
        folder = Path(folder)
        meta = _read_json(folder / "meta.json")
        return cls(
            folder=folder,
            roll=np.load(folder / "roll.npy").astype(float),
            width_mm=float(meta["width_mm"]),
            depth_mm=float(meta["depth_mm"]),
        )

    @property
    def frame_paths(self) -> list[Path]:
        return sorted((self.folder / "frames").glob("*.png"))

    def __len__(self) -> int:
        return len(self.frame_paths)

    def frame(self, k: int) -> np.ndarray:
        return read_frame(self.frame_paths[k])

    @property
    def roll_aligned(self) -> np.ndarray:
        """Roll on the cores' reference. Some scanners store the sweep roll on a reference turned by 180 degrees (a range over
        200 degrees or values beyond +-120); those are wrapped to (-180, 180]."""
        r = self.roll
        if r.max() - r.min() > 200 or np.abs(r).max() > 120:
            return (r + 360) % 360 - 180
        return r

    def mm_per_px(self, shape) -> tuple[float, float]:
        return self.depth_mm / shape[0], self.width_mm / shape[1]


@dataclass
class Case:
    """One patient: the sweep (if any), every core (in acquisition order) and age / PSA."""

    case_id: str
    folder: Path
    center: str | None = None
    age: float | None = None
    psa: float | None = None
    sweep: Sweep | None = None
    cores: list[Core] = field(default_factory=list)

    @classmethod
    def load(cls, folder, center: str | None = None) -> "Case":
        folder = Path(folder)
        clin = _read_json(folder / "clinical.json")
        if center is None and (folder.parent / "cohort.csv").exists():
            t = pd.read_csv(folder.parent / "cohort.csv", dtype={"case_id": str}).set_index("case_id")
            center = t.center.get(folder.name)
        sweep = Sweep.load(folder / "sweep") if (folder / "sweep" / "roll.npy").exists() else None
        cores = [Core.load(d) for d in sorted((folder / "cores").iterdir()) if d.is_dir()] if (folder / "cores").exists() else []
        return cls(
            case_id=folder.name, folder=folder, center=center, age=clin.get("age"), psa=clin.get("psa"), sweep=sweep, cores=cores
        )

    def core(self, core_id: str) -> Core:
        return next(c for c in self.cores if c.core_id == core_id)

    @property
    def cspca(self) -> int | None:
        """Patient label: csPCa in any core."""
        y = [c.cspca for c in self.cores if c.cspca is not None]
        return max(y) if y else None


class Cohort:
    """A folder of cases with `cohort.csv` (case_id, center[, fold]); folds come from the csv or from a splits file in the
    format of splits/optimum_5fold_patient_splits.json."""

    def __init__(self, root, splits_file=None):
        self.root = Path(root)
        self.table = pd.read_csv(self.root / "cohort.csv", dtype={"case_id": str}).set_index("case_id")
        if splits_file is not None:
            s = json.loads(Path(splits_file).read_text())
            n = sum(k.startswith("stratified-kfold_fold-") and "center" not in k for k in s)
            fold = {c: f for f in range(n) for c in s[f"stratified-kfold_fold-{f}"]["val"]}
            self.table["fold"] = self.table.index.map(fold)
        self.table = self.table[[(self.root / c).is_dir() for c in self.table.index]]

    @property
    def case_ids(self) -> list[str]:
        return list(self.table.index)

    def fold_of(self, case_id: str) -> int | None:
        f = self.table.fold.get(case_id) if "fold" in self.table else None
        return None if f is None or pd.isna(f) else int(f)

    def split(self, fold: int) -> tuple[list[str], list[str]]:
        """(training, held-out) case ids of a fold."""
        f = self.table.fold
        return list(self.table.index[f.notna() & (f != fold)]), list(self.table.index[f == fold])

    def case(self, case_id: str) -> Case:
        return Case.load(self.root / case_id, center=self.table.center.get(case_id))

    def __iter__(self):
        return (self.case(c) for c in self.case_ids)


# ── datasets and dataloaders (training the specialists) ─────────────────────────────────────────────────────────────


class CoreFrameDataset:
    """One item per core with an annotated needle track: its fire frame, the track, csPCa, involvement and grade group
    (NeedleNet's training data)."""

    def __init__(self, cohort: Cohort, case_ids, transform=None, require_needle=True):
        self.items = [
            c for cid in case_ids for c in cohort.case(cid).cores if not require_needle or (c.folder / "needle_mask.png").exists()
        ]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        c = self.items[i]
        img = c.frame()
        lab = c.labels or {}
        item = {
            "image": img,
            "needle_mask": c.needle_mask(img.shape),
            "core_id": c.core_id,
            "label": float(c.cspca or 0),
            "involvement": float(lab.get("involvement") or 0.0),
            "grade_group": float(lab.get("grade_group") or 0),
        }
        return self.transform(item) if self.transform else item


def core_frame_loader(cohort: Cohort, case_ids, transform, batch_size=8, shuffle=False, num_workers=4):
    from torch.utils.data import DataLoader

    return DataLoader(
        CoreFrameDataset(cohort, case_ids, transform), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
