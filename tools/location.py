"""Nebula's estimate of where a core is, without the operator's record: side from the sign of the probe roll, medial or
lateral from |roll|, and apex / mid / base from the registration shift between the core's frame and the sweep.

Thresholds come from the context's cores whose recorded location is a three-letter code (side, then level/zone letters,
e.g. RMM, LAL, RBM): the zone threshold is the midpoint of the median |roll| of medial and of lateral cores (the median |roll|
of all cores without such codes), and each level is the median registration shift of its cores; a core gets the level whose
median shift is nearest (unknown without codes).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

LEVEL_NAMES = {"A": "apex", "M": "mid", "B": "base"}


def parse_location(code) -> tuple[str | None, str | None]:
    """Three-letter location code -> (level A/M/B, zone M/L); (None, None) if not such a code."""
    m = re.fullmatch(r"([LR])([ABLM])([ABLM])", str(code).upper().strip())
    if not m:
        return None, None
    a, b = m.group(2), m.group(3)
    if a in "AB":
        return a, (b if b in "ML" else None)
    if a == "L":
        return (b if b in "AMB" else None), "L"
    return (b, "M") if b in "AB" else ("M", b)


class CoreLocator:
    def __init__(self, context_cores: pd.DataFrame):
        """context_cores: columns angle, location, reg_shift_mm (the context's cores)."""
        lv = context_cores.location.map(parse_location)
        cores = context_cores.assign(level=[level for level, _ in lv], zone=[z for _, z in lv], absang=context_cores.angle.abs())
        med = cores[cores.zone.isin(["M", "L"])].groupby("zone").absang.median()
        self.zone_threshold = float((med["M"] + med["L"]) / 2) if {"M", "L"} <= set(med.index) else float(cores.absang.median())
        self.level_shift = (
            cores[cores.level.isin(list("AMB")) & cores.reg_shift_mm.notna()].groupby("level").reg_shift_mm.median()
        )

    def zone(self, angle: float) -> str:
        return "lateral" if abs(angle) >= self.zone_threshold else "medial"

    def level(self, reg_shift_mm: float) -> str:
        if reg_shift_mm != reg_shift_mm or self.level_shift.empty:
            return "unknown"
        return LEVEL_NAMES[self.level_shift.index[int(np.argmin(np.abs(self.level_shift.values - reg_shift_mm)))]]

    def describe(self, angle: float, reg_shift_mm: float) -> str:
        level = self.level(reg_shift_mm)
        words = " ".join(
            x for x in ("left" if angle < 0 else "right", None if level == "unknown" else level, self.zone(angle)) if x
        )
        if level == "unknown":
            words += " (apex/mid/base unknown: no sweep frame at this angle to register against)"
        return f"{words} (Nebula's estimate)"
