"""Patient-level cross-validated evaluation.

    python3 scripts/evaluate.py --context /path/to/context                 # the three composers, out of fold, from the context
    python3 scripts/evaluate.py --context /path/to/context --episodes runs.jsonl   # Nebula episodes written by run.py --out

Per fold (each fold's patients scored with the context of the other folds): AUROC, balanced accuracy at the composer's
threshold, and sensitivity at 60% specificity; reported as mean +- sd over folds. Patient label: csPCa in any core; core
label: csPCa in the core.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from composer.context import Context
from composer.features import CORE_FEATURES, PATIENT_FEATURES, TARGETING_FEATURES
from composer.tabpfn import TARGETING_THRESHOLD


def metrics(y, p, t) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    yes = p >= t
    return {
        "AUROC": roc_auc_score(y, p),
        "balanced accuracy": (yes[y == 1].mean() + (~yes)[y == 0].mean()) / 2,
        "sensitivity at 60% specificity": float(np.interp(0.4, fpr, tpr)),
    }


def summary(per_fold: list[dict]) -> str:
    return ", ".join(
        f"{k} {100 * np.mean([m[k] for m in per_fold]):.1f} ± {100 * np.std([m[k] for m in per_fold]):.1f}" for k in per_fold[0]
    )


def composers(root):
    patients = pd.read_csv(Path(root) / "patients.csv", dtype={"case_id": str})
    cores = pd.read_csv(Path(root) / "cores.csv", dtype={"case_id": str, "core_id": str}).dropna(subset=["label"])
    tasks = {
        "post-biopsy, patient (patient composer)": (patients, PATIENT_FEATURES, "patient_composer", "patient"),
        "post-biopsy, core (core composer)": (cores, CORE_FEATURES, "core_composer", "core"),
        "intra-procedural, core (targeting composer)": (cores, TARGETING_FEATURES, "targeting_composer", "targeting"),
    }
    for name, (table, keys, composer, threshold) in tasks.items():
        per_fold = []
        for f in sorted(table.fold.unique()):
            ctx = Context(root, int(f))
            held = table[table.fold == f]
            p = getattr(ctx, composer).proba(held[keys].values.astype(float))
            t = TARGETING_THRESHOLD if threshold == "targeting" else ctx.thresholds[threshold]
            per_fold.append(metrics(held.label.values.astype(int), p, t))
        print(f"{name}: {summary(per_fold)}")


def episodes(root, path):
    """Episodes of one workflow: risk = the composer's output in the episode; folds and labels from the context tables."""
    E = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if E[0]["workflow"] == "post":
        t = pd.read_csv(Path(root) / "patients.csv", dtype={"case_id": str}).set_index("case_id")
        rows = [
            (t.fold[e["case"]], t.label[e["case"]], e["final"]["risk"], e["final"]["threshold"])
            for e in E
            if e["case"] in t.index
        ]
    else:
        t = pd.read_csv(Path(root) / "cores.csv", dtype={"core_id": str}).set_index("core_id")
        rows = [
            (t.fold[e["core"]], t.label[e["core"]], e["final"]["probability"], e["final"]["threshold"])
            for e in E
            if e["core"] in t.index
        ]
    D = pd.DataFrame(rows, columns=["fold", "y", "p", "t"])
    per_fold = [metrics(g.y.values.astype(int), g.p.values, g.t.values[0]) for _, g in D.groupby("fold")]
    print(f"Nebula episodes ({E[0]['workflow']}, {len(D)}): {summary(per_fold)}; forced compose in {sum(e['forced'] for e in E)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", required=True)
    ap.add_argument("--episodes")
    a = ap.parse_args()
    if a.episodes:
        episodes(a.context, a.episodes)
    else:
        composers(a.context)


if __name__ == "__main__":
    main()
