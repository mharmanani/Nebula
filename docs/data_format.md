# Data format

This repository ships **no data**. This page describes how to arrange *your own* data so the tools, the composer and the
evaluation scripts can read it (`data/OPTIMUM.py`). All values below are placeholders.

## Cohort

```
<cohort_root>/
├── cohort.csv              # case_id, center[, fold]   (one row per patient)
└── <case_id>/              # one folder per patient
```

The paper's folds are released in `splits/optimum_5fold_patient_splits.json` (patient IDs and fold assignments only): keys
`stratified-kfold_fold-<k>` hold the `train` / `val` patient IDs of fold `k`, and `stratified-kfold_fold-<k>_center-<C>`
the same folds restricted to centre `C`. `splits/excluded_patients.json` lists the four patients of the folds that are not
in the paper's 156-patient evaluation. With your own cohort, give the folds in a `fold` column of `cohort.csv` or in a splits
file of the same format.

## Case folder

```
<case_id>/
├── clinical.json           # {"age": <float>, "psa": <float>}      optional; missing values are imputed by the composer
├── sweep/                  # the rotational sweep; needed by Compass, the sweep tools and the prostate-volume estimate
│   ├── frames/0000.png …   # B-mode frames in acquisition order
│   ├── roll.npy            # probe roll angle in degrees, one value per frame
│   └── meta.json           # {"width_mm": <float>, "depth_mm": <float>}
└── cores/<core_id>/        # one folder per biopsy core, in acquisition order
    ├── frames/0000.png …   # the core's cine (a single frame also works)
    ├── meta.json           # {"fire_frame": <int>, "roll_deg": <float>, "width_mm": <float>, "depth_mm": <float>,
    │                       #  "location": <str, optional>}
    ├── needle_mask.png     # optional: the annotated needle track on the fire frame
    └── labels.json         # training / evaluation only: {"grade_group": <int>, "involvement": <float>, "primus": <int, optional>}
```

| Field | Used by | Meaning |
|---|---|---|
| `fire_frame` | every frame tool | index of the last frame before the needle fires (default: the last frame): the live frame while the core is aimed, the core's frame after the biopsy |
| `roll_deg`, `roll.npy` | sweep tools, Compass, targeting composer | probe roll at the fire frame / per sweep frame, on the same reference; a sweep whose roll is on a reference turned by 180° (range > 200° or values beyond ±120°) is wrapped to (−180°, 180°] for the geometric tools |
| `width_mm`, `depth_mm` | lesion, registration, volume | field of view; pixels need not be square |
| `location` | context only | the operator's recorded location; three-letter codes (side, then level / zone, e.g. `RMM`, `LAL`, `RBM`) calibrate Nebula's own location estimate (`tools/location.py`); never shown to the coordinator |
| `needle_mask.png` | context, training | each centre's standard needle path (mean annotated track) and NeedleNet's training data; never used at inference: core scores are measured inside NeedleNet's track, as in the paper |
| `labels.json` | `build_context.py`, `evaluate.py`, training | never read at inference; csPCa = grade group ≥ 2; the patient label is csPCa in any core; `involvement` = cancer fraction of the core (0–1); `primus` = the operator's PRI-MUS score (optional; shown by `similar_past_cases`) |

## NCT2013-format cores (training the DINOv3 head)

See the docstring of `data/NCT2013.py`: a `metadata.csv` (core_id, patient_id, center, bmode_path, prostate_mask_path,
grade_group, involvement) and one fixed `needle_mask.npy` template.
