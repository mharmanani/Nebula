# Tools

## Specialists (one model per tool; `tools/<name>.py`)

No weights are released. Train or obtain each specialist and set its checkpoint in `configs/paths.yaml`.

| Class | Network (`models/`) | Input | Output | Weights |
|---|---|---|---|---|
| `MicroSegNet` | TransUNet R50-ViT-B/16, 224 px | frame | prostate mask | public (MicroSegNet, Hugging Face `pfrwilson/micro_seg_net`), used zero-shot |
| `NeedleNet` | same network | frame | needle track (fitted rotated rectangle) | train with `scripts/train_needlenet.py` (from the MicroSegNet weights; your annotated tracks, optionally NCT2013) |
| `ProstNFound` | ProstNFound+ (MedSAM ViT-B with adapters, age / PSA prompts), 256 px | frame + age, PSA | cancer-likelihood map | the network is in `models/prostnfound.py`; its training code is not part of this repository (the paper's model was pooled over centres and heatmap-supervised, and was trained on vertically flipped frames, so the tool flips each frame in and the map back) |
| `DINOv3` | DINOv3 ViT-L/16 + prompt-free mask decoder, 384 px | frame | cancer-likelihood map | train with `scripts/train_dinov3_head.py` on NCT2013-format cores (the paper averaged the maps of the five fold models of this head; here one model) |
| `Compass` | frozen ProstNFound+ backbone + BERT over biopsy and sweep tokens with roll embeddings, 224 px | study (sweep + cores + age, PSA) | study risk, per-core scores | train with github.com/mharmanani/Compass (`configs/compass.yaml`: sinusoidal roll embedding, 16 sweep frames, prompts age and PSA) |

Standalone: `python3 run_tool.py --tool <class name, lower case> --modality <frame|cine|sweep|study|all|clinical> --case <dir>`.

| Tool | frame | cine | sweep | study / all |
|---|---|---|---|---|
| microsegnet | yes | yes | yes | |
| needlenet | yes | yes | no (no needle during the sweep) | |
| prostnfound | yes | yes | yes | |
| dinov3 | yes | yes | yes | |
| compass | the cores alone (all, or `--core ID`) | | the sweep alone | sweep + cores (the default, as trained) |

Compass was trained on the sweep and the cores together; on either alone it warns that this was never validated.

## The coordinator's tools (`tools/registry.py`, implemented in `agent/session.py`)

| Tool | Post-biopsy review | Intra-procedural targeting (one core aimed) |
|---|---|---|
| `scan_sweep_targets` | up to five distinct sweep hot spots (≥ 10° apart) ranked by the experience score: TabPFN-3 over previous patients' hot spots labelled through their cores | same |
| `similar_past_cases` | the five most similar labelled hot spots of previous patients | same |
| `locate_core` | — | side, zone and level of the aimed core (Nebula's estimate) and the sweep targets at its angle and depth |
| `place_on_cores` | the core at the same depth within 10° of a target, and the distance from the target to its track after registration | — |
| `open_core` | a core's ProstNFound+ / DINOv3 scores in its track, their ranks, gland and track geometry | — |
| `target_lesion` | the lesion on a core's frame, whether the track crosses it, whether it is shown (core composite risk ≥ threshold) | the lesion on the live frame |
| `record_findings` | one quality-assurance finding | same |
| `assess_gland` | outline quality (first frame of the first core) | prostate in view on the live frame |
| `locate_needle` | needle-track quality (same frame) | — |
| `frame_cancer_map` | ProstNFound+ in each core's track: max (patient score), mean, p90 over cores | live frame: in the prostate, hot spots, along the usual needle path |
| `second_opinion_map` | DINOv3 in each core's track: p90 (patient score), max, mean | same as above, DINOv3 |
| `study_level_risk` | Compass | — |
| `clinical_context` | age, PSA, pseudo PSA density | same |
| `compose` | patient composer (14 inputs) and core composer (3 inputs) | targeting composer (23 inputs) |

The composers' inputs are listed in `composer/features.py`; the patient and core thresholds are fixed on the context (`scripts/build_context.py`); targeting uses `run.py --threshold` (default 0.14, the paper's: about the top 30% of cores).
