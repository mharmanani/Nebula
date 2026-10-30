# Third-party code

Network code in `models/` is copied from the projects below, trimmed to the configurations Nebula uses. Each keeps its
original license; the license texts sit next to the code.

| Code in this repository | Source | License |
|---|---|---|
| `models/segment_anything/`; the decoder blocks in `models/dinov3_head.py` | [Segment Anything](https://github.com/facebookresearch/segment-anything) (Meta Platforms), as used by [MedSAM](https://github.com/bowang-lab/MedSAM) | Apache-2.0 (`models/segment_anything/LICENSE`) |
| `models/microsegnet/` | [MicroSegNet](https://github.com/mirthAI/MicroSegNet), adapted from [TransUNet](https://github.com/Beckschen/TransUNet) | MIT (`models/microsegnet/LICENSE`); TransUNet parts Apache-2.0 (`models/microsegnet/LICENSE.TransUNet`) |
| `models/compass.py`, `models/bert_mil.py` | [Compass](https://github.com/mharmanani/Compass) | MIT |

ProstNFound+ (`models/prostnfound.py`, `models/medsam_adapter.py`) and the DINOv3 cancer head (`models/dinov3_head.py`)
are our own implementations; they build on the Segment Anything code above.

Installed as dependencies, not copied: [DINOv3](https://github.com/facebookresearch/dinov3) (DINOv3 License), the
[TabPFN](https://github.com/PriorLabs/TabPFN) package and the coordinator language models, each under its own license.
