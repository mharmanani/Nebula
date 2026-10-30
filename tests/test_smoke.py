"""Smoke tests on synthetic inputs (random-noise frames, randomly initialised weights, a random context): they check that
every piece runs end to end, not that anything is accurate. CPU is enough; TabPFN-3's weights must be downloadable or
cached. Needs the `dinov3` package.

    python3 -m pytest tests -q
"""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.coordinator import Coordinator  # noqa: E402
from agent.llm import parse_gpt_oss, parse_mistral, parse_qwen  # noqa: E402
from agent.session import Session  # noqa: E402
from agent.wording import risk_statement, unsupported_numbers  # noqa: E402
from composer.context import Context  # noqa: E402
from composer.features import CORE_FEATURES, PATIENT_FEATURES, TARGETING_FEATURES  # noqa: E402
from data.OPTIMUM import Case  # noqa: E402
from tools.analysis import CaseAnalysis  # noqa: E402
from tools.registry import build_specialists  # noqa: E402
from tools.sweep import FEATURES as SPOT_FEATURES  # noqa: E402

RNG = np.random.default_rng(0)


class ProtocolLLM:
    """Stands in for the coordinator model in tests: calls the protocol's tools in order (in Qwen's tool-call format), then
    writes a fixed report."""

    ORDER = {
        "post": [
            ("assess_gland", {}),
            ("locate_needle", {}),
            ("frame_cancer_map", {}),
            ("second_opinion_map", {}),
            ("study_level_risk", {}),
            ("clinical_context", {}),
            ("scan_sweep_targets", {}),
            ("place_on_cores", {"target": "T1"}),
            ("open_core", {}),
            ("target_lesion", {}),
            ("record_findings", {"location": "T1", "core": "see place_on_cores", "justification": "test"}),
            ("compose", {}),
        ],
        "intra": [
            ("scan_sweep_targets", {}),
            ("locate_core", {}),
            ("assess_gland", {}),
            ("frame_cancer_map", {}),
            ("second_opinion_map", {}),
            ("target_lesion", {}),
            ("compose", {}),
        ],
    }
    parse = staticmethod(parse_qwen)

    def __init__(self, workflow: str):
        self.calls = iter(self.ORDER[workflow])

    def generate(self, messages, tools=None, max_new_tokens=500) -> str:
        if tools is None:
            return "FINDINGS: test report."
        name, args = next(self.calls)
        params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items())
        return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"

    def assistant_turn(self, text, call) -> dict:
        return {"role": "assistant", "content": text}

    def tool_turn(self, name, content, assistant) -> dict:
        return {"role": "tool", "name": name, "content": content}


def noise_frame(h=112, w=160):
    yy, xx = np.mgrid[:h, :w]
    gland = ((yy - h * 0.55) / (h * 0.3)) ** 2 + ((xx - w / 2) / (w * 0.3)) ** 2 <= 1
    return np.clip(RNG.normal(90, 25, (h, w)) + 60 * gland, 0, 255).astype(np.uint8)


def make_case(root: Path) -> Path:
    case = root / "cohort" / "XX-001"
    for k in range(8):
        (case / "sweep" / "frames").mkdir(parents=True, exist_ok=True)
        Image.fromarray(noise_frame()).save(case / "sweep" / "frames" / f"{k:04d}.png")
    np.save(case / "sweep" / "roll.npy", np.linspace(-40, 40, 8))
    (case / "sweep" / "meta.json").write_text(json.dumps({"width_mm": 46.0, "depth_mm": 28.0}))
    for i, roll in enumerate((-20.0, 15.0)):
        c = case / "cores" / f"XX-001-00{i + 1}"
        (c / "frames").mkdir(parents=True)
        for k in range(3):
            Image.fromarray(noise_frame()).save(c / "frames" / f"{k:04d}.png")
        (c / "meta.json").write_text(
            json.dumps(
                {"fire_frame": 2, "roll_deg": roll, "width_mm": 46.0, "depth_mm": 28.0, "location": "LMM" if roll < 0 else "RML"}
            )
        )
        if i == 0:  # an annotated track (training / the standard path only; inference uses NeedleNet)
            m = np.zeros((112, 160), np.uint8)
            m[40:46, 30:130] = 255
            Image.fromarray(m).save(c / "needle_mask.png")
    (case / "clinical.json").write_text(json.dumps({"age": 64, "psa": 7.1}))
    (root / "cohort" / "cohort.csv").write_text("case_id,center,fold\nXX-001,XX,0\n")
    return case


def make_weights(root: Path) -> dict:
    from models.compass import build_compass
    from models.dinov3_head import build_dinov3_head
    from models.microsegnet import MicroSegNetwork
    from models.prostnfound import build_prostnfound

    nets = {
        "microsegnet": lambda: MicroSegNetwork().net,
        "needlenet": lambda: MicroSegNetwork().net,
        "prostnfound": build_prostnfound,
        "dinov3": lambda: build_dinov3_head(pretrained_backbone=False),
        "compass": build_compass,
    }
    paths = {}
    for name, build in nets.items():
        paths[name] = str(root / f"{name}.pth")
        torch.save(build().state_dict(), paths[name])
    return paths


def make_context(root: Path) -> Path:
    ctx = root / "context"
    ctx.mkdir()
    n_p, n_c, n_s = 30, 60, 80
    P = pd.DataFrame(RNG.random((n_p, len(PATIENT_FEATURES))), columns=PATIENT_FEATURES).assign(
        case_id=[f"P{i}" for i in range(n_p)], center="XX", fold=np.arange(n_p) % 5, label=np.arange(n_p) % 2
    )
    cols = list(dict.fromkeys(CORE_FEATURES + TARGETING_FEATURES))
    C = pd.DataFrame(RNG.random((n_c, len(cols))), columns=cols).assign(
        core_id=[f"C{i}" for i in range(n_c)],
        case_id=[f"P{i % n_p}" for i in range(n_c)],
        center="XX",
        fold=np.arange(n_c) % 5,
        label=np.arange(n_c) % 2,
        primus=RNG.integers(1, 6, n_c),
        angle=RNG.uniform(-40, 40, n_c),
        depth_mm=28.0,
        location=RNG.choice(["LMM", "RML", "LAM", "RBL"], n_c),
        reg_shift_mm=RNG.normal(0, 5, n_c),
    )
    H = pd.DataFrame(RNG.random((n_s, len(SPOT_FEATURES))), columns=SPOT_FEATURES).assign(
        case_id="P0", fold=np.arange(n_s) % 5, label=np.arange(n_s) % 2, primus=3
    )
    P.to_csv(ctx / "patients.csv", index=False)
    C.to_csv(ctx / "cores.csv", index=False)
    H.to_csv(ctx / "hotspots.csv", index=False)
    th = {"patient": 0.5, "core": 0.5}
    (ctx / "calibration.json").write_text(
        json.dumps(
            {"sweep_z": {"pnf": [0.5, 0.1], "dino": [0.5, 0.1]}, "thresholds": {k: th for k in ["0", "1", "2", "3", "4", "all"]}}
        )
    )
    np.savez(
        ctx / "guide_paths.npz", **{f"{k}_XX": RNG.random((28, 40)).astype(np.float32) for k in ["0", "1", "2", "3", "4", "all"]}
    )
    return ctx


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("nebula")
    cfg = {
        "device": "cpu",
        "weights": make_weights(root),
        "cohort": str(root / "cohort"),
        "context": str(make_context(root)),
    }
    case_dir = make_case(root)
    (root / "paths.yaml").write_text(yaml.safe_dump({k: cfg[k] for k in ("weights", "cohort", "context")}))
    (root / "nebula.yaml").write_text(yaml.safe_dump({"device": cfg["device"]}))
    return {"cfg": cfg, "case": case_dir, "root": root, "specialists": build_specialists(cfg)}


def test_specialists(setup):
    case = Case.load(setup["case"])
    sp = setup["specialists"]
    f = case.cores[0].frame()
    assert sp["microsegnet"].segment(f).shape == f.shape and sp["needlenet"].segment(f).dtype == bool
    for m in (sp["prostnfound"].heatmap(f, case.age, case.psa), sp["dinov3"].heatmap(f)):
        assert m.shape == f.shape and 0 <= m.min() <= m.max() <= 1
    out = sp["compass"].predict(case)
    assert 0 <= out["patient_risk"] <= 1 and set(out["core_scores"]) == {c.core_id for c in case.cores}
    assert 0 <= sp["compass"].predict(replace(case, cores=[]))["patient_risk"] <= 1  # the sweep alone
    assert set(sp["compass"].predict(replace(case, sweep=None))["core_scores"]) == {c.core_id for c in case.cores}  # cores alone


@pytest.mark.parametrize("workflow", ["post", "intra"])
def test_episode(setup, workflow):
    case = Case.load(setup["case"])
    session = Session(
        CaseAnalysis(case, setup["specialists"]),
        Context(setup["cfg"]["context"], fold=0),
        workflow,
        case.cores[0].core_id if workflow == "intra" else None,
    )
    episode = Coordinator(ProtocolLLM(workflow)).run(session)
    assert episode["calls"] == [name for name, _ in ProtocolLLM.ORDER[workflow]] and not episode["forced"]
    assert episode["final"] is not None and "risk_statement" in episode["final"]
    assert all("result" in t for t in episode["trace"])


def test_run_tool_cli(setup):
    env = {**os.environ, "NEBULA_PATHS": str(setup["root"] / "paths.yaml"), "NEBULA_CONFIG": str(setup["root"] / "nebula.yaml")}
    run = lambda *a: subprocess.run(
        [sys.executable, str(ROOT / "run_tool.py"), *a, "--case", str(setup["case"])], env=env, capture_output=True, text=True
    )
    r = run("--tool", "prostnfound", "--modality", "frame")
    assert r.returncode == 0 and "max" in r.stdout, r.stderr
    assert run("--tool", "needlenet", "--modality", "sweep").returncode != 0  # no needle on the sweep


def test_parsers_and_wording():
    assert parse_qwen("<tool_call>\n<function=open_core>\n<parameter=core>\ncore 2\n</parameter>\n</function>\n</tool_call>")[
        1
    ] == {"name": "open_core", "arguments": {"core": "core 2"}}
    assert (
        parse_gpt_oss("<|channel|>commentary to=functions.compose <|constrain|>json<|message|>{}<|call|>")[1]["name"] == "compose"
    )
    assert parse_mistral('[TOOL_CALLS]place_on_cores[ARGS]{"target": "T1"}')[1] == {
        "name": "place_on_cores",
        "arguments": {"target": "T1"},
    }
    assert risk_statement(0.95, 0.3).startswith("Clinically significant cancer at this core is most likely")
    assert unsupported_numbers("risk 0.42 and 77", "probability 0.4213") == ["77"]
