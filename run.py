"""Run Nebula on one case.

Usage:
    python3 run.py -m nebula --workflow post  --case /path/to/case                 # post-biopsy review of one patient
    python3 run.py -m nebula --workflow intra --case /path/to/case --core CORE_ID  # one core while it is aimed
    options: --agent qwen | gpt-oss | mistral   (default qwen, the paper's coordinator)
             --thinking     qwen: turn the model's thinking on (off by default, as in the paper)
             --threshold T  intra: biopsy if P(csPCa) >= T (default 0.14, the paper's: about the top 30% of cores)
             --fold K   the patient's fold, so that the context excludes it (found automatically for patients of the context)
             --out FILE  write the episode (tool calls, outputs, composer result, report) as JSON
"""

import argparse
import json
from pathlib import Path

from agent.coordinator import Coordinator
from agent.llm import LocalLLM
from agent.session import Session
from composer.context import Context
from composer.tabpfn import TARGETING_THRESHOLD
from config import load_config
from data.OPTIMUM import Case
from tools.analysis import CaseAnalysis
from tools.registry import build_specialists


def fold_in_context(context_root, case_id):
    import pandas as pd

    p = pd.read_csv(f"{context_root}/patients.csv", dtype={"case_id": str}).set_index("case_id")
    return None if case_id not in p.index else int(p.fold[case_id])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--method", type=str.lower, choices=["nebula"], default="nebula")
    ap.add_argument("--workflow", choices=["post", "intra"], required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--core")
    ap.add_argument("--center")
    ap.add_argument("--agent", choices=["qwen", "gpt-oss", "mistral"], default="qwen")
    ap.add_argument("--thinking", action="store_true", help="qwen: turn the model's thinking on (default off, as in the paper)")
    ap.add_argument("--threshold", type=float, default=TARGETING_THRESHOLD)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.workflow == "intra" and not a.core:
        ap.error("--workflow intra needs --core (the core being aimed)")
    if a.thinking and a.agent != "qwen":
        ap.error("--thinking is only available with --agent qwen")
    cfg = load_config()
    case = Case.load(a.case, center=a.center)
    fold = a.fold if a.fold is not None else fold_in_context(cfg["context"], case.case_id)
    context = Context(cfg["context"], fold)
    session = Session(CaseAnalysis(case, build_specialists(cfg)), context, a.workflow, a.core, a.threshold)
    coordinator = Coordinator(LocalLLM(a.agent, cfg.get("coordinator", {}).get(a.agent), thinking=a.thinking))
    episode = coordinator.run(session)
    if a.out:
        Path(a.out).write_text(json.dumps(episode, indent=1, default=str))
    print(episode["report"] + "\n\n" + episode["footer"])


if __name__ == "__main__":
    main()
