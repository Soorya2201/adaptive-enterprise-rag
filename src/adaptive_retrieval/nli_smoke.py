"""Run actual pinned-model NLI sanity checks (not benchmark performance)."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .nli import NLI_MODEL, NLI_REVISION, NliJudge

CASES = [
    ("The runbook is due in five business days.", "The runbook is due in five business days.", "entailment"),
    ("Only administrators can rotate keys.", "Any employee can rotate keys.", "contradiction"),
    ("The runbook is due in five business days.", "The office is in Paris.", "neutral"),
    ("Can the supplier deliver by Monday?", "The supplier promised delivery by Monday.", "neutral"),
    (("Unrelated infrastructure notes. " * 180) + " The runbook is due in five business days.",
     "The runbook is due in five business days.", "entailment"),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/outputs/nli_smoke.json"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    judge = NliJudge()
    judge.load(download=args.download)
    results = [{"premise": premise, "hypothesis": hypothesis, "expected": expected,
                **judge.assess(premise, hypothesis)} for premise, hypothesis, expected in CASES]
    passed = sum(item["label"] == item["expected"] for item in results)
    result = {"purpose": "Synthetic sanity checks only; not benchmark performance or threshold calibration",
              "model": NLI_MODEL, "revision": NLI_REVISION, "threshold": judge.threshold,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "torch_version": torch.__version__, "passed": passed, "cases": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"NLI smoke: {passed}/{len(CASES)} passed; {args.out}")
    if passed != len(CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
