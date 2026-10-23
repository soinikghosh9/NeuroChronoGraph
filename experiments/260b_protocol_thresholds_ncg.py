"""Threshold transport under a change of recording condition (exp 260) for NCG-v3, the NeuroChronoGraph version with
the highest development AUC, so Fig. 3c can show it beside SpectraLite. Uses the saved per-subject scores of the
three seeds (no retraining); the result is stored under "thresholds_ncg_v3" and leaves every other entry unchanged.

    python experiments/260b_protocol_thresholds_ncg.py
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


if __name__ == "__main__":
    P = _load("p260", "260_protocol_shift.py")
    C = _load("c211", "211_canonical_benchmark.py")
    f = P.OUTDIR / "protocol_shift.json"
    res = json.loads(f.read_text())
    res["thresholds_ncg_v3"] = P.threshold_transport(C, model="ncg_v3")
    f.write_text(json.dumps(res, indent=1, default=float))
    t = res["thresholds_ncg_v3"]
    print({k: (v.get("spec") if "spec" in v else v.get("spec_mean")) for k, v in t.items() if isinstance(v, dict)})
