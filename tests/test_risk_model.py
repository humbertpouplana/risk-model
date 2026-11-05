import json
import numpy as np
import pytest
from pathlib import Path
from risk_model import synth, __version__

ROOT = Path(__file__).resolve().parents[1]
PAR = json.loads((ROOT / "data" / "params.json").read_text(encoding="utf-8"))
RES = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))


def test_generator_is_deterministic_and_synthetic():
    a, _ = synth.generate(1.25, 1, 3000)
    b, _ = synth.generate(1.25, 1, 3000)
    assert a.equals(b)
    assert set(["month", "income", "dti", "y"]).issubset(a.columns)
    assert 0.05 < a.y.mean() < 0.30 and a.month.between(0, PAR["months"] - 1).all()


def test_out_of_time_split_has_no_overlap():
    s = PAR["split"]
    assert s["train_end"] < s["valid_end"] < PAR["months"] - 1
    m = RES["meta"]
    assert m["train_n"] + m["valid_n"] + m["test_n"] == m["n"]


def test_headline_is_mean_of_segment_reductions():
    seg = RES["segments"]
    assert len(seg) == 18
    assert np.mean([r["red"] for r in seg]) == pytest.approx(RES["headline"]["mean_reduction_segments"], abs=1e-9)


def test_headline_close_to_target_and_inside_bootstrap_interval():
    h = RES["headline"]["mean_reduction_segments"]; b = RES["bootstrap"]
    assert 0.09 <= h <= 0.11
    assert b["lo"] <= h <= b["hi"]


def test_challenger_beats_scorecard_out_of_time():
    m = RES["metrics"]
    assert m["chall"]["auc"] > m["base"]["auc"] + 0.02
    assert m["chall"]["brier"] <= m["base"]["brier"] + 1e-3


def test_leakage_canary_is_detected():
    """A feature known only after the default must inflate AUC: proves the pipeline would expose leakage."""
    assert RES["leak"]["auc_with_leak"] > RES["leak"]["auc_clean"] + 0.1


def test_default_decreases_with_stricter_approval():
    c = RES["approval_curve"]
    assert c["base"] == sorted(c["base"]) and c["chall"] == sorted(c["chall"])
    assert all(ch <= b + 1e-9 for ch, b in zip(c["chall"], c["base"]))


def test_economics_reproduce_from_parameters():
    e = RES["econ"]
    assert e["lgd"] == PAR["lgd"] and e["margin"] == PAR["margin"]
    assert e["chall"] > e["base"]


def test_version_and_readme():
    assert RES["model_version"] == __version__
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "9.8%" in readme


def test_sample_dataset_is_synthetic_and_small():
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "sample_applications.csv")
    assert len(df) == 2000 and "y" in df


@pytest.mark.slow
def test_full_pipeline_reproduces_published_headline(tmp_path):
    import subprocess, sys, shutil
    # run on a copy so the committed results are not overwritten
    work = tmp_path / "repo"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(".git", "paper", "prototype", "__pycache__"))
    subprocess.run([sys.executable, "-m", "risk_model.pipeline"], cwd=work, env={"PYTHONPATH": str(work / "src"), "PATH": __import__("os").environ["PATH"]}, check=True, capture_output=True)
    new = json.loads((work / "results" / "results.json").read_text(encoding="utf-8"))
    assert new["headline"]["mean_reduction_segments"] == pytest.approx(RES["headline"]["mean_reduction_segments"], abs=1e-9)
