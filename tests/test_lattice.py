"""Tests for the two-layer Seed-and-Extend + 32-keyword-family lattice module."""

from __future__ import annotations

import json

import numpy as np

from structural_fingerprint.cli import main as cli_main
from structural_fingerprint.lattice import (
    LanguageDeltaModel,
    StyleAxis,
    fit_language_deltas,
    fit_style_axis,
    keyword_lattice_distance,
    keyword_lattice_profile,
    seed_and_extend_with_lattice,
)
from structural_fingerprint.reference import load_reference
from structural_fingerprint.scan_io import load_scan


def test_keyword_lattice_profile_and_distance():
    sigs_a = np.array([[10, 0, 0], [8, 2, 0]], dtype=np.float64)
    sigs_b = np.array([[0, 0, 10], [0, 2, 8]], dtype=np.float64)
    norm_a, dom_a, dist_a = keyword_lattice_profile(sigs_a)
    norm_b, dom_b, dist_b = keyword_lattice_profile(sigs_b)

    assert np.allclose(np.linalg.norm(norm_a, axis=1), 1.0)
    assert dom_a.tolist() == [0, 0]
    assert dom_b.tolist() == [2, 2]
    assert np.isclose(dist_a.sum(), 1.0)
    assert np.isclose(keyword_lattice_distance(dist_a, dist_a), 0.0, atol=1e-9)
    assert keyword_lattice_distance(dist_a, dist_b) > 0.8


def test_seed_and_extend_with_lattice_separates_fork_from_unrelated(cohort):
    ref = load_reference("portable")
    sc_base = load_scan(cohort["base"], ref.signal_columns)
    sc_fork = load_scan(cohort["fork"], ref.signal_columns)
    sc_other = load_scan(cohort["other"], ref.signal_columns)

    same = seed_and_extend_with_lattice(sc_base, sc_base, ref, outer=15)
    fork = seed_and_extend_with_lattice(sc_base, sc_fork, ref, outer=15)
    other = seed_and_extend_with_lattice(sc_base, sc_other, ref, outer=15)

    assert np.isclose(same.kw_jsd, 0.0, atol=1e-9)
    assert same.matched_kw_cos > 0.99
    assert fork.fgw_cost < other.fgw_cost or fork.kw_jsd < other.kw_jsd
    assert fork.kw_jsd < other.kw_jsd
    assert fork.matched_kw_cos > other.matched_kw_cos


def test_cli_lattice_compare_json(cohort, capsys):
    rc = cli_main([
        "lattice-compare",
        str(cohort["base"]),
        str(cohort["fork"]),
        "--lens", "portable",
        "--matches",
        "--json",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["repo_a"] == "base"
    assert payload["repo_b"] == "fork"
    assert 0.0 <= payload["kw_jsd"] <= 1.0
    assert payload["se_n"] >= 3
    assert len(payload["matches"]) == payload["se_n"]
    assert all(m["tier"] >= 0 for m in payload["matches"])


def test_language_delta_model_and_style_axis():
    # Two languages ("c" and "py") shifted by a pure additive language offset,
    # plus a subtle author-style offset (+pos vs -neg) on dims 0 vs 1.
    c_pos = np.array([0.65, 0.25, 0.10])
    c_neg = np.array([0.45, 0.45, 0.10])
    py_pos = np.array([0.25, 0.25, 0.50])
    py_neg = np.array([0.05, 0.45, 0.50])

    model = fit_language_deltas({"c": [c_pos, c_neg], "py": [py_pos, py_neg]})
    restored = LanguageDeltaModel.from_dict(model.to_dict())

    # Centering removes the language offset so same-style cross-language pairs align closely
    assert restored.centered_distance(c_pos, "c", py_pos, "py") < 0.05
    assert restored.centered_distance(c_pos, "c", py_neg, "py") > 0.15

    # Translation prediction C -> py recovers py_pos from c_pos
    pred_py = restored.predict_translation(c_pos, "c", "py")
    assert keyword_lattice_distance(pred_py, py_pos) < 0.05

    # Style axis trained on centered C runs transfers zero-shot to centered Python runs
    axis = fit_style_axis(
        [restored.center(c_pos, "c")],
        [restored.center(c_neg, "c")],
        pos_label="claude",
        neg_label="codex",
    )
    axis_rt = StyleAxis.from_dict(axis.to_dict())
    assert axis_rt.score(restored.center(py_pos, "py")) > axis_rt.score(restored.center(py_neg, "py"))


def test_resolve_or_scan_auto_scans_source_dir_and_errors_without_gitgalaxy(cohort, tmp_path, monkeypatch):
    import importlib.util
    import subprocess
    from pathlib import Path
    import pytest
    from structural_fingerprint import scan_io

    # 1. Existing scan DB passes straight through without scanning
    existing = scan_io.resolve_or_scan([cohort["base"]])
    assert len(existing) == 1 and existing[0].name.endswith(scan_io.DB_SUFFIX)

    # 2. Raw source dir without gitgalaxy raises a clear ScanError
    raw_dir = tmp_path / "raw_repo"
    raw_dir.mkdir()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "gitgalaxy" else object())
    with pytest.raises(scan_io.ScanError, match="structural-fingerprint-scanner\\[scan\\]"):
        scan_io.resolve_or_scan([raw_dir], cache_dir=tmp_path / "cache")

    # 3. Raw source dir with gitgalaxy invokes galaxyscope and returns produced DB
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    def fake_run(cmd, capture_output, text, check):
        out_arg = Path(cmd[cmd.index("--output") + 1])
        out_arg.mkdir(parents=True, exist_ok=True)
        (out_arg / "raw_repo_galaxy_master.db").write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    scanned = scan_io.resolve_or_scan([raw_dir], cache_dir=tmp_path / "cache")
    assert len(scanned) == 1
    assert scanned[0].name == "raw_repo_galaxy_master.db"


def test_matched_rows_are_scan_rows_when_short_functions_are_dropped(tmp_path):
    """Functions under min_loc are dropped before matching; the features, keyword profiles and
    reported rows must still refer to the right scan rows (they were shifted onto the wrong ones)."""
    from .conftest import random_functions, write_scan

    ref = load_reference("portable")
    rng = np.random.default_rng(11)
    code = random_functions(120, rng, "fn")
    short = [{**f, "name": f"tiny_{i}", "loc": 1, "complexity": 1} for i, f in enumerate(random_functions(40, rng, "t"))]
    mixed = []
    for i, f in enumerate(code):
        if i % 3 == 0:
            mixed.append(short[i // 3])
        mixed.append(f)
    sc_mixed = load_scan(write_scan(tmp_path, "mixed", mixed), ref.signal_columns)
    sc_clean = load_scan(write_scan(tmp_path, "clean", code), ref.signal_columns)

    res = seed_and_extend_with_lattice(sc_mixed, sc_clean, ref, outer=15, include_matches=True)

    assert res.matches
    assert all(sc_mixed.loc[m.row_a] >= 3 for m in res.matches)
    agree = sum(sc_mixed.func_name[m.row_a] == sc_clean.func_name[m.row_b] for m in res.matches)
    assert agree / len(res.matches) > 0.95
    assert res.matched_kw_cos > 0.99
