"""Tests for scan quality assessment and GitGalaxy Epic issue reporting."""

from __future__ import annotations

import json
import shutil

import numpy as np

from structural_fingerprint.cli import main as cli_main
from structural_fingerprint.quality import assess_quality
from structural_fingerprint.reference import load_reference
from structural_fingerprint.scan_io import ScanData, load_scan


def _make_scan(
    *,
    name: str = "demo",
    file_path: list[str],
    file_language: list[str],
    file_coding_loc: list[int],
    func_file: list[int],
    loc: list[float],
    signals: np.ndarray,
    call_edges: np.ndarray,
    missing_signals: tuple[str, ...] = (),
) -> ScanData:
    n = len(loc)
    return ScanData(
        name=name,
        db_path=None,  # type: ignore[arg-type]
        metadata={},
        func_id=np.arange(1, n + 1, dtype=np.int64),
        func_name=[f"fn_{i}" for i in range(n)],
        func_file=np.asarray(func_file, dtype=np.int64),
        loc=np.asarray(loc, dtype=np.float64),
        complexity=np.ones(n, dtype=np.float64),
        args=np.ones(n, dtype=np.float64),
        keyword_density=np.full(n, 0.1, dtype=np.float64),
        fan_in=np.zeros(n, dtype=np.float64),
        fan_out=np.zeros(n, dtype=np.float64),
        engine_archetype=["Compute Core"] * n,
        signals=signals,
        signal_columns=tuple(f"sig_{i}" for i in range(signals.shape[1])),
        missing_signals=missing_signals,
        file_path=file_path,
        file_language=file_language,
        file_coding_loc=np.asarray(file_coding_loc, dtype=np.int64),
        file_archetype=["logic"] * len(file_path),
        call_edges=call_edges,
    )


def test_quality_healthy_cohort_scan(cohort):
    ref = load_reference("portable")
    sc = load_scan(cohort["base"], ref.signal_columns)
    rep = assess_quality(sc)
    assert rep.overall_status == "ok"
    assert rep.accurate_numbers is True
    assert rep.tools_used == ("internal",)


def test_quality_detects_zero_signal_and_emits_epic_issue():
    sigs = np.zeros((6, 4), dtype=np.float64)
    sc = _make_scan(
        name="broken-lang",
        file_path=["src/a.xyz", "src/b.xyz"],
        file_language=["xyz", "xyz"],
        file_coding_loc=[80, 60],
        func_file=[0, 0, 0, 0, 0, 0],
        loc=[12.0, 10.0, 14.0, 9.0, 11.0, 8.0],
        signals=sigs,
        call_edges=np.zeros((0, 2), dtype=np.int64),
    )
    rep = assess_quality(sc)
    assert rep.overall_status == "fail"
    assert rep.accurate_numbers is False
    lq = rep.languages[0]
    assert lq.language == "xyz"
    assert lq.zero_function_files == 1
    assert lq.zero_signal_rate == 1.0

    issue = rep.to_epic_issue(privacy="standard")
    assert "# [Epic] GitGalaxy lexical & slicer tuning for `broken-lang` (xyz)" in issue
    assert "0 keyword-signal hits" in issue
    assert "`src/b.xyz`" in issue

    paranoid_issue = rep.to_epic_issue(privacy="paranoid")
    assert "[REDACTED_REPO]" in paranoid_issue
    assert "broken-lang" not in paranoid_issue
    assert "src/a.xyz" not in paranoid_issue
    assert "src/b.xyz" not in paranoid_issue
    assert "fn_0" not in paranoid_issue
    assert "`.xyz`" in paranoid_issue

    triage = rep.to_llm_triage_prompt()
    assert "expected false positives" in triage
    assert "TRUE_GG_MISS" in triage
    assert "zero_signal_function" in triage


def test_quality_verify_with_ctags_and_treesitter(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    c_file = src / "main.c"
    c_file.write_text(
        "int f1(int x) { if (x) return x + 1; return 0; }\n"
        "int f2(int x) { if (x) return x + 2; return 0; }\n"
        "int f3(int x) { if (x) return x + 3; return 0; }\n"
        "int f4(int x) { if (x) return x + 4; return 0; }\n"
        "int f5(int x) { if (x) return x + 5; return 0; }\n"
    )
    # Pretend GitGalaxy only extracted 1 of the 5 functions
    sigs = np.array([[2.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    sc = _make_scan(
        name="missed-c",
        file_path=["main.c"],
        file_language=["c"],
        file_coding_loc=[5],
        func_file=[0],
        loc=[5.0],
        signals=sigs,
        call_edges=np.zeros((0, 2), dtype=np.int64),
    )
    rep = assess_quality(sc, source_dir=src, verify=True)
    lq = rep.languages[0]
    if shutil.which("ctags"):
        assert "ctags" in rep.tools_used
        assert lq.ctags_functions == 5
        assert lq.ctags_recall == 0.2
        assert rep.overall_status == "fail"
        assert rep.accurate_numbers is False
        assert any(c.kind == "missed_by_gitgalaxy" and c.source_tool == "ctags" for c in rep.disagreements)
    try:
        import tree_sitter_language_pack  # noqa: F401
        assert "tree-sitter" in rep.tools_used
        assert lq.treesitter_functions == 5
        assert lq.treesitter_recall == 0.2
    except ImportError:
        pass


def test_cli_quality_json_and_issue_out(cohort, tmp_path, capsys):
    issue_path = tmp_path / "epic.md"
    rc = cli_main([
        "quality",
        str(cohort["base"]),
        "--lens", "portable",
        "--issue-out", str(issue_path),
        "--json",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["repo"] == "base"
    assert payload["overall_status"] == "ok"
    assert issue_path.is_file()
    assert "Quality OK" in issue_path.read_text()
