from __future__ import annotations

import os
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from structural_fingerprint.cli import main as cli_main
from structural_fingerprint.ledger import Snapshot
from structural_fingerprint.port_evidence import (
    build_port_evidence,
    card_to_dict,
    extract_function_comments,
    format_evidence_card,
    isotonic_nonincreasing,
    same_name,
    suggest_checkpoints,
)

from .conftest import SIGNALS, write_scan


def _git(repo: Path, *args: str) -> str:
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t",
    )
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True, env=env,
    ).stdout.strip()


def test_isotonic_nonincreasing_pava():
    assert isotonic_nonincreasing([0.8, 0.7, 0.5]) == pytest.approx([0.8, 0.7, 0.5])
    assert isotonic_nonincreasing([0.8, 0.6, 0.7, 0.4]) == pytest.approx([0.8, 0.65, 0.65, 0.4])
    assert isotonic_nonincreasing([0.2, 0.5, 0.8]) == pytest.approx([0.5, 0.5, 0.5])
    assert isotonic_nonincreasing([]) == []


def test_same_name_helper():
    assert same_name("P_GiveAmmo", "p_GiveAmmo")
    assert same_name("DoLockedDoor", "EV_DoLockedDoor")
    assert not same_name("foo", "bar")


def test_suggest_checkpoints_picks_birth_churn_and_head(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    for i in range(6):
        if i == 3:
            for k in range(5):
                (repo / f"m{k}.py").write_text(f"def f{k}():\n    return {k}\n")
        else:
            (repo / "main.py").write_text(f"def main():\n    return {i}\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", f"c{i}")

    cps = suggest_checkpoints(repo, max_snapshots=3)
    assert len(cps) == 3
    shas = [c[0] for c in cps]
    # First and last commit included, plus high-churn commit #3
    c0 = _git(repo, "rev-list", "--reverse", "HEAD").splitlines()[0][:len(shas[0])]
    c3 = _git(repo, "rev-list", "--reverse", "HEAD").splitlines()[3][:len(shas[1])]
    c_last = _git(repo, "rev-parse", "--short", "HEAD")
    assert shas[0] == c0
    assert shas[1] == c3
    assert shas[2].startswith(c_last[:6])


def _make_synth_funcs(rng: np.random.Generator, n: int, prefix: str):
    rows = []
    for i in range(n):
        sigs = {s: int(rng.integers(0, 8)) for s in SIGNALS}
        rows.append({
            "path": "a.c",
            "name": f"{prefix}{i}",
            "loc": int(rng.integers(10, 40)),
            "complexity": int(rng.integers(1, 10)),
            "args": int(rng.integers(0, 4)),
            "keyword_density": float(rng.uniform(0.05, 0.25)),
            "signals": sigs,
        })
    edges = [(i + 1, ((i + 1) % n) + 1) for i in range(n)]
    return rows, edges


def test_build_port_evidence_end_to_end(tmp_path: Path):
    rng = np.random.default_rng(7)
    origin_rows, origin_edges = _make_synth_funcs(rng, 18, "fn_")
    origin_db = write_scan(tmp_path, "origin", origin_rows, edges=origin_edges)

    repo = tmp_path / "port"
    repo.mkdir()
    _git(repo, "init")

    # Commit 0: close structural copy of origin
    (repo / "a.c").write_text("\n".join(
        f"// unique comment token_{i} alpha_{i} beta_{i}\nint fn_{i}(void) {{ return {i}; }}\n"
        for i in range(18)
    ))
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "t0")
    sha0 = _git(repo, "rev-parse", "HEAD")
    db0 = write_scan(tmp_path, "t0", origin_rows, edges=origin_edges)

    # Commit 1: slightly diverged
    port_rows_1 = [dict(row, loc=row["loc"] + 2) for row in origin_rows]
    (repo / "a.c").write_text("\n".join(
        f"// unique comment token_{i} alpha_{i} beta_{i}\nint fn_{i}(void) {{ return {i + 1}; }}\n"
        for i in range(18)
    ))
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "t1")
    sha1 = _git(repo, "rev-parse", "HEAD")
    db1 = write_scan(tmp_path, "t1", port_rows_1, edges=origin_edges)

    card = build_port_evidence(
        origin_db,
        repo,
        [Snapshot(db0, sha0, "t0"), Snapshot(db1, sha1, "t1")],
        origin_label="origin",
        port_label="port",
        score_names=True,
    )
    assert len(card.snapshots) == 2
    assert card.ledger_lineages == 18
    assert card.ledger_links_per_step == [18]
    assert card.decay.verdict in ("one_time_port", "active_fork_or_mirror", "unrelated_or_clean_room")
    text = format_evidence_card(card)
    assert "PORT EVIDENCE CARD: port <- origin" in text
    assert "Isotonic Decay:" in text

    d = card_to_dict(card)
    assert d["port_label"] == "port"
    assert len(d["snapshots"]) == 2

    # Landmark path (max_fgw_nodes=10 < 18) and CLI --json path
    rc = cli_main([
        "port-evidence",
        "--origin-scan", str(origin_db),
        "--port-repo", str(repo),
        "--snapshot", f"{db0}@{sha0}@t0",
        "--snapshot", f"{db1}@{sha1}@t1",
        "--max-fgw-nodes", "10",
        "--json",
    ])
    assert rc == 0


def test_extract_cobol_comments(tmp_path: Path) -> None:
    cbl = tmp_path / "PROG.cbl"
    cbl.write_text(
        "000100* Validate customer billing account\n"
        "000200 1000-VALIDATE-ACCT.\n"
        "000300/ Check expiration date\n"
        "       *> Free format inline comment\n"
        "           DISPLAY 'OK'.\n"
    )
    db = tmp_path / "cobol.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE file_data (id INTEGER PRIMARY KEY, file_path TEXT)")
    conn.execute(
        "CREATE TABLE function_data (id INTEGER PRIMARY KEY, file_id INTEGER, func_name TEXT, "
        "start_line INTEGER, loc INTEGER, docstring TEXT)"
    )
    conn.execute("INSERT INTO file_data VALUES (1, 'PROG.cbl')")
    conn.execute("INSERT INTO function_data VALUES (1, 1, '1000-VALIDATE-ACCT', 2, 4, '')")
    conn.commit()
    conn.close()

    ft = SimpleNamespace(
        name=np.array(["1000-VALIDATE-ACCT"]),
        path=np.array(["PROG.cbl"]),
    )
    toks = extract_function_comments(db, ft, np.array([0]), repo=tmp_path)
    assert {"validate", "customer", "billing", "account", "check", "expiration", "date", "free", "format", "inline", "comment"} <= toks[0]
