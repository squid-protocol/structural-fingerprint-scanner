import os
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pytest

from structural_fingerprint import ledger as LG

from .conftest import SIGNALS, write_scan

VOCAB = ["alpha", "bravo", "cargo", "delta", "echo", "fable", "gamma", "hotel", "india", "juliet", "kilo", "lima",
         "mike", "novel", "oscar", "papa", "quest", "romeo", "sierra", "tango", "umber", "victor", "whisky", "xray"]


def make_body(name: str, seed: int, words: bool = True) -> list[str]:
    """A small, distinctive function: its own identifiers and constants, so unrelated bodies share no shingles."""
    r = np.random.default_rng(seed)
    v = lambda: VOCAB[int(r.integers(len(VOCAB)))] + str(int(r.integers(100)))
    lines = [f"def {name}(x):"]
    if words:
        lines.append(f"    # {v()} {v()} {v()} {v()} {v()} {v()} notes for {name}")
    for _ in range(7):
        lines.append(f"    {v()} = {v()} + {int(r.integers(1000))} * {v()}")
    lines.append(f"    return {v()}")
    return lines


def git(repo: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


def commit_tree(repo: Path, files: dict[str, list[tuple[str, int]]], scans: Path, tag: str, start_line: bool = True):
    """Write ``files`` (path -> [(function name, seed)]) to the repo, commit, and write a matching scan database."""
    for p in repo.rglob("*.py"):
        p.unlink()
    funcs = []
    for path, fl in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        text, line = [], 1
        for name, seed in fl:
            body = make_body(name, seed)
            text += body + [""]
            sig = {c: int(np.random.default_rng(seed + i).poisson(1.2)) for i, c in enumerate(SIGNALS)}
            funcs.append({"name": name, "loc": len(body), "complexity": 2 + seed % 4, "args": 1,
                          "keyword_density": 1.0, "path": path, "signals": sig, "start": line,
                          "doc": body[1] if body[1].startswith("    #") else ""})
            line += len(body) + 1
        (repo / path).write_text("\n".join(text) + "\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", tag)
    db = write_scan(scans, tag, funcs)
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE function_data ADD COLUMN start_line INTEGER")
    con.execute("ALTER TABLE function_data ADD COLUMN docstring TEXT")
    for k, f in enumerate(funcs, 1):
        con.execute("UPDATE function_data SET start_line=?, docstring=? WHERE id=?",
                    (f["start"] if start_line else None, f["doc"], k))
    if not start_line:  # a scan with neither line numbers nor comments: nothing to verify with
        con.execute("UPDATE function_data SET start_line=NULL, docstring=''")
    con.commit()
    con.close()
    return LG.Snapshot(db, git(repo, "rev-parse", "HEAD"), tag)


@pytest.fixture()
def history(tmp_path):
    repo, scans = tmp_path / "repo", tmp_path / "scans"
    repo.mkdir()
    git(repo, "init", "-q")
    a = [(f"fa{i}", 100 + i) for i in range(10)]
    b = [(f"gb{i}", 200 + i) for i in range(6)]
    s0 = commit_tree(repo, {"src/a.py": a, "src/b.py": b}, scans, "s0")
    # s1: a.py is renamed to core.py; fa3 is renamed to fa3_new (same body); fa5 moves to b.py; gb0's body is replaced
    a1 = [(("fa3_new" if n == "fa3" else n), s) for n, s in a if n != "fa5"]
    b1 = [("gb0", 999)] + b[1:] + [("fa5", 105)]
    s1 = commit_tree(repo, {"src/core.py": a1, "src/b.py": b1}, scans, "s1")
    # s2: gb5 is deleted
    s2 = commit_tree(repo, {"src/core.py": a1, "src/b.py": b1[:5] + b1[6:]}, scans, "s2")
    return repo, [s0, s1, s2], scans


def lineage_of(ledger: LG.Ledger, s: int, path: str, name: str) -> int:
    return int(ledger.ids[s][ledger.keys[s].index((path, name))])


def test_certain_stages_follow_file_rename_function_rename_and_move(history):
    repo, snaps, _ = history
    led = LG.build_ledger(repo, snaps)
    # a file rename: every function keeps its id
    for i in (0, 1, 2, 4):
        assert lineage_of(led, 0, "src/a.py", f"fa{i}") == lineage_of(led, 1, "src/core.py", f"fa{i}")
    # a function rename (body unchanged): same id, linked by structure and verified by text
    assert lineage_of(led, 0, "src/a.py", "fa3") == lineage_of(led, 1, "src/core.py", "fa3_new")
    # a move between files, same name: same id
    assert lineage_of(led, 0, "src/a.py", "fa5") == lineage_of(led, 1, "src/b.py", "fa5")
    # untouched functions are carried by identity across the whole history
    assert lineage_of(led, 0, "src/b.py", "gb2") == lineage_of(led, 2, "src/b.py", "gb2")
    stages = {led.keys[1][lk.new][1]: lk.stage for lk in led.links[0].values()}
    assert stages["fa3_new"] in ("in_file_structure", "twin", "name_unique")
    assert stages["fa0"] == "name_in_mapped_file"


def test_verification_refuses_a_name_match_with_unrelated_text(history):
    repo, snaps, _ = history
    led = LG.build_ledger(repo, snaps)
    # gb0 in s1 has the same name and file but an unrelated body: the identity stage links it (certain stages do
    # not look at text), so it keeps the id; the point of verification is the proposed stages. A proposed link
    # with unrelated text must not appear: no verified link may have both similarities below the threshold.
    for links in led.links:
        for lk in links.values():
            if lk.stage in LG.VERIFIED_STAGES:
                assert (lk.body or 0) >= 0.5 or (lk.doc or 0) >= 0.5


def test_a_deleted_function_is_a_death_with_a_cause(history):
    repo, snaps, _ = history
    led = LG.build_ledger(repo, snaps)
    gone = led.keys[1].index(("src/b.py", "gb5"))
    assert gone in set(led.deaths(1))
    assert lineage_of(led, 1, "src/b.py", "gb5") not in set(led.ids[2])
    dead, born = LG.classify_unlinked(led, 1)
    assert dead[gone] == "deleted"
    assert not born  # nothing was born between s1 and s2


def test_without_line_numbers_or_comments_only_certain_stages_link(history):
    repo, snaps, scans = history
    bare = commit_tree(repo, {"src/core.py": [(f"fa{i}", 100 + i) for i in range(10)]}, scans, "s3", start_line=False)
    led = LG.build_ledger(repo, [snaps[2], bare])
    assert all(lk.stage in ("identity", "name_in_mapped_file") for lk in led.links[0].values())


def test_ledger_bookkeeping(history):
    repo, snaps, _ = history
    led = LG.build_ledger(repo, snaps)
    assert [len(a) for a in led.ids] == [len(k) for k in led.keys]
    assert len(led.births(0)) == len(led.keys[0])
    assert set(led.stage[0]) == {"birth0"}
    span = led.span()
    assert all(f <= last for f, last in span.values())
    rows = led.rows()
    assert len(rows) == sum(len(k) for k in led.keys)
    assert {"snapshot", "lineage", "path", "name", "linked_by"} <= set(rows[0])


def test_comments_switch_changes_nothing_when_bodies_agree(history):
    repo, snaps, _ = history
    on = LG.build_ledger(repo, snaps)
    off = LG.build_ledger(repo, snaps, use_comments=False)
    assert [list(a) for a in on.ids] == [list(a) for a in off.ids]


def test_build_needs_a_snapshot(tmp_path):
    with pytest.raises(ValueError):
        LG.build_ledger(tmp_path, [])
