"""Function ledger: a persistent ID for every function, carried across snapshots of one repository.

Path-and-name identity ("the function called ``f`` in ``a.c``") loses a function when it is renamed or moved,
and records a death plus a birth. The ledger follows functions through such changes by a cascade of evidence,
strongest first. Each link records the stage that made it, so any analysis can keep or drop a class of links.

Stages, applied to the functions left over by the stage before:

1. ``identity``: same path and name, same count in both snapshots; matched by order. Certain.
2. ``name_in_mapped_file``: same name in the file git says the old file became (``git diff -M50%``). Certain
   when the name is unique there; ties are broken by structure and flagged ``ambiguous``.
3. ``name_unique``: a name carried by exactly one dead and one new function. Verified by text.
4. ``in_file_structure``: structure (the lens vector) plus relative order inside the mapped file, mutual best
   match. Comment text breaks ties between look-alikes. Verified by text.
5. ``twin``: name-blind structural twins (:mod:`structural_fingerprint.twins`). Verified by text.

**Verification** (stages 3-5) keeps a proposed link only if the function's body text at the two commits
overlaps (Jaccard of 4-token shingles >= ``min_body``, own name masked, comments stripped) or its
docstring/preceding-comment words overlap (Jaccard >= ``min_doc``, at least 4 words each). Structure
proposes; text decides. Comments are independent of the body, which is why they work as a second witness.
Without ``start_line`` in the scan or git access to the commits there is no body text, and comments alone
verify; with neither, stages 3-5 link nothing and only the certain stages run.

Call-graph extension (propagating matches along call edges) was tried and removed: in four repositories
only 27% of its links had matching comments, against 91-99% for body-verified links.

What the ledger does not do: it does not link across a split or merge (one function becoming several, or
the reverse); :func:`classify_unlinked` names those causes for the births and deaths left over, using
same-file text containment. It reads the git object database; it never checks anything out.
"""

from __future__ import annotations

import collections
import logging
import re
import sqlite3
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import blind as BL
from .fingerprint import FunctionTable, build
from .reference import Reference, load_reference
from .scan_io import load_scan
from .twins import TwinParams, find_twins

log = logging.getLogger(__name__)

STAGES = ("identity", "name_in_mapped_file", "name_unique", "in_file_structure", "twin")
VERIFIED_STAGES = ("name_unique", "in_file_structure", "twin")
_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|[^\s\w]")
_WORD = re.compile(r"[A-Za-z]{3,}")
_HASH_COMMENT = (".py", ".rb", ".sh", ".pl", ".r", ".jl", ".ex", ".exs", ".yml", ".yaml")


@dataclass(frozen=True)
class Snapshot:
    """One scanned commit: a GitGalaxy scan database and the commit it was scanned at."""

    scan: Path
    commit: str
    label: str = ""


@dataclass(frozen=True)
class Link:
    """A function in snapshot *s* continued as function ``new`` in snapshot *s+1*."""

    new: int
    stage: str
    score: float = 0.0
    ambiguous: bool = False
    body: float | None = None  # body-text similarity, when it was measured
    doc: float | None = None  # comment-text similarity, when it was measured


@dataclass
class Ledger:
    snapshots: list[Snapshot]
    keys: list[list[tuple[str, str]]]  # per snapshot, per function: (path, name)
    ids: list[np.ndarray]  # per snapshot, per function: lineage id
    stage: list[list[str]]  # how each function got its id there: a stage, "birth0" (first snapshot) or "birth"
    links: list[dict[int, Link]]  # per consecutive pair: index in snapshot s -> Link
    _snaps: list = field(default_factory=list, repr=False)
    _repo: Path | None = field(default=None, repr=False)

    @property
    def n_lineages(self) -> int:
        return int(max((a.max() for a in self.ids if len(a)), default=-1)) + 1

    def span(self) -> dict[int, tuple[int, int]]:
        """lineage id -> (first snapshot, last snapshot) it was seen in."""
        out: dict[int, tuple[int, int]] = {}
        for s, a in enumerate(self.ids):
            for x in np.unique(a):
                f, _ = out.get(int(x), (s, s))
                out[int(x)] = (f, s)
        return out

    def births(self, s: int) -> np.ndarray:
        """Indices in snapshot *s* of functions with no ancestor in snapshot *s-1* (all of them for s = 0)."""
        return np.asarray([i for i, st in enumerate(self.stage[s]) if st.startswith("birth")], dtype=int)

    def deaths(self, s: int) -> np.ndarray:
        """Indices in snapshot *s* of functions with no successor in snapshot *s+1*."""
        if s >= len(self.links):
            return np.zeros(0, dtype=int)
        return np.asarray([i for i in range(len(self.keys[s])) if i not in self.links[s]], dtype=int)

    def rows(self) -> list[dict]:
        """One dict per (snapshot, function): lineage id, path, name, how it was linked. For CSV/JSON export."""
        out = []
        for s in range(len(self.ids)):
            for i, (path, name) in enumerate(self.keys[s]):
                out.append({"snapshot": s, "label": self.snapshots[s].label, "lineage": int(self.ids[s][i]),
                            "path": path, "name": name, "linked_by": self.stage[s][i]})
        return out


# ------------------------------------------------------------------ git and text


class _Git:
    """Reads blobs and file renames from the object database, cached."""

    def __init__(self, repo: Path):
        self.repo = Path(repo)
        self._blob: dict[tuple[str, str], list[str] | None] = {}
        self._ren: dict[tuple[str, str], dict[str, str]] = {}

    def lines(self, commit: str, path: str) -> list[str] | None:
        k = (commit, path)
        if k not in self._blob:
            r = subprocess.run(["git", "-C", str(self.repo), "show", f"{commit}:{path}"], capture_output=True, check=False)
            self._blob[k] = r.stdout.decode("utf-8", "replace").splitlines() if r.returncode == 0 else None
        return self._blob[k]

    def renames(self, c1: str, c2: str) -> dict[str, str]:
        """old path -> new path for files git sees as renamed between two commits (>= 50% similar)."""
        if (c1, c2) not in self._ren:
            r = subprocess.run(["git", "-C", str(self.repo), "diff", "-M50%", "--name-status", c1, c2],
                               capture_output=True, text=True, check=False)
            parts = (ln.split("\t") for ln in r.stdout.splitlines()) if r.returncode == 0 else ()
            self._ren[(c1, c2)] = {p[1]: p[2] for p in parts if p[0].startswith("R") and len(p) == 3}
        return self._ren[(c1, c2)]


def body_tokens(lines: list[str] | None, path: str, name: str, start: int, loc: int) -> list[str] | None:
    """Tokens of one function's text: comments stripped, its own name masked as ``@``."""
    if lines is None or start < 1:
        return None
    txt = "\n".join(lines[start - 1:start - 1 + max(loc, 1)])
    if path.lower().endswith(_HASH_COMMENT):
        txt = re.sub(r"#.*", "", txt)
    else:
        txt = re.sub(r"//.*", "", re.sub(r"/\*.*?\*/", "", txt, flags=re.DOTALL))
    return ["@" if t == name else t for t in _TOKEN.findall(txt)]


def shingles(tokens: list[str], k: int = 4) -> set[int]:
    return {hash(tuple(tokens[i:i + k])) for i in range(max(1, len(tokens) - k + 1))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


class _Snap:
    """One snapshot, prepared: function keys and lens vectors, plus (start line, length, comment) per function."""

    def __init__(self, snap: Snapshot, ref: Reference, min_loc: int, git: _Git | None):
        self.snap, self.git = snap, git
        scan = load_scan(snap.scan, ref.signal_columns)
        self.table = build(scan, ref, min_loc=min_loc, keyword_mode="composition").functions
        self.view = BL.blind_view(self.table)
        rows = self.view.rows
        self.keys = [(str(self.table.path[r]), str(self.table.name[r])) for r in rows]
        self.vec = self.table.vectors[rows]
        self.meta = self._meta(min_loc)
        self._sh: dict[int, set[int] | None] = {}
        self._wd: dict[int, set[str] | None] = {}

    def _meta(self, min_loc: int) -> list[tuple[int, int, str] | None]:
        """(start_line, loc, docstring) per function, matched to keys by (path, name, nth same name)."""
        con = sqlite3.connect(f"file:{self.snap.scan}?mode=ro", uri=True)
        try:
            cols = {r[1] for r in con.execute("PRAGMA table_info(function_data)")}
            start = "start_line" if "start_line" in cols else "NULL"
            doc = "docstring" if "docstring" in cols else "NULL"
            files = {r[0]: r[1] for r in con.execute("SELECT id, file_path FROM file_data")}
            by_key: dict[tuple[str, str], list] = collections.defaultdict(list)
            for fid, name, st, loc, d in con.execute(
                    f"SELECT file_id, func_name, {start}, loc, {doc} FROM function_data ORDER BY id"):
                if fid in files and (loc or 0) >= min_loc:
                    by_key[(files[fid], name)].append((int(st or 0), int(loc or 0), str(d or "")))
        finally:
            con.close()
        seen: collections.Counter = collections.Counter()
        out = []
        for k in self.keys:
            i = seen[k]
            seen[k] += 1
            rows = by_key.get(k, [])
            out.append(rows[i] if i < len(rows) else None)
        return out

    def shingles(self, i: int) -> set[int] | None:
        if i not in self._sh:
            m, sh = self.meta[i], None
            if m and self.git:
                t = body_tokens(self.git.lines(self.snap.commit, self.keys[i][0]), self.keys[i][0], self.keys[i][1],
                                m[0], m[1])
                sh = shingles(t) if t and len(t) >= 8 else None
            self._sh[i] = sh
        return self._sh[i]

    def words(self, i: int) -> set[str] | None:
        if i not in self._wd:
            m = self.meta[i]
            w = {x.lower() for x in _WORD.findall(m[2])} - {self.keys[i][1].lower()} if m else set()
            self._wd[i] = w if len(w) >= 4 else None
        return self._wd[i]

    def sub(self, idx: list[int]) -> FunctionTable:
        t, r = self.table, self.view.rows[np.asarray(idx, dtype=int)]
        return FunctionTable(repo=t.repo, name=t.name[r], path=t.path[r], language=t.language[r], loc=t.loc[r],
                             archetype=t.archetype[r], key=t.key[r], vectors=t.vectors[r], edges=None)


def similarity(a: _Snap, i: int, b: _Snap, j: int) -> tuple[float | None, float | None]:
    """(body-text similarity, comment-text similarity); None where a side has no usable text."""
    sa, sb = a.shingles(i), b.shingles(j)
    wa, wb = a.words(i), b.words(j)
    return (_jaccard(sa, sb) if sa and sb else None), (_jaccard(wa, wb) if wa and wb else None)


# ------------------------------------------------------------------ one pair of snapshots


def _positions(keys: list[tuple[str, str]]) -> dict[tuple[str, str], list[int]]:
    d: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i, k in enumerate(keys):
        d[k].append(i)
    return d


def match_pair(a: _Snap, b: _Snap, git: _Git | None, *, min_body: float = 0.5, min_doc: float = 0.5,
               use_comments: bool = True, twin_params: TwinParams | None = None) -> dict[int, Link]:
    """Link functions of snapshot *a* to functions of the next snapshot *b*. Returns old index -> Link."""
    ka, kb = a.keys, b.keys
    link: dict[int, Link] = {}
    used: set[int] = set()
    pa, pb = _positions(ka), _positions(kb)
    for k, al in pa.items():  # stage 1: identical path+name, equal count
        bl = pb.get(k)
        if bl and len(al) == len(bl):
            for x, y in zip(al, bl, strict=True):
                link[x] = Link(y, "identity")
                used.add(y)
    dead = [x for x in range(len(ka)) if x not in link]
    born = [y for y in range(len(kb)) if y not in used]
    fr = git.renames(a.snap.commit, b.snap.commit) if git else {}

    def take(pairs: list[tuple[int, int, float, bool]], stage: str, verify: bool) -> None:
        nonlocal dead, born
        for x, y, score, amb in pairs:
            body = doc = None
            if verify:
                body, doc = similarity(a, x, b, y)
                if not use_comments:
                    doc = None
                if not ((body is not None and body >= min_body) or (doc is not None and doc >= min_doc)):
                    continue
            link[x] = Link(y, stage, score, amb, body, doc)
            used.add(y)
        dead = [x for x in dead if x not in link]
        born = [y for y in born if y not in used]

    # stage 2: same name in the file git says the old one became
    by_key: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for y in born:
        by_key[kb[y]].append(y)
    grp: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for x in dead:
        grp[(fr.get(ka[x][0], ka[x][0]), ka[x][1])].append(x)
    pairs = []
    for k, al in grp.items():
        bl = by_key.get(k)
        if not bl:
            continue
        if len(al) == len(bl):
            pairs += [(x, y, 0.0, False) for x, y in zip(al, bl, strict=True)]
        else:  # unequal counts: mutual nearest in structure, flagged ambiguous
            d = np.sqrt(((a.vec[al][:, None, :] - b.vec[bl][None, :, :]) ** 2).sum(2))
            bi, di = d.argmin(1), d.argmin(0)
            pairs += [(x, bl[bi[i]], float(d[i, bi[i]]), True) for i, x in enumerate(al) if di[bi[i]] == i]
    take(pairs, "name_in_mapped_file", False)

    # stage 3: a name that exactly one dead and one new function carry
    dn = collections.Counter(ka[x][1] for x in dead)
    bn = collections.Counter(kb[y][1] for y in born)
    by_name = {kb[y][1]: y for y in born}
    take([(x, by_name[ka[x][1]], 0.0, False) for x in dead if dn[ka[x][1]] == 1 and bn.get(ka[x][1]) == 1],
         "name_unique", True)

    # stage 4: structure + relative order inside the mapped file; comments break ties
    pools: dict[str, tuple[list[int], list[int]]] = collections.defaultdict(lambda: ([], []))
    for x in dead:
        pools[fr.get(ka[x][0], ka[x][0])][0].append(x)
    for y in born:
        pools[kb[y][0]][1].append(y)
    pairs = []
    for ds, bs in pools.values():
        if not ds or not bs:
            continue
        d = np.sqrt(((a.vec[ds][:, None, :] - b.vec[bs][None, :, :]) ** 2).sum(2))
        rp = (np.argsort(np.argsort(ds)) + 0.5) / len(ds)
        rq = (np.argsort(np.argsort(bs)) + 0.5) / len(bs)
        t = d + 0.5 * np.abs(rp[:, None] - rq[None, :])
        if use_comments:
            for i, x in enumerate(ds):
                wa = a.words(x)
                if wa:
                    for j, y in enumerate(bs):
                        wb = b.words(y)
                        if wb:
                            t[i, j] -= 0.6 * _jaccard(wa, wb)
        bi, di = t.argmin(1), t.argmin(0)
        for i, x in enumerate(ds):
            j = bi[i]
            if di[j] == i:
                srt = np.sort(t[i])
                pairs.append((x, bs[j], float(t[i, j]), bool(len(srt) > 1 and srt[1] - srt[0] < 1e-6)))
    take(pairs, "in_file_structure", True)

    # stage 5: name-blind structural twins on what is left
    pairs = []
    if len(dead) >= 3 and len(born) >= 3:
        try:
            r = find_twins(a.sub(dead), b.sub(born), twin_params or TwinParams(min_loc=3, ratio=0.8, fdr=0.01))
            pairs = [(dead[int(q)], born[int(t)], float(dd), False)
                     for q, t, dd in zip(r.q_index, r.t_index, r.distance, strict=True)]
        except ValueError as e:  # too few usable functions on a side
            log.debug("twin stage skipped: %s", e)
    take(pairs, "twin", True)
    return link


# ------------------------------------------------------------------ the whole history


def build_ledger(repo: Path, snapshots: list[Snapshot], ref: Reference | None = None, *, min_loc: int = 3,
                 min_body: float = 0.5, min_doc: float = 0.5, use_comments: bool = True,
                 twin_params: TwinParams | None = None, progress: Callable[[int, int], None] | None = None) -> Ledger:
    """Assign a persistent lineage id to every function across ``snapshots`` (oldest first) of one repository.

    ``repo`` is the git checkout (or bare repository) the commits belong to; it is only read. Snapshots whose
    scan databases lack ``start_line`` still work, but only the certain stages can link."""
    if len(snapshots) < 1:
        raise ValueError("need at least one snapshot")
    ref = ref or load_reference()
    git = _Git(repo)
    snaps = [_Snap(s, ref, min_loc, git) for s in snapshots]
    keys = [s.keys for s in snaps]
    links = []
    for i in range(len(snaps) - 1):
        links.append(match_pair(snaps[i], snaps[i + 1], git, min_body=min_body, min_doc=min_doc,
                                use_comments=use_comments, twin_params=twin_params))
        if progress:
            progress(i + 1, len(snaps) - 1)
    ids = [np.arange(len(keys[0]), dtype=np.int64)]
    stage = [["birth0"] * len(keys[0])]
    nxt = len(keys[0])
    for s, link in enumerate(links):
        cur = np.full(len(keys[s + 1]), -1, dtype=np.int64)
        st = ["birth"] * len(keys[s + 1])
        for x, lk in link.items():
            cur[lk.new] = ids[s][x]
            st[lk.new] = lk.stage
        for y in np.flatnonzero(cur < 0):
            cur[y] = nxt
            nxt += 1
        ids.append(cur)
        stage.append(st)
    return Ledger(list(snapshots), keys, ids, stage, links, _snaps=snaps, _repo=Path(repo))


# ------------------------------------------------------------------ causes of the births and deaths left over


def classify_unlinked(ledger: Ledger, s: int, *, containment: float = 0.7, coverage: float = 0.5) -> tuple[dict, dict]:
    """Name the cause of every unlinked death (snapshot *s*) and unlinked birth (snapshot *s+1*).

    Returns ``(dead_cause, born_cause)``: index -> label. Death labels: ``split``, ``merged``,
    ``file_removed``, ``deleted``. Birth labels: ``split_or_extracted``, ``merge_result``, ``new_file``,
    ``added``. Split/merge/extract/inline are detected by text containment inside the same (git-mapped) file:
    a dead function whose text reappears across two or more new functions is a split; a new function holding
    two or more dead functions is a merge; a surviving function that shrank to <= 70% while a new function
    holds its missing text is an extraction (and the reverse, growing by >= 40%, an inlining)."""
    if ledger._repo is None or not ledger._snaps:
        raise ValueError("this ledger has no repository attached; build it with build_ledger")
    a, b = ledger._snaps[s], ledger._snaps[s + 1]
    link = ledger.links[s]
    fr = _Git(ledger._repo).renames(a.snap.commit, b.snap.commit)
    used = {lk.new for lk in link.values()}
    dead = [x for x in range(len(a.keys)) if x not in link]
    born = [y for y in range(len(b.keys)) if y not in used]
    files_b = {k[0] for k in b.keys}
    files_a = {k[0] for k in a.keys} | set(fr.values())
    pools: dict[str, tuple[list[int], list[int]]] = collections.defaultdict(lambda: ([], []))
    for x in dead:
        pools[fr.get(a.keys[x][0], a.keys[x][0])][0].append(x)
    for y in born:
        pools[b.keys[y][0]][1].append(y)

    def cont(x: set, y: set) -> float:
        return len(x & y) / len(x) if x else 0.0

    dc: dict[int, str] = {}
    bc: dict[int, str] = {}
    for ds, bs in pools.values():
        for x in ds:
            sx = a.shingles(x)
            if sx and bs:
                parts = [y for y in bs if b.shingles(y) and cont(b.shingles(y), sx) >= containment]
                if len(parts) >= 2 and len(set().union(*[b.shingles(y) & sx for y in parts])) / len(sx) >= coverage:
                    dc[x] = "split"
                    for y in parts:
                        bc[y] = "split_or_extracted"
        for y in bs:
            sy = b.shingles(y)
            if sy and ds:
                srcs = [x for x in ds if a.shingles(x) and cont(a.shingles(x), sy) >= containment]
                if len(srcs) >= 2 and len(set().union(*[a.shingles(x) & sy for x in srcs])) / len(sy) >= coverage:
                    bc.setdefault(y, "merge_result")
                    for x in srcs:
                        dc[x] = "merged"
    for x, lk in link.items():
        ma, mb = a.meta[x], b.meta[lk.new]
        if not ma or not mb or ma[1] == mb[1]:
            continue
        sa, sb = a.shingles(x), b.shingles(lk.new)
        if not sa or not sb:
            continue
        ds, bs = pools.get(fr.get(a.keys[x][0], a.keys[x][0]), ([], []))
        if mb[1] <= 0.7 * ma[1]:
            for y in bs:
                sy = b.shingles(y)
                if sy and y not in bc and cont(sy, sa) >= containment and cont(sy, sb) < 0.3:
                    bc[y] = "split_or_extracted"
        if mb[1] >= 1.4 * ma[1]:
            for z in ds:
                sz = a.shingles(z)
                if sz and z not in dc and cont(sz, sb) >= containment and cont(sz, sa) < 0.3:
                    dc[z] = "merged"
    for x in dead:
        if x not in dc:
            dc[x] = "file_removed" if fr.get(a.keys[x][0], a.keys[x][0]) not in files_b else "deleted"
    for y in born:
        if y not in bc:
            bc[y] = "new_file" if b.keys[y][0] not in files_a else "added"
    return dc, bc
