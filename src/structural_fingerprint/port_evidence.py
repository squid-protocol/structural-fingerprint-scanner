"""Multi-snapshot Port Evidence Card: Structure + Comments + Ledger + Isotonic Decay.

Combines four orthogonal lines of evidence to detect and characterize code ports
from a suspect origin repository into a target repository:

1. **Pure Structure (Tier A):** ``NO_COMPLEXITY`` FGW + Seed-and-Extend
   (descriptors + hop-distance call graph, zero names, zero comments).
2. **Multi-Modal (+ Comments, Tier B):** Optional comment/docstring Jaccard
   boost in FGW cost and 1-hop extension to recover OOP-refactored ports.
3. **Persistent Birth IDs (Ledger):** Carries early-snapshot matches forward
   to ``HEAD`` via :func:`structural_fingerprint.ledger.build_ledger`.
4. **Isotonic Decay Fit:** Pool Adjacent Violators (PAVA) non-increasing fit on
   the structural match curve ``M(t0) -> M(t1) -> ... -> M(HEAD)`` to distinguish
   one-time copy + divergence from ongoing mirrors or clean-room rewrites.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import blind as BL
from . import features as F
from .fingerprint import FunctionTable, build
from .ledger import Ledger, Snapshot, _Git, build_ledger
from .reference import Reference, load_reference
from .scan_io import ScanData, load_scan

_STOP = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "are", "not", "but",
    "have", "has", "was", "were", "will", "can", "all", "one", "two", "use",
    "used", "using", "when", "where", "which", "into", "out", "only", "also",
    "than", "then", "them", "they", "their", "there", "these", "those", "what",
    "todo", "fixme", "note", "return", "returns", "function", "method", "param",
    "int", "void", "char", "const", "static", "public", "private", "bool",
    "true", "false", "null", "nil", "none", "self", "cls", "var", "let",
})
_WORD_RE = re.compile(r"[a-z]{3,}")
_HASH_EXTS = (".py", ".rb", ".sh", ".pl", ".r", ".jl", ".ex", ".exs")
_DROP_COMPLEXITY = ("log_complexity", "func_internal_density")


def _norm_name(s: str) -> str:
    return s.strip().lower().replace("-", "_")


def same_name(a: str, b: str) -> bool:
    """Post-hoc name agreement helper (never used during matching)."""
    na, nb = _norm_name(a), _norm_name(b)
    if na == nb:
        return True
    if "_" in nb and na.split("_", 1)[-1] == nb.split("_", 1)[-1]:
        return len(na.split("_", 1)[-1]) >= 4
    return False


@dataclass(frozen=True)
class SnapshotEvidence:
    """Alignment summary for one target snapshot compared against origin."""

    label: str
    commit: str
    n_port: int
    n_origin: int
    n_commented_port: int
    struct_matched: int
    struct_high_cos: int
    struct_mean_cos: float
    struct_coverage: float
    multimodal_matched: int
    multimodal_mean_cos: float
    multimodal_coverage: float
    comment_rbh: int
    edge_pres: float = 0.0
    rbh_hop_mae: float = 1.0
    birth0_retained: int = 0
    new_births: int = 0
    post_t0_cmt_rbh: int = 0
    struct_name_ok: int | None = None
    multimodal_name_ok: int | None = None
    comment_name_ok: int | None = None


@dataclass(frozen=True)
class IsotonicDecay:
    """Non-increasing PAVA fit over the chronological structural match curve."""

    labels: list[str]
    raw_rates: list[float]
    fitted_rates: list[float]
    birth_rate: float
    head_rate: float
    decay_drop: float
    verdict: str


@dataclass(frozen=True)
class PortEvidenceCard:
    """Full forensic summary combining structure, comments, ledger, and decay."""

    origin_label: str
    port_label: str
    snapshots: list[SnapshotEvidence]
    ledger_lineages: int
    ledger_links_per_step: list[int]
    head_direct_n: int
    head_projected_n: int
    head_confirmed_n: int
    head_union_n: int
    decay: IsotonicDecay
    head_direct_ok: int | None = None
    head_projected_ok: int | None = None
    head_confirmed_ok: int | None = None
    head_union_ok: int | None = None


def isotonic_nonincreasing(
    values: Sequence[float],
    weights: Sequence[float] | None = None,
) -> list[float]:
    """Pool Adjacent Violators Algorithm (PAVA) for non-increasing fit."""
    n = len(values)
    if n == 0:
        return []
    w_in = [1.0] * n if weights is None else [float(x) for x in weights]
    blocks: list[list[float | int]] = []  # [weighted_sum, weight_sum, count]
    for v, w in zip(values, w_in, strict=True):
        blocks.append([float(v) * w, w, 1])
        while len(blocks) >= 2:
            prev = blocks[-2]
            curr = blocks[-1]
            if (prev[0] / prev[1]) < (curr[0] / curr[1]):
                prev[0] += curr[0]
                prev[1] += curr[1]
                prev[2] += curr[2]
                blocks.pop()
            else:
                break
    out: list[float] = []
    for wsum, wtot, cnt in blocks:
        mean = wsum / wtot if wtot > 0 else 0.0
        out.extend([mean] * int(cnt))
    return out


def suggest_checkpoints(repo: Path, max_snapshots: int = 5) -> list[tuple[str, str]]:
    """Suggest ``(commit_sha, label)`` checkpoints from git history without checking out.

    Picks the first commit with substantial code, up to ``max_snapshots - 2``
    highest-churn refactor commits, and ``HEAD``.
    """
    r = subprocess.run(
        ["git", "-C", str(repo), "log", "--first-parent", "--reverse", "--shortstat", "--format=COMMIT %h %cs"],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return []

    commits: list[tuple[str, str, int]] = []
    cur_sha, cur_date = "", ""
    for ln in r.stdout.splitlines():
        if ln.startswith("COMMIT "):
            parts = ln.split()
            cur_sha = parts[1]
            cur_date = parts[2] if len(parts) > 2 else ""
        elif "changed" in ln and cur_sha:
            m = re.search(r"(\d+)\s+file", ln)
            files_changed = int(m.group(1)) if m else 0
            commits.append((cur_sha, cur_date, files_changed))
            cur_sha = ""
    if cur_sha:
        commits.append((cur_sha, cur_date, 0))
    if not commits:
        return []
    if len(commits) <= max_snapshots:
        return [(sha, f"{dt} ({sha})") for sha, dt, _ in commits]

    chosen_idx = {0, len(commits) - 1}
    interior = sorted(range(1, len(commits) - 1), key=lambda i: commits[i][2], reverse=True)
    for idx in interior:
        if len(chosen_idx) >= max_snapshots:
            break
        # Avoid clustering adjacent commits
        if all(abs(idx - c) >= max(2, len(commits) // (max_snapshots * 2)) for c in chosen_idx):
            chosen_idx.add(idx)
    for idx in interior:
        if len(chosen_idx) >= max_snapshots:
            break
        chosen_idx.add(idx)

    return [(commits[i][0], f"{commits[i][1]} ({commits[i][0]})") for i in sorted(chosen_idx)]


def _is_comment_line(line: str, is_hash: bool) -> bool:
    s = line.lstrip()
    if is_hash:
        return s.startswith("#")
    return s.startswith("//") or s.startswith("/*") or s.startswith("*") or s.startswith("*/")


def extract_function_comments(
    scan_path: Path,
    ft: FunctionTable,
    rows: np.ndarray,
    repo: Path | None = None,
    commit: str | None = None,
) -> list[set[str]]:
    """Extract lowercase comment+docstring token sets aligned to ``rows`` of ``ft``."""
    con = sqlite3.connect(f"file:{scan_path}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(function_data)")}
        start_col = "start_line" if "start_line" in cols else "NULL"
        doc_col = "docstring" if "docstring" in cols else "NULL"
        files = {r[0]: r[1] for r in con.execute("SELECT id, file_path FROM file_data")}
        by_key: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
        for fid, fname, sl, loc, doc in con.execute(
            f"SELECT file_id, func_name, {start_col}, loc, {doc_col} FROM function_data ORDER BY id"
        ):
            if fid in files:
                by_key[(files[fid], str(fname))].append((int(sl or 0), int(loc or 0), str(doc or "")))
    finally:
        con.close()

    git = _Git(repo) if (repo is not None and commit is not None) else None
    fs_cache: dict[str, list[str] | None] = {}

    def get_lines(rel_path: str) -> list[str] | None:
        if git is not None:
            return git.lines(commit, rel_path)
        if repo is None:
            return None
        if rel_path not in fs_cache:
            fp = repo / rel_path
            try:
                fs_cache[rel_path] = fp.read_text(encoding="utf-8", errors="ignore").splitlines() if fp.exists() else None
            except OSError:
                fs_cache[rel_path] = None
        return fs_cache[rel_path]

    seen: Counter[tuple[str, str]] = Counter()
    out: list[set[str]] = []
    for r in rows:
        k = (str(ft.path[r]), str(ft.name[r]))
        idx_occ = seen[k]
        seen[k] += 1
        meta_list = by_key.get(k, [])
        if idx_occ >= len(meta_list):
            out.append(set())
            continue
        sl, loc, doc = meta_list[idx_occ]
        chunks: list[str] = [doc] if doc else []
        lines = get_lines(k[0])
        if lines and sl > 0:
            n_lines = len(lines)
            sl_c = max(1, min(sl, n_lines))
            is_hash = k[0].lower().endswith(_HASH_EXTS)
            lead: list[str] = []
            li = sl_c - 2
            steps = 0
            while 0 <= li < n_lines and steps < 15:
                s = lines[li].strip()
                if not s:
                    if lead:
                        break
                    li -= 1
                    steps += 1
                    continue
                if _is_comment_line(s, is_hash):
                    lead.append(s)
                    li -= 1
                    steps += 1
                else:
                    break
            if lead:
                chunks.extend(reversed(lead))
            b_start = max(0, sl_c - 1)
            b_end = min(n_lines, b_start + max(1, loc))
            for bl in lines[b_start:b_end]:
                if is_hash:
                    if "#" in bl:
                        chunks.append(bl[bl.index("#"):])
                elif "//" in bl:
                    chunks.append(bl[bl.index("//"):])
                elif "/*" in bl:
                    chunks.append(bl[bl.index("/*"):])
                elif bl.lstrip().startswith("*"):
                    chunks.append(bl.strip())
        toks = {t for t in _WORD_RE.findall("\n".join(chunks).lower()) if t not in _STOP}
        out.append(toks)
    return out


def _jaccard_matrix(p_toks: list[set[str]], r_toks: list[set[str]]) -> np.ndarray:
    inv_r: dict[str, list[int]] = defaultdict(list)
    for ru, toks in enumerate(r_toks):
        for tok in toks:
            inv_r[tok].append(ru)
    j_mat = np.zeros((len(p_toks), len(r_toks)), dtype=np.float64)
    for pu, toks in enumerate(p_toks):
        if not toks:
            continue
        counts: Counter[int] = Counter()
        for tok in toks:
            for ru in inv_r.get(tok, ()):
                counts[ru] += 1
        lp = len(toks)
        for ru, inter in counts.items():
            j_mat[pu, ru] = inter / (lp + len(r_toks[ru]) - inter)
    return j_mat


@dataclass
class _PreparedSide:
    ft: FunctionTable
    view: BL.BlindView
    x: np.ndarray
    h: np.ndarray
    comp: np.ndarray
    norm: np.ndarray
    nbrs: dict[int, set[int]]
    toks: list[set[str]]


def _prepare_side(
    scan_path: Path,
    ref: Reference,
    bp: BL.BlindParams,
    keep_cols: list[int],
    min_loc: int,
    repo: Path | None,
    commit: str | None,
) -> _PreparedSide:
    scan: ScanData = load_scan(scan_path, ref.signal_columns)
    ft = build(scan, ref, min_loc=min_loc, keyword_mode="composition").functions
    assert ft is not None
    view = BL.blind_view(ft)
    x = BL.descriptors(view, len(ref.archetype_names), bp.use_graph, bp.use_measures, None, bp.geometry_only)
    h = BL.hop_matrix(view, bp.hop_cap)
    comp_all = F.raw_features(
        ref, scan.loc, scan.complexity, scan.args, scan.keyword_density, scan.signals,
        keyword_mode="composition",
    )[:, keep_cols]
    comp = comp_all[view.rows]
    norm = comp / np.maximum(np.linalg.norm(comp, axis=1, keepdims=True), 1e-9)
    nbrs: dict[int, set[int]] = defaultdict(set)
    for u, w in view.edges.tolist():
        nbrs[u].add(w)
        nbrs[w].add(u)
    toks = extract_function_comments(scan_path, ft, view.rows, repo=repo, commit=commit)
    return _PreparedSide(ft=ft, view=view, x=x, h=h, comp=comp, norm=norm, nbrs=nbrs, toks=toks)


def _seed_and_extend(
    p: _PreparedSide,
    r: _PreparedSide,
    bp: BL.BlindParams,
    *,
    j_mat: np.ndarray | None = None,
    fgw_comment_w: float = 0.0,
    ext_comment_w: float = 0.0,
    seed_frac: float = 0.15,
    hops: int = 3,
) -> tuple[dict[int, int], float, int, float, float]:
    if p.view.n == 0 or r.view.n == 0:
        return {}, 0.0, 0, 0.0, 1.0
    m = BL._sqdist(p.x, r.x)
    m = m / (m.max() or 1.0)
    if j_mat is not None and fgw_comment_w > 0.0:
        m = np.maximum(0.0, m - fgw_comment_w * j_mat)
    t = BL.fused_gw(m, p.h, r.h, bp.alpha, bp.epsilon, bp.outer)

    best_r = np.argmax(t, axis=1)
    best_p = np.argmax(t, axis=0)
    rbh = [(i, int(best_r[i]), float(t[i, best_r[i]])) for i in range(p.view.n) if best_p[best_r[i]] == i]
    rbh.sort(key=lambda x: x[2], reverse=True)
    k_seed = max(1, int(len(rbh) * seed_frac)) if rbh else 0
    seeds = rbh[:k_seed]

    if len(rbh) > 1:
        pi_rbh = np.array([i for i, _, _ in rbh], dtype=np.int64)
        ri_rbh = np.array([j for _, j, _ in rbh], dtype=np.int64)
        rbh_hop_mae = float(np.mean(np.abs(p.h[np.ix_(pi_rbh, pi_rbh)] - r.h[np.ix_(ri_rbh, ri_rbh)])))
    else:
        rbh_hop_mae = 1.0

    anchor: dict[int, int] = {i: j for i, j, _ in seeds}
    used_r: set[int] = set(anchor.values())
    if seeds:
        pi_s = np.array([i for i, _, _ in seeds], dtype=np.int64)
        ri_s = np.array([j for _, j, _ in seeds], dtype=np.int64)
        delta = np.mean(p.comp[pi_s] - r.comp[ri_s], axis=0)
    else:
        delta = np.zeros(p.comp.shape[1])
    p_sh = p.comp - delta
    p_sh_norm = p_sh / np.maximum(np.linalg.norm(p_sh, axis=1, keepdims=True), 1e-9)

    for _ in range(hops):
        proposals: list[tuple[float, int, int]] = []
        for pu in range(p.view.n):
            if pu in anchor:
                continue
            cand_votes: Counter[int] = Counter()
            for pn in p.nbrs.get(pu, ()):
                if pn in anchor:
                    for rc in r.nbrs.get(anchor[pn], ()):
                        if rc not in used_r:
                            cand_votes[rc] += 1
            if not cand_votes:
                continue
            cands = np.array(list(cand_votes.keys()), dtype=np.int64)
            votes = np.array([cand_votes[c] for c in cands], dtype=np.float64)
            cos = r.norm[cands] @ p_sh_norm[pu]
            j_vec = j_mat[pu, cands] if (j_mat is not None and ext_comment_w > 0.0) else 0.0
            scores = votes * (1.0 + np.maximum(cos, 0.0) + ext_comment_w * j_vec) + 0.25 * t[pu, cands] / (t[pu].max() or 1.0)
            best_idx = int(np.argmax(scores))
            proposals.append((float(scores[best_idx]), pu, int(cands[best_idx])))
        if not proposals:
            break
        proposals.sort(reverse=True)
        added = 0
        for _, pu, ru in proposals:
            if pu not in anchor and ru not in used_r:
                anchor[pu] = ru
                used_r.add(ru)
                added += 1
        if added == 0:
            break

    if not anchor:
        return anchor, 0.0, 0, 0.0, rbh_hop_mae
    cos_vals = [float(r.norm[ru] @ p_sh_norm[pu]) for pu, ru in anchor.items()]
    high_cos = sum(1 for c in cos_vals if c >= 0.95)
    total_e = 0
    pres_e = 0
    for pu, ru in anchor.items():
        for pv in p.nbrs.get(pu, ()):
            if pv > pu and pv in anchor:
                total_e += 1
                if anchor[pv] in r.nbrs.get(ru, ()):
                    pres_e += 1
    edge_pres = pres_e / total_e if total_e > 0 else 0.0
    return anchor, float(np.mean(cos_vals)), high_cos, edge_pres, rbh_hop_mae


def _comment_rbh(j_mat: np.ndarray, p_toks: list[set[str]], r_toks: list[set[str]], min_toks: int = 3, min_j: float = 0.25) -> dict[int, int]:
    if j_mat.size == 0:
        return {}
    p_valid = np.array([len(t) >= min_toks for t in p_toks], dtype=bool)
    r_valid = np.array([len(t) >= min_toks for t in r_toks], dtype=bool)
    jm = j_mat.copy()
    jm[~p_valid, :] = 0.0
    jm[:, ~r_valid] = 0.0
    best_r = np.argmax(jm, axis=1)
    best_p = np.argmax(jm, axis=0)
    return {
        pu: int(best_r[pu])
        for pu in range(len(p_toks))
        if best_p[best_r[pu]] == pu and jm[pu, best_r[pu]] >= min_j
    }


def _count_name_ok(mapping: dict[int, int], p: _PreparedSide, r: _PreparedSide) -> int:
    return sum(
        1 for pu, ru in mapping.items()
        if same_name(str(p.ft.name[p.view.rows[pu]]), str(r.ft.name[r.view.rows[ru]]))
    )


def build_port_evidence(
    origin_scan: Path,
    port_repo: Path,
    snapshots: list[Snapshot],
    *,
    origin_repo: Path | None = None,
    origin_label: str = "origin",
    port_label: str | None = None,
    ref: Reference | None = None,
    min_loc: int = 3,
    score_names: bool = False,
) -> PortEvidenceCard:
    """Build a multi-snapshot Port Evidence Card for ``port_repo`` vs. ``origin_scan``."""
    if not snapshots:
        raise ValueError("need at least one snapshot")
    ref = ref or load_reference("portable")
    bp = BL.BlindParams()
    keep_cols = [i for i, n in enumerate(ref.feature_names) if n not in _DROP_COMPLEXITY]

    origin = _prepare_side(origin_scan, ref, bp, keep_cols, min_loc, origin_repo, None)
    ledger: Ledger = build_ledger(port_repo, snapshots, ref=ref, min_loc=min_loc)

    snap_ev: list[SnapshotEvidence] = []
    sides: list[_PreparedSide] = []
    struct_maps: list[dict[int, int]] = []
    cmt_maps: list[dict[int, int]] = []
    birth0_lin_to_origin: dict[int, int] = {}
    n_birth0 = len(ledger.keys[0])

    for s_idx, snap in enumerate(snapshots):
        lbl = snap.label or snap.commit[:7]
        side = _prepare_side(snap.scan, ref, bp, keep_cols, min_loc, port_repo, snap.commit)
        sides.append(side)
        j_mat = _jaccard_matrix(side.toks, origin.toks)

        m_struct, cos_s, high_cos_s, edge_pres_s, hop_mae_s = _seed_and_extend(side, origin, bp)
        m_multi, cos_m, _, _, _ = _seed_and_extend(side, origin, bp, j_mat=j_mat, fgw_comment_w=0.25, ext_comment_w=1.0)
        m_cmt = _comment_rbh(j_mat, side.toks, origin.toks)
        struct_maps.append(m_struct)
        cmt_maps.append(m_cmt)

        key_to_lin_s = {k: int(ledger.ids[s_idx][i]) for i, k in enumerate(ledger.keys[s_idx])}
        if s_idx == 0:
            for pu0, ru in m_struct.items():
                r0 = int(side.view.rows[pu0])
                k0 = (str(side.ft.path[r0]), str(side.ft.name[r0]))
                if k0 in key_to_lin_s:
                    birth0_lin_to_origin[key_to_lin_s[k0]] = ru

        retained = 0
        post_t0_cmt = 0
        for pu, ru in m_struct.items():
            rs = int(side.view.rows[pu])
            ks = (str(side.ft.path[rs]), str(side.ft.name[rs]))
            lin_id = key_to_lin_s.get(ks)
            if lin_id is not None and birth0_lin_to_origin.get(lin_id) == ru:
                retained += 1
        if s_idx > 0:
            for pu in m_cmt:
                rs = int(side.view.rows[pu])
                ks = (str(side.ft.path[rs]), str(side.ft.name[rs]))
                lin_id = key_to_lin_s.get(ks)
                if lin_id is not None and lin_id >= n_birth0:
                    post_t0_cmt += 1

        denom = max(1, min(side.view.n, origin.view.n))
        snap_ev.append(SnapshotEvidence(
            label=lbl,
            commit=snap.commit,
            n_port=side.view.n,
            n_origin=origin.view.n,
            n_commented_port=sum(1 for t in side.toks if len(t) >= 3),
            struct_matched=len(m_struct),
            struct_high_cos=high_cos_s,
            struct_mean_cos=cos_s,
            struct_coverage=len(m_struct) / denom,
            multimodal_matched=len(m_multi),
            multimodal_mean_cos=cos_m,
            multimodal_coverage=len(m_multi) / denom,
            comment_rbh=len(m_cmt),
            edge_pres=edge_pres_s,
            rbh_hop_mae=hop_mae_s,
            birth0_retained=retained,
            new_births=len(ledger.births(s_idx)) if s_idx > 0 else 0,
            post_t0_cmt_rbh=post_t0_cmt,
            struct_name_ok=_count_name_ok(m_struct, side, origin) if score_names else None,
            multimodal_name_ok=_count_name_ok(m_multi, side, origin) if score_names else None,
            comment_name_ok=_count_name_ok(m_cmt, side, origin) if score_names else None,
        ))

    # Project t0 structural matches forward to HEAD via ledger lineage IDs
    key_to_linH = {k: int(ledger.ids[-1][i]) for i, k in enumerate(ledger.keys[-1])}
    lin_to_origin = birth0_lin_to_origin

    m_direct = struct_maps[-1]
    m_proj: dict[int, int] = {}
    for puH in range(sides[-1].view.n):
        rH = int(sides[-1].view.rows[puH])
        kH = (str(sides[-1].ft.path[rH]), str(sides[-1].ft.name[rH]))
        lin_id = key_to_linH.get(kH)
        if lin_id is not None and lin_id in lin_to_origin:
            m_proj[puH] = lin_to_origin[lin_id]

    m_confirmed = {pu: ru for pu, ru in m_direct.items() if m_proj.get(pu) == ru}
    m_union = dict(m_direct)
    for pu, ru in m_proj.items():
        m_union.setdefault(pu, ru)

    # Isotonic decay curve:
    # - With score_names=True: pure-structural name accuracy
    # - Blind (score_names=False): edge-preserving structural rate (edge_pres * struct_coverage)
    raw_rates = [
        (ev.struct_name_ok / max(1, ev.struct_matched)) if (score_names and ev.struct_name_ok is not None)
        else (ev.edge_pres * ev.struct_coverage)
        for ev in snap_ev
    ]
    fitted = isotonic_nonincreasing(raw_rates)
    birth_r = fitted[0]
    head_r = fitted[-1]
    drop = birth_r - head_r
    clears_null = any(
        (ev.rbh_hop_mae < 0.095) or (ev.edge_pres >= 0.65) or (ev.comment_rbh >= 20)
        for ev in snap_ev
    )
    if (score_names and birth_r < 0.05 and head_r < 0.05) or (not score_names and not clears_null):
        verdict = "unrelated_or_clean_room"
    elif drop >= 0.02:
        verdict = "one_time_port"
    else:
        verdict = "active_fork_or_mirror"

    decay = IsotonicDecay(
        labels=[ev.label for ev in snap_ev],
        raw_rates=raw_rates,
        fitted_rates=fitted,
        birth_rate=birth_r,
        head_rate=head_r,
        decay_drop=drop,
        verdict=verdict,
    )

    return PortEvidenceCard(
        origin_label=origin_label,
        port_label=port_label or port_repo.name,
        snapshots=snap_ev,
        ledger_lineages=ledger.n_lineages,
        ledger_links_per_step=[len(lk) for lk in ledger.links],
        head_direct_n=len(m_direct),
        head_projected_n=len(m_proj),
        head_confirmed_n=len(m_confirmed),
        head_union_n=len(m_union),
        decay=decay,
        head_direct_ok=_count_name_ok(m_direct, sides[-1], origin) if score_names else None,
        head_projected_ok=_count_name_ok(m_proj, sides[-1], origin) if score_names else None,
        head_confirmed_ok=_count_name_ok(m_confirmed, sides[-1], origin) if score_names else None,
        head_union_ok=_count_name_ok(m_union, sides[-1], origin) if score_names else None,
    )


def format_evidence_card(card: PortEvidenceCard) -> str:
    """Format a :class:`PortEvidenceCard` as a compact multi-line report."""
    lines = [
        f"=== PORT EVIDENCE CARD: {card.port_label} <- {card.origin_label} ===",
        f"Ledger: {card.ledger_lineages} lineages | step links={card.ledger_links_per_step}",
        "Snapshots:",
    ]
    for ev in card.snapshots:
        s_str = f"{ev.struct_name_ok}/{ev.struct_matched}" if ev.struct_name_ok is not None else f"{ev.struct_matched}"
        m_str = f"{ev.multimodal_name_ok}/{ev.multimodal_matched}" if ev.multimodal_name_ok is not None else f"{ev.multimodal_matched}"
        c_str = f"{ev.comment_name_ok}/{ev.comment_rbh}" if ev.comment_name_ok is not None else f"{ev.comment_rbh}"
        lines.append(
            f"  {ev.label:18s} | n={ev.n_port:4d} | struct={s_str:>8s} (edge={ev.edge_pres:.3f}, hop_mae={ev.rbh_hop_mae:.3f}) "
            f"| +cmt={m_str:>8s} | cmt_rbh={c_str:>7s} (post_t0={ev.post_t0_cmt_rbh:2d})"
        )
    if card.head_direct_ok is not None:
        lines.append(
            f"HEAD Temporal Projection: direct={card.head_direct_ok}/{card.head_direct_n}, "
            f"projected={card.head_projected_ok}/{card.head_projected_n}, "
            f"confirmed={card.head_confirmed_ok}/{card.head_confirmed_n}, "
            f"union={card.head_union_ok}/{card.head_union_n}"
        )
    else:
        lines.append(
            f"HEAD Temporal Projection: direct={card.head_direct_n}, projected={card.head_projected_n}, "
            f"confirmed={card.head_confirmed_n}, union={card.head_union_n}"
        )
    raw_s = " -> ".join(f"{x:.3f}" for x in card.decay.raw_rates)
    fit_s = " -> ".join(f"{x:.3f}" for x in card.decay.fitted_rates)
    lines.append(
        f"Isotonic Decay: raw=[{raw_s}] | fitted=[{fit_s}] | drop={card.decay.decay_drop:+.3f} | verdict={card.decay.verdict}"
    )
    return "\n".join(lines)
