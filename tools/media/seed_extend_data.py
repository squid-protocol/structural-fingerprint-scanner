#!/usr/bin/env python3
"""Record every step of Seed-and-Extend + the keyword lattice on two real scans, for the explainer film.

Runs the same steps as ``lattice.seed_and_extend_with_lattice`` (FGW transport plan -> reciprocal-best seeds ->
strongest ``seed_frac`` kept as anchors -> median offset -> ``hops`` rounds of call-graph extension -> 32-family
keyword check on the matched pairs), keeps the intermediates the library throws away, and refuses to write anything
unless its final pairs and summary equal the library's own result. Then it picks one small real neighbourhood of
the call graph to draw, and grades every pair against an answer key built from normalized function names (never
used for matching).

    python tools/media/seed_extend_data.py --a <scan dir or db> --b <scan dir or db> --out docs/media/seed-extend.json

Needs numpy and scipy (the scanner's own dependencies).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from structural_fingerprint import blind as BL
from structural_fingerprint import lattice as LT
from structural_fingerprint.reference import load_reference
from structural_fingerprint.scan_io import load_scan, resolve_scans

SIGNAL_LABEL = {
    "struct_branch": "branches",
    "struct_linear": "linear flow",
    "struct_args": "arguments",
    "struct_func_start": "function decl",
    "struct_class_start": "class / struct",
    "def_safety": "guards",
    "state_safety_neg": "safety bypass",
    "arch_io": "I/O",
    "arch_api": "public surface",
    "state_flux": "state mutation",
    "def_test": "tests",
    "arch_concurrency": "concurrency",
    "arch_ui_framework": "UI",
    "struct_closures": "closures",
    "arch_globals": "globals",
    "struct_decorators": "decorators",
    "struct_generics": "generics",
    "struct_comprehensions": "comprehensions",
    "arch_scientific": "math / scientific",
    "state_heat_triggers": "hot-path triggers",
    "arch_import": "imports",
    "struct_macros": "macros",
    "state_pointers": "pointers",
    "state_memory_alloc": "allocation",
    "state_print_hits": "debug print",
    "state_cast_hits": "casts",
    "state_bailout_hits": "panic / abort",
    "bitwise_ops": "bitwise ops",
    "def_freeze_hits": "immutability",
    "def_cleanup": "cleanup",
    "def_encapsulation": "encapsulation",
    "def_listeners": "listeners",
}


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def same_function(name_a: str, name_b: str) -> bool:
    """Answer key: same name ignoring case and underscores, also after dropping one short module prefix the port
    added (gore writes linuxdoom's ``IdentifyVersion`` as ``d_IdentifyVersion``)."""
    na = norm_name(name_a)
    return na == norm_name(name_b) or ("_" not in name_a and na == norm_name(re.sub(r"^[a-z]{1,3}_", "", name_b)))


def fr_layout(n: int, edges: list[tuple[int, int]], seed: int = 3, iters: int = 400) -> np.ndarray:
    """Fruchterman-Reingold with a weak pull to the centre (keeps small components near the main one);
    deterministic. Returns positions scaled into [-1, 1]."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, 2))
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    k = 2.0 / np.sqrt(n)
    temp = 0.1
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=2), 1e-4)
        disp = ((k * k / dist**2)[:, :, None] * delta).sum(axis=1)
        if len(e):
            d = pos[e[:, 0]] - pos[e[:, 1]]
            ln = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-4)
            f = d * ln / k
            np.add.at(disp, e[:, 0], -f)
            np.add.at(disp, e[:, 1], f)
        disp -= pos * 0.6 * k * np.sqrt(n) / 4
        ln = np.maximum(np.linalg.norm(disp, axis=1, keepdims=True), 1e-9)
        pos += disp / ln * np.minimum(ln, temp)
        temp = max(0.002, temp * 0.985)
    lo, hi = np.percentile(pos, 1, axis=0), np.percentile(pos, 99, axis=0)
    return np.clip((pos - (lo + hi) / 2) / ((hi - lo) / 2), -1, 1)


def run_steps(sc_a, sc_b, ref, *, use_delta=True, min_loc=3, alpha=0.5, eps=0.05, outer=40, hops=3, seed_frac=0.15):
    """The library's algorithm, step by step, keeping what the film needs."""
    bp = BL.BlindParams()
    A = LT._prep_repo(sc_a, ref, bp, min_loc)
    B = LT._prep_repo(sc_b, ref, bp, min_loc)
    cost = cdist(A["xd"], B["xd"], metric="euclidean")
    cost /= float(np.median(cost)) + 1e-12
    T = BL.fused_gw(cost, A["hm"], B["hm"], alpha=alpha, eps=eps, outer=outer)
    best_r = np.argmax(T, axis=1)
    best_q = np.argmax(T, axis=0)
    seeds = [(qi, int(best_r[qi]), float(T[qi, best_r[qi]])) for qi in range(A["n"]) if best_q[best_r[qi]] == qi]
    seeds.sort(key=lambda x: -x[2])
    k_seed = max(3, min(len(seeds), round(seed_frac * len(seeds))))
    top = seeds[:k_seed]
    delta = np.median(A["comp"][[s[0] for s in top]] - B["comp"][[s[1] for s in top]], axis=0)
    if not use_delta:
        delta = np.zeros_like(delta)
    matched = {qi: ri for qi, ri, _ in top}
    tier_of = {qi: 0 for qi, _, _ in top}
    used_b = {ri for _, ri, _ in top}
    b_shift = B["comp"] + delta
    b_snorm = b_shift / np.maximum(np.linalg.norm(b_shift, axis=1, keepdims=True), 1e-9)
    b_norm = B["cnorm"]
    why: dict[int, dict] = {}
    for hop in range(1, hops + 1):
        cand = []
        for qi in range(A["n"]):
            if qi in matched:
                continue
            qn = [u for u in A["nbrs"][qi] if u in matched]
            if not qn:
                continue
            rc_cnt: dict[int, int] = defaultdict(int)
            via: dict[int, list[int]] = defaultdict(list)
            for u in qn:
                for rc in B["nbrs"][matched[u]]:
                    if rc not in used_b:
                        rc_cnt[rc] += 1
                        via[rc].append(u)
            for rc, cnt in rc_cnt.items():
                cos = float(np.dot(A["cnorm"][qi], b_snorm[rc]))
                cand.append((cos + 0.1 * cnt, qi, rc, cos, cnt, via[rc], len(rc_cnt)))
        cand.sort(key=lambda x: -x[0])
        for score, qi, rc, cos, cnt, via_u, n_opts in cand:
            if qi in matched or rc in used_b:
                continue
            matched[qi] = rc
            tier_of[qi] = hop
            used_b.add(rc)
            why[qi] = {
                "score": score,
                "cos_shifted": cos,
                "cos_unshifted": float(np.dot(A["cnorm"][qi], b_norm[rc])),
                "shared_anchors": cnt,
                "via": [int(u) for u in via_u],
                "options": n_opts,
            }
    return A, B, T, seeds, top, delta, matched, tier_of, why


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="original: scan dir or *_galaxy_master.db")
    ap.add_argument("--b", required=True, help="port: scan dir or *_galaxy_master.db")
    ap.add_argument("--lens", default="portable")
    ap.add_argument("--anchors", nargs="+", help="names of the anchors the neighbourhood starts from")
    ap.add_argument("--caps", default="2,5,7,7", help="max functions drawn per tier (anchors, hop 1, 2, 3)")
    ap.add_argument("--summary-only", action="store_true", help="record the totals only (a contrast pair), no drawing")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    ref = load_reference(a.lens)
    da, db = resolve_scans([Path(a.a)])[0], resolve_scans([Path(a.b)])[0]
    sc_a, sc_b = load_scan(da, ref.signal_columns), load_scan(db, ref.signal_columns)
    A, B, T, seeds, top, delta, matched, tier_of, why = run_steps(sc_a, sc_b, ref)

    lib = LT.seed_and_extend_with_lattice(sc_a, sc_b, ref, include_matches=True)
    lib_pairs = {(m.row_a, m.row_b, m.tier) for m in lib.matches}
    ours = {(int(A["rows"][q]), int(B["rows"][r]), tier_of[q]) for q, r in matched.items()}
    if ours != lib_pairs or lib.t1_n != len(top):
        raise SystemExit("refusing: step-by-step replay differs from seed_and_extend_with_lattice")

    verdict = {(m.row_a, m.row_b): (m.sufficient, m.evidence) for m in lib.matches}

    def suff(q, r):
        return verdict[(int(A["rows"][q]), int(B["rows"][r]))]

    name_a = [sc_a.func_name[int(r)] for r in A["rows"]]
    name_b = [sc_b.func_name[int(r)] for r in B["rows"]]
    file_a = [sc_a.file_path[int(sc_a.func_file[int(r)])] for r in A["rows"]]
    file_b = [sc_b.file_path[int(sc_b.func_file[int(r)])] for r in B["rows"]]
    keys_full = {norm_name(x) for x in name_b}
    keys_stripped = {norm_name(re.sub(r"^[a-z]{1,3}_", "", x)) for x in name_b}

    def grade(q, r):
        n = name_a[q]
        if norm_name(n) not in keys_full and ("_" in n or norm_name(n) not in keys_stripped):
            return "unkeyed"
        return "right" if same_function(n, name_b[r]) else "wrong"

    tiers = []
    for t in range(4):
        qs = [q for q in matched if tier_of[q] == t]
        g = Counter(grade(q, matched[q]) for q in qs)
        tiers.append({"tier": t, "n": len(qs), "right": g["right"], "wrong": g["wrong"], "unkeyed": g["unkeyed"]})

    summary = {
        "a": sc_a.name,
        "b": sc_b.name,
        "functions_a": int(A["n"]),
        "functions_b": int(B["n"]),
        "reciprocal_best": len(seeds),
        "anchors": len(top),
        "matched": len(matched),
        "right": sum(t["right"] for t in tiers),
        "coverage": round(lib.se_coverage, 4),
        "fgw_cost": round(lib.fgw_cost, 4),
        "kw_jsd": round(lib.kw_jsd, 4),
        "arch_jsd": round(lib.arch_jsd, 4),
        "matched_kw_cos": round(lib.matched_kw_cos, 4),
        "matched_kw_dom_agree": round(lib.matched_kw_dom_agree, 4),
        "matched_arch_agree": round(lib.matched_arch_agree, 4),
        "post_delta_res": round(lib.post_delta_res, 4),
        "tiers": tiers,
    }
    sg = Counter(grade(q, r) for q, r in matched.items() if suff(q, r)[0])
    summary["evidence"] = {
        "fdr": lib.evidence.fdr,
        "draws": lib.evidence.draws,
        "est_fdr": round(lib.evidence.est_fdr, 4),
        "threshold": round(lib.evidence.threshold, 4),
        "decoy_passing_per_draw": round(lib.evidence.decoy_passing, 2),
        "sufficient": lib.evidence.n_sufficient,
        "right": sg["right"],
        "wrong": sg["wrong"],
        "unkeyed": sg["unkeyed"],
    }

    def n_right(pairs):
        return sum(same_function(name_a[q], name_b[r]) for q, r in pairs)

    cost = cdist(A["xd"], B["xd"])
    br, bq = cost.argmin(1), cost.argmin(0)
    shape_only = [(q, int(br[q])) for q in range(A["n"]) if bq[br[q]] == q]
    no_off = run_steps(sc_a, sc_b, ref, use_delta=False)[6]
    ablation = {
        "shape_only_pairs": len(shape_only),
        "shape_only_right": n_right(shape_only),
        "fgw_reciprocal_right": n_right([(q, r) for q, r, _ in seeds]),
        "extend_right": n_right(matched.items()),
        "extend_no_offset_pairs": len(no_off),
        "extend_no_offset_right": n_right(no_off.items()),
    }
    summary["ablation"] = ablation
    if a.summary_only:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": summary}, indent=1))
        print(json.dumps({k: v for k, v in summary.items() if k != "tiers"}), tiers)
        return 0

    # --- the neighbourhood to draw ------------------------------------------------------------------------------
    caps = [int(x) for x in a.caps.split(",")]
    by_name = {n: i for i, n in enumerate(name_a)}
    # every extended match hangs off the matched neighbour(s) it was reached through: a forest rooted at anchors
    children: dict[int, list[int]] = defaultdict(list)
    for q, w in why.items():
        for u in w["via"]:
            if tier_of[u] == tier_of[q] - 1:
                children[u].append(q)

    def tree(roots):
        pick, level = {q: 0 for q in roots}, list(roots)
        for t in range(1, 4):
            kids = [u for q in level for u in children[q] if u not in pick]
            kids = list(dict.fromkeys(kids))
            right = [u for u in kids if grade(u, matched[u]) == "right"]
            other = [u for u in kids if grade(u, matched[u]) != "right"]
            n_other = min(len(other), max(1, caps[t] // 3)) if t >= 2 else min(len(other), 1)
            take = right[: caps[t] - n_other] + other[:n_other]
            for u in take:
                pick[u] = t
            level = take
        return pick

    if a.anchors:
        roots = [by_name[n] for n in a.anchors]
        if any(tier_of.get(q) != 0 for q in roots):
            raise SystemExit("refusing: a requested anchor is not an anchor")
    else:
        good = [q for q, _, _ in top if grade(q, matched[q]) == "right"]
        best = max(
            (
                (x, y)
                for i, x in enumerate(good)
                for y in good[i + 1 :]
                if y in A["nbrs"][x] or A["nbrs"][x] & A["nbrs"][y]
            ),
            key=lambda xy: (sum(1 for t in tree(list(xy)).values() if t == 3), len(tree(list(xy)))),
        )
        roots = list(best)
    pick = tree(roots)
    # a few neighbours that never got a partner, so the film shows extension can stop
    loose = []
    for q in list(pick):
        for u in A["nbrs"][q]:
            if u not in matched and u not in pick and len(loose) < 3:
                loose.append(u)
    qa = list(pick) + loose
    qb = [matched[q] for q in pick]
    # port-side neighbours of drawn partners that stayed unmatched, so the right panel is not just a mirror
    used_b = set(matched.values())
    for r in list(qb):
        for v in B["nbrs"][r]:
            if v not in qb and v not in used_b and len(qb) < len(pick) + 3:
                qb.append(v)
    ia = {q: k for k, q in enumerate(qa)}
    ib = {r: k for k, r in enumerate(qb)}
    edges_a = sorted({tuple(sorted((ia[u], ia[v]))) for u in qa for v in A["nbrs"][u] if v in ia and u != v})
    edges_b = sorted({tuple(sorted((ib[u], ib[v]))) for u in qb for v in B["nbrs"][u] if v in ib and u != v})

    feat = [n for n in ref.feature_names if n not in ("log_complexity", "func_internal_density")]
    sig_idx = [k for k, n in enumerate(feat) if n.startswith("log_density_")]
    sig_names = [feat[k].removeprefix("log_density_") for k in sig_idx]

    def kw_profile(sc, row):
        s = np.asarray(sc.signals[row], dtype=float)
        return (s / max(1.0, s.sum())).round(4).tolist()

    pairs = []
    for q in pick:
        r = matched[q]
        pairs.append(
            {
                "a": ia[q],
                "b": ib[r],
                "tier": tier_of[q],
                "grade": grade(q, r),
                "kw_cos": round(float(np.dot(A["kw_norm"][q], B["kw_norm"][r])), 3),
                "arch_agree": bool(A["arch"][q] == B["arch"][r]),
                "transport_rank": int(np.argsort(-T[q]).tolist().index(r)) + 1,
                "sufficient": suff(q, r)[0],
                "evidence": round(suff(q, r)[1], 3),
                "why": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in why.get(q, {}).items()}
                | ({"via": [ia[u] for u in why[q]["via"] if u in ia]} if q in why else {}),
            }
        )

    reciprocal = {qi for qi, _, _ in seeds}

    def full_map():
        """Every original function, laid out on its own call graph, with its final verdict."""
        e_all = sorted({(min(u, v), max(u, v)) for u in range(A["n"]) for v in A["nbrs"][u] if u != v})
        # functions with no resolved call go in a band along the bottom; the rest on their call graph above it
        deg = np.zeros(A["n"], dtype=np.int64)
        for u, v in e_all:
            deg[u] += 1
            deg[v] += 1
        con = np.flatnonzero(deg > 0)
        iso = np.flatnonzero(deg == 0)
        idx = {int(q): k for k, q in enumerate(con)}
        sub = fr_layout(len(con), [(idx[u], idx[v]) for u, v in e_all])
        xy = np.zeros((A["n"], 2))
        xy[con, 0] = sub[:, 0]
        xy[con, 1] = sub[:, 1] * 0.84 - 0.16
        cols = max(1, int(np.ceil(len(iso) / 2)))
        for k, q in enumerate(iso):
            xy[q] = (-1 + 2 * (k % cols + 0.5) / cols, 0.86 + 0.1 * (k // cols))

        def status(q):
            if q not in matched:
                return "unmatched"
            if not suff(q, matched[q])[0]:
                return "insufficient"
            return {"right": "confirmed", "wrong": "wrong"}.get(grade(q, matched[q]), "unchecked")

        st = [status(q) for q in range(A["n"])]
        return {
            "xy": xy.round(4).tolist(),
            "status": st,
            "edges": e_all,
            "drawn": [int(q) for q in qa],
            "no_calls": len(iso),
            "counts": dict(Counter(st)),
        }

    out = {
        "source": {
            "a": {
                "name": sc_a.name,
                "db": da.name,
                "engine": sc_a.metadata.get("engine_identity"),
                "remote": sc_a.metadata.get("remote"),
                "functions": int(A["n"]),
                "language": Counter(sc_a.file_language).most_common(1)[0][0],
            },
            "b": {
                "name": sc_b.name,
                "db": db.name,
                "engine": sc_b.metadata.get("engine_identity"),
                "remote": sc_b.metadata.get("remote"),
                "functions": int(B["n"]),
                "language": Counter(sc_b.file_language).most_common(1)[0][0],
            },
            "lens": a.lens,
            "spec": "seed_and_extend_with_lattice defaults: min_loc 3, alpha 0.5, eps 0.05, outer 40, hops 3, seed_frac 0.15",
        },
        "summary": summary,
        "ablation": ablation,
        "delta_size": round(float(delta[feat.index("log_loc")]), 4),
        "delta": [
            {"signal": s, "label": SIGNAL_LABEL.get(s, s), "shift": round(float(delta[k]), 4)}
            for k, s in zip(sig_idx, sig_names, strict=True)
        ],
        "nodes_a": [
            {
                "name": name_a[q],
                "file": file_a[q],
                "tier": pick.get(q),
                "reciprocal": q in reciprocal,
                "kw": kw_profile(sc_a, int(A["rows"][q])),
            }
            for q in qa
        ],
        "nodes_b": [{"name": name_b[r], "file": file_b[r], "kw": kw_profile(sc_b, int(B["rows"][r]))} for r in qb],
        "edges_a": edges_a,
        "full_map": full_map(),
        "edges_b": edges_b,
        "pairs": pairs,
        "signals": [SIGNAL_LABEL.get(c, c) for c in ref.signal_columns],
        "signal_columns": list(ref.signal_columns),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    s = out["summary"]
    print(
        f"{sc_a.name} -> {sc_b.name}: {s['anchors']} anchors of {s['reciprocal_best']} reciprocal best, {s['matched']} matched"
    )
    for t in tiers:
        print(f"  tier {t['tier']}: {t['n']} (right {t['right']}, wrong {t['wrong']}, unkeyed {t['unkeyed']})")
    for p in pairs:
        print(
            f"  [{p['tier']}] {name_a[qa[p['a']]]:28s} -> {name_b[qb[p['b']]]:28s} {p['grade']:7s} kw_cos {p['kw_cos']}  {p['why']}"
        )
    print(f"  drawn: {len(qa)} original, {len(qb)} port, {len(edges_a)}/{len(edges_b)} calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
