"""Two-layer structural comparison: Call-Graph Seed-and-Extend + 32-Keyword-Family Lattice."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.distance import cdist

from . import blind as BL
from . import compare as C
from . import features as F
from . import twins as TW
from .fingerprint import build
from .reference import Reference, load_reference
from .scan_io import ScanData


@dataclass(frozen=True)
class MatchedPair:
    """One-to-one function alignment produced by Layer 1 (Seed-and-Extend)."""

    row_a: int
    row_b: int
    tier: int
    kw_cos: float
    delta_res: float
    # evidence distance (lower = stronger; decoy-scaled, see ``evidence_gate``) and its verdict
    evidence: float = float("nan")
    sufficient: bool = False


@dataclass(frozen=True)
class EvidenceGate:
    """Per-pair evidence check calibrated on decoys (target-decoy competition, as in ``twins``).

    A pair's evidence distance adds its keyword-mix disagreement (1 - cosine of the 32-family mixes) and its
    shape residual after the language offset, each divided by its median over decoy pairs, so neither scale is
    tuned by hand. Decoy pairs keep every matched (a, b) slot but give b a decoy function: each of b's measurements
    shuffled among b-side functions of the same engine archetype. The threshold is the largest distance at which
    (decoy pairs passing / draws + 1) / (real pairs passing) <= ``fdr``. With fewer than 1/fdr real pairs nothing
    can pass, by design: small or unrelated matches report no sufficient evidence rather than chance ones.

    The default ``fdr`` is 2%, not ``twins``' 1%: at 1% at least 100 pairs must pass with no decoy among them, which
    certified nothing on linuxdoom -> gore. The rate is against random functions of the same kind; it understates
    mistakes between near-identical siblings (``G_WriteDemoTiccmd`` matched to ``g_ReadDemoTiccmd``). On that pair,
    11% of certified pairs were not the same function by name."""

    fdr: float
    draws: int
    threshold: float
    est_fdr: float
    n_sufficient: int
    decoy_passing: float  # mean decoy pairs under the threshold, per draw


@dataclass(frozen=True)
class TwoLayerResult:
    """Output of Layer 1 (Seed-and-Extend) and Layer 2 (Keyword & Archetype Lattice)."""

    repo_a: str
    repo_b: str
    n_a: int
    n_b: int
    fgw_cost: float
    t1_n: int
    se_n: int
    se_coverage: float
    kw_jsd: float
    arch_jsd: float
    matched_kw_cos: float
    matched_kw_dom_agree: float
    matched_arch_agree: float
    post_delta_res: float
    matches: tuple[MatchedPair, ...] | None = None
    evidence: EvidenceGate | None = None

    @property
    def n_sufficient(self) -> int:
        return self.evidence.n_sufficient if self.evidence else 0

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["matches"] is None:
            d.pop("matches")
        if d["evidence"] is None:
            d.pop("evidence")
        return d


@dataclass(frozen=True)
class LanguageDeltaModel:
    """Additive language-translation vectors (delta_lang) in 32-keyword-family space."""

    lang_centroids: dict[str, np.ndarray]
    global_centroid: np.ndarray

    def center(self, dist: np.ndarray, lang: str) -> np.ndarray:
        """Subtract language offset (lang_centroid - global_centroid) and renormalize."""
        v = np.asarray(dist, dtype=np.float64)
        key = lang.lower()
        if key in self.lang_centroids:
            v = np.clip(v - (self.lang_centroids[key] - self.global_centroid), 0.0, None)
        tot = float(v.sum())
        return v / tot if tot > 1e-12 else np.zeros_like(v)

    def predict_translation(self, dist_src: np.ndarray, src_lang: str, dst_lang: str) -> np.ndarray:
        """Predict target-language 32-keyword distribution via additive delta."""
        v = np.asarray(dist_src, dtype=np.float64)
        s_key, d_key = src_lang.lower(), dst_lang.lower()
        if s_key in self.lang_centroids and d_key in self.lang_centroids:
            v = np.clip(v + (self.lang_centroids[d_key] - self.lang_centroids[s_key]), 0.0, None)
        tot = float(v.sum())
        return v / tot if tot > 1e-12 else np.zeros_like(v)

    def centered_distance(self, dist_a: np.ndarray, lang_a: str, dist_b: np.ndarray, lang_b: str) -> float:
        """Jensen-Shannon distance after subtracting per-language translation vectors."""
        return keyword_lattice_distance(self.center(dist_a, lang_a), self.center(dist_b, lang_b))

    def to_dict(self) -> dict:
        return {
            "lang_centroids": {k: v.tolist() for k, v in sorted(self.lang_centroids.items())},
            "global_centroid": self.global_centroid.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> LanguageDeltaModel:
        return cls(
            lang_centroids={k: np.asarray(v, dtype=np.float64) for k, v in data["lang_centroids"].items()},
            global_centroid=np.asarray(data["global_centroid"], dtype=np.float64),
        )


def fit_language_deltas(distributions_by_lang: dict[str, list[np.ndarray]]) -> LanguageDeltaModel:
    """Fit per-language centroids and global centroid from 32-keyword-family distributions."""
    centroids = {
        lang.lower(): np.mean(np.asarray(dists, dtype=np.float64), axis=0)
        for lang, dists in distributions_by_lang.items()
        if dists
    }
    if not centroids:
        raise ValueError("distributions_by_lang must contain at least one non-empty language")
    global_centroid = np.mean(list(centroids.values()), axis=0)
    return LanguageDeltaModel(lang_centroids=centroids, global_centroid=global_centroid)


@dataclass(frozen=True)
class StyleAxis:
    """Linear attribution axis in language-centered 32-keyword-family space."""

    weights: np.ndarray
    pos_label: str = "pos"
    neg_label: str = "neg"

    def score(self, centered_dist: np.ndarray) -> float:
        """Project a language-centered distribution onto the attribution axis."""
        return float(np.dot(np.asarray(centered_dist, dtype=np.float64), self.weights))

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "pos_label": self.pos_label,
            "neg_label": self.neg_label,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StyleAxis:
        return cls(
            weights=np.asarray(data["weights"], dtype=np.float64),
            pos_label=str(data.get("pos_label", "pos")),
            neg_label=str(data.get("neg_label", "neg")),
        )


def fit_style_axis(
    pos_centered: list[np.ndarray],
    neg_centered: list[np.ndarray],
    *,
    pos_label: str = "pos",
    neg_label: str = "neg",
) -> StyleAxis:
    """Fit difference-of-means attribution vector in language-centered keyword space."""
    if not pos_centered or not neg_centered:
        raise ValueError("both pos_centered and neg_centered must be non-empty")
    w = np.mean(np.asarray(pos_centered, dtype=np.float64), axis=0) - np.mean(
        np.asarray(neg_centered, dtype=np.float64), axis=0
    )
    return StyleAxis(weights=w, pos_label=pos_label, neg_label=neg_label)


def keyword_lattice_profile(
    signals: np.ndarray,
    rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-function unit-norm keyword-family vectors, dominant indices, and repo distribution."""
    sigs = np.asarray(signals, dtype=np.float64)
    if rows is not None:
        sigs = sigs[rows]
    if len(sigs) == 0:
        n_sig = sigs.shape[1] if sigs.ndim == 2 else 0
        return np.zeros((0, n_sig)), np.zeros(0, dtype=np.int64), np.zeros(n_sig)
    sums = np.maximum(sigs.sum(axis=1, keepdims=True), 1.0)
    kw_comp = sigs / sums
    kw_norm = kw_comp / np.maximum(np.linalg.norm(kw_comp, axis=1, keepdims=True), 1e-9)
    kw_dom = np.argmax(sigs, axis=1)
    repo_dist = kw_comp.mean(axis=0)
    tot = float(repo_dist.sum())
    if tot > 0:
        repo_dist = repo_dist / tot
    return kw_norm, kw_dom, repo_dist


def keyword_lattice_distance(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    """Jensen-Shannon distance (0..1, base 2) between two 32-keyword-family distributions."""
    return C._js_distance(np.asarray(dist_a, dtype=np.float64), np.asarray(dist_b, dtype=np.float64))


def _prep_repo(sc: ScanData, ref: Reference, bp: BL.BlindParams, min_loc: int) -> dict:
    ft = build(sc, ref, min_loc=min_loc, keyword_mode="composition").functions
    bv = BL.blind_view(ft)
    n_arch = len(ref.archetype_names)
    xd = BL.descriptors(bv, n_arch, bp.use_graph, bp.use_measures, None, bp.geometry_only)
    hm = BL.hop_matrix(bv, bp.hop_cap)
    # bv.rows index the min_loc-filtered function table; scan arrays need scan rows.
    scan_rows = np.flatnonzero(sc.loc >= min_loc)[bv.rows]
    keep_cols = [
        i for i, n in enumerate(ref.feature_names)
        if n not in ("log_complexity", "func_internal_density")
    ]
    comp_all = F.raw_features(
        ref, sc.loc, sc.complexity, sc.args, sc.keyword_density, sc.signals,
        keyword_mode="composition",
    )[:, keep_cols]
    comp = comp_all[scan_rows]
    cnorm = comp / np.maximum(np.linalg.norm(comp, axis=1, keepdims=True), 1e-9)
    nbrs: dict[int, set[int]] = defaultdict(set)
    for u, v in bv.edges.tolist():
        if u != v:
            nbrs[u].add(v)
            nbrs[v].add(u)
    arch = np.asarray(ft.archetype[bv.rows], dtype=np.int64)
    arch_cnt = np.bincount(arch, minlength=n_arch).astype(np.float64)
    arch_dist = arch_cnt / max(1.0, float(arch_cnt.sum()))
    kw_norm, kw_dom, kw_dist = keyword_lattice_profile(sc.signals, scan_rows)
    return {
        "name": sc.name,
        "rows": scan_rows,
        "n": len(bv.rows),
        "xd": xd,
        "hm": hm,
        "comp": comp,
        "cnorm": cnorm,
        "nbrs": nbrs,
        "arch": arch,
        "arch_dist": arch_dist,
        "kw_norm": kw_norm,
        "kw_dom": kw_dom,
        "kw_dist": kw_dist,
        "keep_cols": keep_cols,
        "raw": {
            "loc": np.asarray(sc.loc)[scan_rows],
            "complexity": np.asarray(sc.complexity)[scan_rows],
            "args": np.asarray(sc.args)[scan_rows],
            "keyword_density": np.asarray(sc.keyword_density)[scan_rows],
            "signals": np.asarray(sc.signals)[scan_rows],
        },
    }


def _decoy_side(B: dict, ref: Reference, seed_text: str) -> tuple[np.ndarray, np.ndarray]:
    """B's functions with every measurement shuffled among B functions of the same engine archetype
    (``twins._decoy``'s rule): realistic functions of each kind, but each real function destroyed.
    Returns the decoy composition features and unit keyword-family vectors."""
    raw = B["raw"]
    scalars = np.column_stack([raw["loc"], raw["complexity"], raw["args"], raw["keyword_density"]]).astype(np.float64)
    table = np.column_stack([scalars, np.asarray(raw["signals"], dtype=np.float64)])
    shuffled = TW._decoy(table, B["arch"], seed_text)
    loc, cx, args, kd = (shuffled[:, k] for k in range(4))
    sig = shuffled[:, 4:]
    comp = F.raw_features(ref, loc, cx, args, kd, sig, keyword_mode="composition")[:, B["keep_cols"]]
    kw_norm, _, _ = keyword_lattice_profile(sig)
    return comp, kw_norm


def _fdr_threshold_draws(target_d: np.ndarray, decoy_d: np.ndarray, draws: int, fdr: float) -> tuple[float, float]:
    """``twins._fdr_threshold`` with ``draws`` decoy sets pooled: decoy counts are per draw."""
    if len(target_d) == 0:
        return 0.0, 0.0
    ts = np.sort(target_d)
    ds = np.sort(decoy_d)
    n_t = np.arange(1, len(ts) + 1)
    est = (np.searchsorted(ds, ts, side="right") / max(1, draws) + 1) / n_t
    q = np.minimum.accumulate(est[::-1])[::-1]
    ok = np.flatnonzero(q <= fdr)
    if len(ok) == 0:
        return float("-inf"), float(q[0])
    k = ok[-1]
    return float(ts[k]), float(est[k])


def evidence_gate(
    A: dict,
    B: dict,
    pairs: list[tuple[int, int]],
    delta: np.ndarray,
    ref: Reference,
    *,
    fdr: float = 0.02,
    draws: int = 5,
) -> tuple[np.ndarray, EvidenceGate]:
    """Score every matched pair's evidence and certify the pairs closer than decoys allow (see ``EvidenceGate``)."""
    qa = np.array([q for q, _ in pairs], dtype=np.int64)
    rb = np.array([r for _, r in pairs], dtype=np.int64)

    def parts(comp_b, kwn_b):
        kw_gap = 1.0 - np.einsum("ij,ij->i", A["kw_norm"][qa], kwn_b[rb])
        shape_gap = np.linalg.norm(A["comp"][qa] - (comp_b[rb] + delta), axis=1)
        return kw_gap, shape_gap

    real = parts(B["comp"], B["kw_norm"])
    fake = [parts(*_decoy_side(B, ref, f"{A['name']}\0{B['name']}\0evidence\0{k}")) for k in range(draws)]
    fk = np.concatenate([f[0] for f in fake])
    fs = np.concatenate([f[1] for f in fake])
    s_kw = float(np.median(fk)) or 1e-9
    s_shape = float(np.median(fs)) or 1e-9
    target = real[0] / s_kw + real[1] / s_shape
    decoy = fk / s_kw + fs / s_shape
    thr, est = _fdr_threshold_draws(target, decoy, draws, fdr)
    ok = target <= thr
    gate = EvidenceGate(
        fdr=fdr,
        draws=draws,
        threshold=thr if np.isfinite(thr) else 0.0,
        est_fdr=est,
        n_sufficient=int(ok.sum()),
        decoy_passing=float((decoy <= thr).sum()) / max(1, draws),
    )
    return target, gate


def seed_and_extend_with_lattice(
    sc_a: ScanData,
    sc_b: ScanData,
    ref: Reference | None = None,
    *,
    min_loc: int = 3,
    alpha: float = 0.5,
    eps: float = 0.05,
    outer: int = 40,
    hops: int = 3,
    seed_frac: float = 0.15,
    include_matches: bool = False,
    evidence_fdr: float | None = 0.02,
    evidence_draws: int = 5,
) -> TwoLayerResult:
    """Run Layer 1 (FGW Seed-and-Extend) followed by Layer 2 (32-keyword & 14-archetype lattice).

    With ``evidence_fdr`` set, every matched pair is also checked against decoys (``evidence_gate``) and marked
    sufficient or not; ``None`` skips the check."""
    if ref is None:
        ref = load_reference("portable")
    bp = BL.BlindParams()
    A = _prep_repo(sc_a, ref, bp, min_loc)
    B = _prep_repo(sc_b, ref, bp, min_loc)

    kw_jsd = keyword_lattice_distance(A["kw_dist"], B["kw_dist"])
    arch_jsd = C._js_distance(A["arch_dist"], B["arch_dist"])

    if A["n"] == 0 or B["n"] == 0:
        return TwoLayerResult(
            repo_a=A["name"], repo_b=B["name"], n_a=A["n"], n_b=B["n"],
            fgw_cost=1.0, t1_n=0, se_n=0, se_coverage=0.0,
            kw_jsd=kw_jsd, arch_jsd=arch_jsd,
            matched_kw_cos=0.0, matched_kw_dom_agree=0.0,
            matched_arch_agree=0.0, post_delta_res=0.0,
        )

    cost = cdist(A["xd"], B["xd"], metric="euclidean")
    cost /= float(np.median(cost)) + 1e-12
    T = BL.fused_gw(cost, A["hm"], B["hm"], alpha=alpha, eps=eps, outer=outer)
    fgw_cost = float(np.sum(T * cost))

    best_r = np.argmax(T, axis=1)
    best_q = np.argmax(T, axis=0)
    seeds = [(qi, int(best_r[qi]), float(T[qi, best_r[qi]]))
             for qi in range(A["n"]) if best_q[best_r[qi]] == qi]
    seeds.sort(key=lambda x: -x[2])

    if not seeds:
        return TwoLayerResult(
            repo_a=A["name"], repo_b=B["name"], n_a=A["n"], n_b=B["n"],
            fgw_cost=fgw_cost, t1_n=0, se_n=0, se_coverage=0.0,
            kw_jsd=kw_jsd, arch_jsd=arch_jsd,
            matched_kw_cos=0.0, matched_kw_dom_agree=0.0,
            matched_arch_agree=0.0, post_delta_res=0.0,
        )

    k_seed = max(3, min(len(seeds), int(round(seed_frac * len(seeds)))))
    top_seeds = seeds[:k_seed]
    delta = np.median(
        A["comp"][[s[0] for s in top_seeds]] - B["comp"][[s[1] for s in top_seeds]],
        axis=0,
    )
    matched = {qi: ri for qi, ri, _ in top_seeds}
    tier_of = {qi: 0 for qi, _, _ in top_seeds}
    used_b = {ri for _, ri, _ in top_seeds}
    b_shifted = B["comp"] + delta
    b_snorm = b_shifted / np.maximum(np.linalg.norm(b_shifted, axis=1, keepdims=True), 1e-9)

    for hop_idx in range(1, hops + 1):
        cand = []
        for qi in range(A["n"]):
            if qi in matched:
                continue
            qn = [u for u in A["nbrs"][qi] if u in matched]
            if not qn:
                continue
            rc_cnt: dict[int, int] = defaultdict(int)
            for u in qn:
                for rc in B["nbrs"][matched[u]]:
                    if rc not in used_b:
                        rc_cnt[rc] += 1
            for rc, cnt in rc_cnt.items():
                cand.append((float(np.dot(A["cnorm"][qi], b_snorm[rc])) + 0.1 * cnt, qi, rc))
        cand.sort(key=lambda x: -x[0])
        for _, qi, rc in cand:
            if qi in matched or rc in used_b:
                continue
            matched[qi] = rc
            tier_of[qi] = hop_idx
            used_b.add(rc)

    pairs = list(matched.items())
    matched_arch_agree = sum(1 for qi, ri in pairs if A["arch"][qi] == B["arch"][ri]) / len(pairs)
    matched_kw_dom_agree = sum(1 for qi, ri in pairs if A["kw_dom"][qi] == B["kw_dom"][ri]) / len(pairs)
    kw_cos_vals = [float(np.dot(A["kw_norm"][qi], B["kw_norm"][ri])) for qi, ri in pairs]
    res_vals = [float(np.linalg.norm(A["comp"][qi] - b_shifted[ri])) for qi, ri in pairs]
    matched_kw_cos = float(np.median(kw_cos_vals))
    post_delta_res = float(np.median(res_vals))
    gate = None
    ev = np.full(len(pairs), np.nan)
    if evidence_fdr is not None:
        ev, gate = evidence_gate(A, B, pairs, delta, ref, fdr=evidence_fdr, draws=evidence_draws)
    thr = gate.threshold if gate is not None and gate.n_sufficient else -np.inf
    match_records = (
        tuple(
            MatchedPair(
                row_a=int(A["rows"][qi]),
                row_b=int(B["rows"][ri]),
                tier=tier_of[qi],
                kw_cos=kw_c,
                delta_res=d_res,
                evidence=float(e),
                sufficient=bool(e <= thr),
            )
            for (qi, ri), kw_c, d_res, e in zip(pairs, kw_cos_vals, res_vals, ev)
        )
        if include_matches
        else None
    )

    return TwoLayerResult(
        repo_a=A["name"],
        repo_b=B["name"],
        n_a=A["n"],
        n_b=B["n"],
        fgw_cost=fgw_cost,
        t1_n=len(top_seeds),
        se_n=len(pairs),
        se_coverage=len(pairs) / max(1, min(A["n"], B["n"])),
        kw_jsd=kw_jsd,
        arch_jsd=arch_jsd,
        matched_kw_cos=matched_kw_cos,
        matched_kw_dom_agree=matched_kw_dom_agree,
        matched_arch_agree=matched_arch_agree,
        post_delta_res=post_delta_res,
        matches=match_records,
        evidence=gate,
    )
