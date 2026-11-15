"""Command line: ``sfs`` / ``structural-fingerprint``."""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import os
import sys
from pathlib import Path

from . import __version__
from . import browser as BR
from . import crosscheck as CC
from . import lattice as LT
from . import quality as QL
from . import subsystems as SS
from . import subway as SW
from .compare import COMPONENTS, IncomparableError, compare_cohort, explain
from .features import KEYWORD_MODES
from .fingerprint import Fingerprint, build, safe_name
from .lineage import descent_matrix, evaluate_lineage, read_declared_lineage
from .reference import LENSES, load_reference
from .report import write_cohort
from .scan_io import ScanError, load_scan, resolve_or_scan, resolve_scans
from .twins import TwinParams, find_twins

log = logging.getLogger("structural_fingerprint")


def _inputs(
    paths: list[str],
    lens: str,
    min_loc: int,
    keyword_mode: str = "density",
    smoothing_k: float = 5.0,
) -> list[Fingerprint]:
    """Load fingerprints (``*.sfp.json``, or directories of them) and build any scans."""
    fps: list[Fingerprint] = []
    scans: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.name.endswith(".sfp.json"):
            fps.append(Fingerprint.load(p))
        elif p.is_dir() and sorted(p.glob("*.sfp.json")):
            fps.extend(Fingerprint.load(f) for f in sorted(p.glob("*.sfp.json")))
        else:
            scans.append(p)
    if scans:
        ref = load_reference(lens)
        for db in resolve_scans(scans):
            log.info("fingerprinting %s", db)
            fps.append(
                build(
                    load_scan(db, ref.signal_columns),
                    ref,
                    min_loc=min_loc,
                    keyword_mode=keyword_mode,
                    smoothing_k=smoothing_k,
                )
            )
    names = [fp.name for fp in fps]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise SystemExit(f"two inputs share a repository name: {sorted(dupes)}; rename one scan directory")
    if not fps:
        raise SystemExit("no inputs")
    return fps


def _twin_params(a: argparse.Namespace) -> TwinParams:
    return TwinParams(min_loc=a.twin_min_loc, ratio=a.ratio, fdr=a.fdr)


def _weights(text: str | None) -> dict[str, float]:
    if not text:
        return {}
    out = {}
    for part in text.split(","):
        k, _, v = part.partition("=")
        if k.strip() not in COMPONENTS:
            raise SystemExit(f"--weights: unknown component {k!r}; components are {', '.join(COMPONENTS)}")
        out[k.strip()] = float(v)
    return out


def cmd_fingerprint(a: argparse.Namespace) -> int:
    fps = _inputs(a.inputs, a.lens, a.min_loc, a.keyword_mode, a.smoothing_k)
    for fp in fps:
        p = fp.save(Path(a.out))
        d = fp.data
        print(f"{fp.name}: {d['size']['functions']:,} functions -> {p}")
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    fps = _inputs(a.inputs, a.lens, a.min_loc, a.keyword_mode, a.smoothing_k)
    if len(fps) < 2:
        raise SystemExit("compare needs at least two repositories")
    out = Path(a.out)
    for fp in fps:
        fp.save(out / "fingerprints")
    cohort = compare_cohort(fps, _weights(a.weights))
    descent = None
    params = _twin_params(a)
    if not a.no_twins:
        missing = [fp.name for fp in fps if fp.functions is None]
        if missing:
            raise SystemExit(f"twin search needs function tables; missing for {missing} (or pass --no-twins)")
        descent = descent_matrix([fp.functions for fp in fps], params, workers=a.workers,
                                 keep_results=not a.no_browser)
    lineage_eval = None
    if a.declared_lineage:
        declared = read_declared_lineage(Path(a.declared_lineage))
        lineage_eval = {}
        if descent is not None:
            lineage_eval["descent"] = evaluate_lineage(cohort.names, descent.distance, declared,
                                                       max_distance=1.0 - a.min_shared)
        if cohort.composite is not None:
            lineage_eval["profile"] = evaluate_lineage(cohort.names, cohort.composite, declared)
    tp = None if a.no_twins else {"min_loc": params.min_loc, "ratio": params.ratio, "fdr": params.fdr}
    extra = []
    if a.subway_reference:
        maps, tax, pos = SW.build_maps([fp.functions for fp in fps], a.subway_reference,
                                       SW.Taxonomy.load(Path(a.taxonomy)) if a.taxonomy else None, params)
        SW.write_outputs(out, maps, tax, a.subway_reference)
        prof = None
        if cohort.composite is not None:
            r = cohort.names.index(a.subway_reference)
            prof = {n: float(cohort.composite[r, i]) for i, n in enumerate(cohort.names)}
        declared = read_declared_lineage(Path(a.declared_lineage)) if a.declared_lineage else None
        extra.append(SW.render_section(maps, pos, tax, a.subway_reference, prof, declared))
        extra.append(_subsystem_section(fps, maps, out, declared))
        if a.crosscheck:
            extra.append(_crosscheck_section(fps, maps, out, a))
    browser_html = ""
    if descent is not None and not a.no_browser:
        data = BR.payload({fp.name: fp.functions for fp in fps}, descent, a.browser_min_shared)
        browser_html = BR.section(data)
    write_cohort(out, fps, cohort, descent, lineage_eval, tp, extra, browser_html)
    print(f"compared {len(fps)} repositories -> {out / 'report.html'}")
    return 0


def _subsystem_section(fps: list[Fingerprint], maps: list[SW.SubwayMap], out: Path,
                       declared: dict[str, list[str]] | None) -> str:
    """Per-subsystem fingerprints on the subway maps' assignment: subsystems.csv + a report section."""
    ref = load_reference(fps[0].data["spec"]["lens"])
    prof = SS.profile({fp.name: fp.functions for fp in fps}, maps, ref)
    SS.write_csv(out / "subsystems.csv", prof)
    order = [m.repo for m in SW.lineage_order(maps, declared)]
    return SS.render_section(prof, order)


def _crosscheck_section(fps: list[Fingerprint], maps: list[SW.SubwayMap], out: Path, a: argparse.Namespace) -> str:
    """Score the twins against an independent source-text verdict: crosscheck.csv/json + a report section."""
    ref = next(fp.functions for fp in fps if fp.name == maps[0].repo)
    positive = tuple(x.strip() for x in a.crosscheck_positive.split(",") if x.strip())
    cc = CC.crosscheck(ref, maps, CC.read_verdicts(Path(a.crosscheck)), positive, a.twin_min_loc)
    CC.write_csv(out / "crosscheck.csv", cc)
    (out / "crosscheck.json").write_text(json.dumps(cc.to_dict(), indent=1) + "\n")
    return CC.render_section(cc, Path(a.crosscheck).name)


def cmd_subway(a: argparse.Namespace) -> int:
    from .html import page

    fps = _inputs(a.inputs, a.lens, a.min_loc, a.keyword_mode, a.smoothing_k)
    missing = [fp.name for fp in fps if fp.functions is None]
    if missing:
        raise SystemExit(f"subway maps need function tables; missing for {missing}")
    tax = SW.Taxonomy.load(Path(a.taxonomy)) if a.taxonomy else None
    try:
        maps, tax, pos = SW.build_maps([fp.functions for fp in fps], a.reference, tax, _twin_params(a))
    except ValueError as e:
        raise SystemExit(str(e)) from e
    out = Path(a.out)
    SW.write_outputs(out, maps, tax, a.reference)
    prof = None
    if len(fps) >= 3:
        cohort = compare_cohort(fps)
        r = cohort.names.index(a.reference)
        prof = {n: float(cohort.composite[r, i]) for i, n in enumerate(cohort.names)}
    meta = f"{len(maps)} repositories on {html.escape(a.reference)}'s layout · {html.escape(tax.name)}"
    declared = read_declared_lineage(Path(a.declared_lineage)) if a.declared_lineage else None
    section = SW.render_section(maps, pos, tax, a.reference, prof, declared)
    sections = [section, _subsystem_section(fps, maps, out, declared)]
    if a.crosscheck:
        sections.append(_crosscheck_section(fps, maps, out, a))
    (out / "subway.html").write_text(page("Subway maps", meta, sections))
    print(f"wrote {out / 'subway.html'}")
    return 0


def cmd_timeline(a: argparse.Namespace) -> int:
    from . import timeline as TL
    from .html import page

    repos = []
    for spec in a.repo:
        name, _, path = spec.partition("=") if "=" in spec else (Path(spec).name, "", spec)
        repos.append((name, Path(path).expanduser()))
    ref = Fingerprint.load(Path(a.reference))
    if ref.functions is None:
        raise SystemExit("the reference fingerprint needs its .sfp.npz function table")
    tax = SW.Taxonomy.load(Path(a.taxonomy)) if a.taxonomy else None
    scanner = TL.Scanner(engine=Path(a.engine).expanduser(), python=a.python)
    out = Path(a.out)
    try:
        thresholds: dict = {}
        rows = TL.run(repos, ref, out, scanner, a.every, a.since, a.until, _twin_params(a), tax, a.keep_scans,
                      thresholds)
    except (RuntimeError, ValueError) as e:
        raise SystemExit(str(e)) from e
    if not rows:
        raise SystemExit("no snapshot could be scanned; see the scan logs (use --keep-scans)")
    meta = {"every": a.every, "since": a.since, "until": a.until, "lens": ref.data["spec"]["lens"],
            "engine": str(a.engine), "pooled_thresholds": thresholds}
    TL.write_outputs(out, rows, ref.name, meta)
    events = TL.read_events(Path(a.events)) if a.events else None
    head = f"{len(repos)} repositories · {len(rows)} snapshots · every {html.escape(a.every)} · reference {html.escape(ref.name)}"
    (out / "timeline.html").write_text(page("Drift over time", head, [TL.render_section(rows, ref.name, events)]))
    print(f"wrote {out / 'timeline.html'} ({len(rows)} snapshots)")
    return 0


def cmd_twins(a: argparse.Namespace) -> int:
    fps = _inputs([a.query, a.target], a.lens, a.min_loc, a.keyword_mode, a.smoothing_k)
    q, t = fps
    if q.functions is None or t.functions is None:
        raise SystemExit("twin search needs the .sfp.npz function table beside each fingerprint")
    r = find_twins(q.functions, t.functions, _twin_params(a))
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"twins_{safe_name(q.name)}__{safe_name(t.name)}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query_function", "query_path", "query_loc", "target_function", "target_path", "target_loc",
                        "distance", "exact_clone", "same_name"])
            for qi, ti, d, ex in sorted(zip(r.q_index, r.t_index, r.distance, r.exact, strict=True), key=lambda x: x[2]):
                qn, tn = str(q.functions.name[qi]), str(t.functions.name[ti])
                w.writerow([qn, q.functions.path[qi], int(q.functions.loc[qi]), tn, t.functions.path[ti],
                            int(t.functions.loc[ti]), f"{d:.6f}", int(ex), int(qn.strip().lower() == tn.strip().lower())])
        print(f"wrote {path}")
    summary = {"query": q.name, "target": t.name, "threshold": r.threshold, "estimated_fdr": r.est_fdr, **r.stats}
    print(json.dumps(summary, indent=1))
    return 0


def cmd_explain(a: argparse.Namespace) -> int:
    fps = _inputs([a.a, a.b], a.lens, a.min_loc, a.keyword_mode, a.smoothing_k)
    try:
        e = explain(fps[0], fps[1], top=a.top)
    except IncomparableError as err:
        raise SystemExit(str(err)) from err
    if a.json:
        print(json.dumps(e, indent=1))
        return 0
    A, B = e["pair"]
    print(f"{A}  vs  {B}\n")
    print("Component distances")
    for k, v in e["distances"].items():
        print(f"  {k:<11} {v:8.4f}")
    print("\nSignal densities (per 100 function LOC), largest shifts")
    for s in e["signals"]:
        print(f"  {s['signal']:<24} {s['per_100_loc'][0]:9.2f} -> {s['per_100_loc'][1]:9.2f}")
    print("\nArchetype shares, largest shifts")
    for s in e["archetypes"]:
        print(f"  {s['archetype']:<32} {100 * s[A]:6.1f}% -> {100 * s[B]:6.1f}%")
    print("\nFunction geometry (median, population-IQR units; fan-in/out log1p)")
    for g in e["geometry"]:
        print(f"  {g['feature']:<24} {g['median'][0]:7.3f} -> {g['median'][1]:7.3f}")
    print("\nCall graph")
    for c in e["callgraph"]:
        print(f"  {c['field']:<20} {c[A]:7.3f} -> {c[B]:7.3f}")
    return 0


def cmd_lattice_compare(a: argparse.Namespace) -> int:
    ref = load_reference(a.lens)
    cache = Path(a.scan_cache) if a.scan_cache else None
    scans = resolve_or_scan([Path(a.a), Path(a.b)], cache_dir=cache)
    sc_a = load_scan(scans[0], ref.signal_columns)
    sc_b = load_scan(scans[1], ref.signal_columns)
    if a.verify:
        src_a = Path(a.source_a) if a.source_a else (Path(a.a) if Path(a.a).is_dir() else None)
        src_b = Path(a.source_b) if a.source_b else (Path(a.b) if Path(a.b).is_dir() else None)
        q_reports = [
            QL.assess_quality(sc_a, source_dir=src_a, verify=True),
            QL.assess_quality(sc_b, source_dir=src_b, verify=True),
        ]
        for qr in q_reports:
            if qr.overall_status != "ok":
                flagged = ", ".join(f"{l.language}:{l.status}" for l in qr.languages if l.status != "ok")
                print(
                    f"warning: {qr.repo} scan quality={qr.overall_status} ({flagged or 'missing signals'}); "
                    "GitGalaxy regexes may be under-tuned for this language",
                    file=sys.stderr,
                )
        if a.issue_out:
            issues = [qr.to_epic_issue(privacy=a.privacy) for qr in q_reports if qr.overall_status != "ok"]
            if issues:
                Path(a.issue_out).write_text("\n---\n\n".join(issues))
        if a.triage_out:
            triages = [qr.to_llm_triage_prompt() for qr in q_reports if qr.disagreements]
            if triages:
                Path(a.triage_out).write_text("\n---\n\n".join(triages))
    res = LT.seed_and_extend_with_lattice(
        sc_a,
        sc_b,
        ref,
        min_loc=a.min_loc,
        hops=a.hops,
        seed_frac=a.seed_frac,
        include_matches=a.matches,
    )
    if a.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(
            f"{res.repo_a} vs {res.repo_b}: "
            f"FGW={res.fgw_cost:.3f}, SE={res.se_n} ({res.se_coverage:.1%}), "
            f"kw_JSD={res.kw_jsd:.3f}, kw_cos={res.matched_kw_cos:.3f}, "
            f"kw_dom={res.matched_kw_dom_agree:.1%}, arch_agree={res.matched_arch_agree:.1%}"
        )
    return 0


def cmd_quality(a: argparse.Namespace) -> int:
    ref = load_reference(a.lens)
    cache = Path(a.scan_cache) if a.scan_cache else None
    scans = resolve_or_scan([Path(a.target)], cache_dir=cache)
    sc = load_scan(scans[0], ref.signal_columns)
    src = Path(a.source) if a.source else (Path(a.target) if Path(a.target).is_dir() else None)
    rep = QL.assess_quality(sc, source_dir=src, verify=a.verify)
    if a.issue_out:
        Path(a.issue_out).write_text(rep.to_epic_issue(privacy=a.privacy))
    if a.triage_out:
        Path(a.triage_out).write_text(rep.to_llm_triage_prompt())
    if a.json:
        print(json.dumps(rep.to_dict(), indent=2))
    elif a.epic_issue:
        print(rep.to_epic_issue(privacy=a.privacy))
    elif a.triage:
        print(rep.to_llm_triage_prompt())
    else:
        print(
            f"{rep.repo}: status={rep.overall_status.upper()} "
            f"accurate_numbers={rep.accurate_numbers} tools={','.join(rep.tools_used)}"
        )
        for lq in rep.languages:
            extra = []
            if lq.ctags_recall is not None:
                extra.append(f"ctags={lq.gg_functions}/{lq.ctags_functions} ({lq.ctags_recall:.0%})")
            if lq.treesitter_recall is not None:
                extra.append(f"ts={lq.gg_functions}/{lq.treesitter_functions} ({lq.treesitter_recall:.0%})")
            extra_s = (" " + " ".join(extra)) if extra else ""
            print(
                f"  {lq.language:<12} {lq.status.upper():<4} "
                f"files={lq.files} loc={lq.coding_loc} funcs={lq.gg_functions} "
                f"0sig={lq.zero_signal_rate:.0%} edges/fn={lq.call_edges_per_func:.2f}{extra_s}"
            )
            for r in lq.reasons:
                print(f"    - {r}")
    return 1 if (a.strict and not rep.accurate_numbers) else 0


def cmd_spec(a: argparse.Namespace) -> int:
    ref = load_reference(a.lens)
    d = ref.describe()
    if a.json:
        print(json.dumps(d, indent=1))
        return 0
    prov = d["reference"]
    print(f"lens {d['lens']} · spec {d['spec_hash']} · reference GitGalaxy {prov.get('engine_commit', '')[:9]} "
          f"({prov.get('engine_commit_date', '')})")
    if d["lens_source"]:
        print(f"lens weights from {d['lens_source'].get('source')} @ {d['lens_source'].get('rosetta_commit', '')[:9]}")
    print(f"\n{'feature':<36} {'median':>8} {'iqr':>8} {'weight':>7}")
    for f in d["features"]:
        print(f"{f['name']:<36} {f['median']:8.3f} {f['iqr']:8.3f} {f['weight']:7.3f}")
    print(f"\n{len(d['archetypes'])} archetypes: {', '.join(d['archetypes'])}")
    return 0


def _parse_snapshot_arg(spec: str):
    from .ledger import Snapshot
    parts = spec.split("@", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ScanError(f"invalid --snapshot '{spec}'; expected SCAN_PATH@COMMIT[@LABEL]")
    scan_p = resolve_scans([parts[0]])[0]
    label = parts[2] if len(parts) == 3 else parts[1][:7]
    return Snapshot(scan=scan_p, commit=parts[1], label=label)


def cmd_port_evidence(a: argparse.Namespace) -> int:
    from .port_evidence import build_port_evidence, card_to_dict, format_evidence_card
    origin_db = resolve_scans([a.origin_scan])[0]
    snaps = [_parse_snapshot_arg(s) for s in a.snapshot]
    card = build_port_evidence(
        origin_db,
        Path(a.port_repo),
        snaps,
        origin_repo=Path(a.origin_repo) if a.origin_repo else None,
        origin_label=a.origin_label,
        port_label=a.port_label,
        min_loc=a.min_loc,
        max_fgw_nodes=a.max_fgw_nodes,
        score_names=a.score_names,
    )
    if a.json:
        print(json.dumps(card_to_dict(card), indent=2))
    else:
        print(format_evidence_card(card))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sfs", description="Structural fingerprints of codebases from GitGalaxy scans.")
    ap.add_argument("--version", action="version", version=f"structural-fingerprint-scanner {__version__}")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--lens", choices=LENSES, default="structure",
                       help="structure (engine space, default) or portable (cross-language weights)")
        p.add_argument("--min-loc", type=int, default=3,
                       help="ignore functions shorter than this in fingerprints (default 3, as the engine trainer)")
        p.add_argument("--keyword-mode", choices=KEYWORD_MODES, default="density",
                       help="how signal counts are normalized: density (default), smoothed_density, presence, counts, composition")
        p.add_argument("--smoothing-k", type=float, default=5.0,
                       help="LOC smoothing constant k for smoothed_density (default 5.0)")

    def crosscheck_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--crosscheck", help="CSV of source-text verdicts per reference function and port "
                       "(variant,file,func,cls; e.g. lineage.py's survival_functions.csv) to score the twins against")
        p.add_argument("--crosscheck-positive", default="exact,near",
                       help="verdict classes that count as surviving (default exact,near)")

    def twin_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--twin-min-loc", type=int, default=6, help="smallest function considered for twins (default 6)")
        p.add_argument("--ratio", type=float, default=0.8, help="distinctiveness ratio test (default 0.8)")
        p.add_argument("--fdr", type=float, default=0.01, help="estimated false-twin rate per pair (default 0.01)")

    p = sub.add_parser("fingerprint", help="fingerprint scans and save them")
    p.add_argument("inputs", nargs="+", help="scan databases or directories")
    p.add_argument("-o", "--out", required=True)
    common(p)
    p.set_defaults(func=cmd_fingerprint)

    p = sub.add_parser("compare", help="compare a cohort: distances, twins, trees, report")
    p.add_argument("inputs", nargs="+", help="scans, scan directories, fingerprints or fingerprint directories")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--no-twins", action="store_true", help="skip function-level twin search")
    p.add_argument("--declared-lineage", help="CSV name,parent to check the trees against")
    p.add_argument("--weights", help="component weights, e.g. archetypes=1,signals=1,geometry=1,centroid=1,callgraph=1")
    p.add_argument("--min-shared", type=float, default=0.02,
                   help="shared structure below this is no evidence of descent (default 0.02)")
    p.add_argument("--no-browser", action="store_true", help="leave the twin browser out of report.html")
    p.add_argument("--browser-min-shared", type=float, default=0.02,
                   help="twin browser: include pairs sharing at least this much structure (default 0.02)")
    p.add_argument("--subway-reference", help="also draw subway maps on this repository's layout")
    p.add_argument("--taxonomy", help="subsystem taxonomy JSON for subway maps (default: top-level directories)")
    crosscheck_opts(p)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    common(p)
    twin_opts(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("subway", help="subway maps: call graphs by subsystem on a reference's layout")
    p.add_argument("inputs", nargs="+", help="scans, scan directories, fingerprints or fingerprint directories")
    p.add_argument("--reference", required=True, help="repository whose subsystems and layout every map uses")
    p.add_argument("--taxonomy", help="subsystem taxonomy JSON (default: the reference's top-level directories)")
    p.add_argument("--declared-lineage", help="CSV name,parent: order the maps along the declared lineage")
    crosscheck_opts(p)
    p.add_argument("-o", "--out", required=True)
    common(p)
    twin_opts(p)
    p.set_defaults(func=cmd_subway)

    p = sub.add_parser("timeline", help="fingerprint repositories through their git history against a reference")
    p.add_argument("--repo", action="append", required=True,
                   help="NAME=PATH of a git repository with its history (repeatable)")
    p.add_argument("--reference", required=True, help="reference fingerprint (.sfp.json with its .sfp.npz)")
    p.add_argument("--engine", required=True, help="path to a gitgalaxy checkout to scan with")
    p.add_argument("--python", default=sys.executable, help="Python that can import the engine (default: this one)")
    p.add_argument("--every", default="1y", help="snapshot step: Ny or Nm (default 1y)")
    p.add_argument("--since", help="first snapshot date (YYYY-MM-DD; default: first commit)")
    p.add_argument("--until", help="last snapshot date (default: the head)")
    p.add_argument("--taxonomy", help="subsystem taxonomy JSON: adds calls kept / wiring per snapshot")
    p.add_argument("--events", help="CSV date,label of events to mark on the charts")
    p.add_argument("--keep-scans", action="store_true", help="keep each snapshot's scan output")
    p.add_argument("-o", "--out", required=True, help="output directory (snapshot fingerprints are cached here)")
    twin_opts(p)
    p.set_defaults(func=cmd_timeline)

    p = sub.add_parser("twins", help="function-level structural twins of QUERY in TARGET")
    p.add_argument("query")
    p.add_argument("target")
    p.add_argument("-o", "--out", help="write the twin list as CSV here")
    common(p)
    twin_opts(p)
    p.set_defaults(func=cmd_twins)

    p = sub.add_parser("explain", help="what separates two repositories")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--json", action="store_true")
    common(p)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("lattice-compare", help="two-layer FGW Seed-and-Extend + 32-keyword-family lattice comparison")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--hops", type=int, default=3, help="1-hop call-graph extension rounds (default 3)")
    p.add_argument("--seed-frac", type=float, default=0.15, help="top reciprocal FGW fraction for Tier-1 seeds (default 0.15)")
    p.add_argument("--scan-cache", help="directory to store auto-generated GitGalaxy scans when given raw source repos")
    p.add_argument("--matches", action="store_true", help="include matched function pairs in JSON output")
    p.add_argument("--verify", action="store_true", help="run scan quality verification (internal + optional ctags/tree-sitter)")
    p.add_argument("--source-a", help="optional source directory for repo A (for ctags/tree-sitter verification)")
    p.add_argument("--source-b", help="optional source directory for repo B (for ctags/tree-sitter verification)")
    p.add_argument("--issue-out", help="write a GitGalaxy Epic issue Markdown report if quality checks warn/fail")
    p.add_argument("--triage-out", help="write an LLM disagreement-cell triage prompt if discrepancies are found")
    p.add_argument("--privacy", choices=["standard", "paranoid"], default="standard",
                   help="Epic issue privacy mode: standard (paths/names) or paranoid (PII-stripped)")
    p.add_argument("--json", action="store_true")
    common(p)
    p.set_defaults(func=cmd_lattice_compare)

    p = sub.add_parser("quality", help="assess GitGalaxy scan quality, triage disagreements, and emit an Epic issue")
    p.add_argument("target", help="scan DB, scan directory, or raw source repository")
    p.add_argument("--source", help="raw source directory for optional ctags/tree-sitter function recall checks")
    p.add_argument("--verify", action="store_true", help="cross-check against ctags and tree-sitter if available")
    p.add_argument("--scan-cache", help="directory to store auto-generated GitGalaxy scan when given a raw source repo")
    p.add_argument("--epic-issue", action="store_true", help="print a copy-pasteable GitGalaxy Epic issue Markdown report")
    p.add_argument("--issue-out", help="write the Epic issue Markdown report to this path")
    p.add_argument("--triage", action="store_true", help="print the LLM cell-validation prompt + disagreement list")
    p.add_argument("--triage-out", help="write the LLM cell-validation prompt to this path")
    p.add_argument("--privacy", choices=["standard", "paranoid"], default="standard",
                   help="Epic issue privacy mode: standard (paths/names) or paranoid (PII-stripped)")
    p.add_argument("--strict", action="store_true", help="exit with code 1 if overall quality fails")
    p.add_argument("--json", action="store_true")
    common(p)
    p.set_defaults(func=cmd_quality)

    p = sub.add_parser("spec", help="print the reference frame and lens")
    p.add_argument("--json", action="store_true")
    p.add_argument("--lens", choices=LENSES, default="structure")
    p.set_defaults(func=cmd_spec)

    p = sub.add_parser("port-evidence", help="multi-snapshot Port Evidence Card (structure + comments + ledger + isotonic decay)")
    p.add_argument("--origin-scan", required=True, help="origin scan DB or directory")
    p.add_argument("--port-repo", required=True, help="target git repository path")
    p.add_argument("--snapshot", action="append", required=True, help="target snapshot as SCAN_PATH@COMMIT[@LABEL] (oldest first)")
    p.add_argument("--origin-repo", help="optional origin repository path for comment extraction")
    p.add_argument("--origin-label", default="origin")
    p.add_argument("--port-label")
    p.add_argument("--max-fgw-nodes", type=int, default=2500, help="landmark cap for FGW on large repos")
    p.add_argument("--score-names", action="store_true", help="enable post-hoc name-agreement scoring")
    p.add_argument("--json", action="store_true")
    common(p)
    p.set_defaults(func=cmd_port_evidence)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * min(a.verbose, 2), format="%(levelname)s %(message)s")
    try:
        return a.func(a)
    except (ScanError, IncomparableError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
