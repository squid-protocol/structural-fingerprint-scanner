"""Scan quality assessment, disagreement cell triage, and GitGalaxy Epic issue generator.

Checks whether a GitGalaxy scan captured enough functions, 32-keyword signals,
and call edges for accurate downstream fingerprinting/alignment. When ``source_dir``
and ``verify=True`` are given, optionally cross-checks extracted function counts and
function boundaries against Universal Ctags and Tree-sitter (if installed), emits
a structured disagreement cell list + LLM triage prompt, and formats an outbound
GitGalaxy Epic issue in either ``standard`` or ``paranoid`` (PII-stripped) mode.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .scan_io import ScanData

_CTAGS_FUNC_KINDS = frozenset({
    "function",
    "method",
    "subroutine",
    "procedure",
    "paragraph",
    "section",
    "func",
})

_TS_FUNC_NODE_TYPES = frozenset({
    "function_definition",
    "function_declaration",
    "function_item",
    "method_definition",
    "method_declaration",
    "function",
    "subroutine_definition",
    "paragraph_header",
    "section_header",
})

_TS_LANG_ALIASES = {
    "c++": "cpp",
    "c#": "c_sharp",
    "csharp": "c_sharp",
    "js": "javascript",
    "ts": "typescript",
    "golang": "go",
    "py": "python",
    "rb": "ruby",
    "rs": "rust",
}


@dataclass(frozen=True)
class DisagreementCell:
    """One function- or file-level discrepancy for immediate user/LLM triage."""

    cell_id: str
    language: str
    kind: str  # missed_by_gitgalaxy | extra_in_gitgalaxy | zero_signal_function | zero_function_file
    source_tool: str  # ctags | tree-sitter | internal
    file_path: str
    func_name: str
    line_start: int | None
    line_end: int | None
    loc: int | None
    note: str


@dataclass(frozen=True)
class LanguageQuality:
    """Per-language scan yield and optional external parser cross-check."""

    language: str
    files: int
    coding_loc: int
    gg_functions: int
    zero_function_files: int
    zero_signal_rate: float
    call_edges_per_func: float
    ctags_functions: int | None
    treesitter_functions: int | None
    ctags_recall: float | None
    treesitter_recall: float | None
    status: str
    reasons: tuple[str, ...]
    sample_files: tuple[str, ...]


@dataclass(frozen=True)
class QualityReport:
    """Repository-wide scan quality report with LLM triage & Epic issue formatting."""

    repo: str
    overall_status: str
    accurate_numbers: bool
    missing_signals: tuple[str, ...]
    tools_used: tuple[str, ...]
    languages: tuple[LanguageQuality, ...]
    disagreements: tuple[DisagreementCell, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_llm_triage_prompt(self, *, max_cells: int = 30) -> str:
        """Render a cell-validation prompt + GitGalaxy slicing instructions for an LLM."""
        cells = list(self.disagreements[:max_cells])
        payload = [asdict(c) for c in cells]
        return "\n".join([
            f"# LLM Scan-Quality Cell Validation (`{self.repo}`)",
            "",
            "> **Note to reviewer:** Most raw parser-vs-slicer disagreements (e.g., C/C++ forward declarations,",
            "> `< 3` LOC helpers, lambdas/closures, or macro expansions) are **expected false positives**.",
            "> Use this prompt to separate true GitGalaxy lexical/slicer bugs from parser noise before filing an issue.",
            "",
            "## System Instructions (GitGalaxy Slicing & Signal Rules)",
            "You are validating function-extraction and keyword-signal disagreements between GitGalaxy and",
            "reference parsers (`ctags` / `tree-sitter`).",
            "- GitGalaxy intentionally skips top-level/module scopes and functions shorter than `min_loc` (typically 3 LOC).",
            "- Header prototypes / forward declarations (no body) are NOT missed functions.",
            "- Classify each cell into exactly one verdict:",
            "  1. `TRUE_GG_MISS`: A real, non-trivial function (`>= 3` LOC with a body) that GitGalaxy failed to extract.",
            "  2. `TRUE_GG_SIGNAL_GAP`: A real function (`>= 5` LOC) extracted by GitGalaxy where all 32 keyword regexes failed to fire.",
            "  3. `FALSE_PARSER_NOISE`: Prototype, forward declaration, `< 3` LOC helper, lambda/anonymous callback, or macro.",
            "",
            "## Task",
            "Inspect the referenced files/lines for each cell below and return a JSON array of:",
            '`{"cell_id": "...", "verdict": "TRUE_GG_MISS|TRUE_GG_SIGNAL_GAP|FALSE_PARSER_NOISE", "symptom": "brief lexical pattern"}`',
            "Then summarize whether downstream `sfs` numbers are trustworthy.",
            "",
            f"## Disagreement Cells ({len(cells)} of {len(self.disagreements)} shown)",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ])

    def to_epic_issue(self, *, privacy: str = "standard") -> str:
        """Render a single copy-pasteable GitHub Epic Issue for GitGalaxy.

        ``privacy='standard'`` includes repo name, file paths, and function names for exact tracing.
        ``privacy='paranoid'`` strips all repo/file/function PII, keeping only language, extension,
        aggregate metrics, and generalized LOC/symptom buckets.
        """
        if privacy not in ("standard", "paranoid"):
            raise ValueError(f"privacy must be 'standard' or 'paranoid', got {privacy!r}")
        paranoid = privacy == "paranoid"
        repo_label = "[REDACTED_REPO]" if paranoid else f"`{self.repo}`"

        flagged = [lq for lq in self.languages if lq.status in ("warn", "fail")]
        if not flagged and not self.missing_signals:
            return (
                f"# [Quality OK] {repo_label} scan passed all checks\n\n"
                f"No GitGalaxy lexical gaps detected (`tools={', '.join(self.tools_used)}`).\n"
            )

        langs_title = ", ".join(lq.language for lq in flagged[:4]) or "signal schema"
        lines = [
            f"# [Epic] GitGalaxy lexical & slicer tuning ({langs_title})"
            if paranoid
            else f"# [Epic] GitGalaxy lexical & slicer tuning for `{self.repo}` ({langs_title})",
            "",
            f"- **Repository:** {repo_label}",
            f"- **Privacy mode:** `{privacy}`",
            f"- **Overall quality:** `{self.overall_status.upper()}` (accurate downstream numbers: `{self.accurate_numbers}`)",
            f"- **Verification tools:** {', '.join(self.tools_used)}",
        ]
        if self.missing_signals:
            lines.append(f"- **Missing signal columns:** `{', '.join(self.missing_signals)}`")
        lines.extend([
            "",
            "## Language Yield Summary",
            "",
            "| Language | Files | Coding LOC | GG Funcs | 0-Func Files | 0-Signal Rate | Edges/Func | Ctags | Tree-sitter | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for lq in flagged:
            ct_str = f"{lq.ctags_functions} ({lq.ctags_recall:.0%})" if lq.ctags_functions is not None and lq.ctags_recall is not None else "-"
            ts_str = f"{lq.treesitter_functions} ({lq.treesitter_recall:.0%})" if lq.treesitter_functions is not None and lq.treesitter_recall is not None else "-"
            lines.append(
                f"| `{lq.language}` | {lq.files} | {lq.coding_loc} | {lq.gg_functions} | "
                f"{lq.zero_function_files} | {lq.zero_signal_rate:.0%} | {lq.call_edges_per_func:.2f} | "
                f"{ct_str} | {ts_str} | **{lq.status.upper()}** |"
            )

        lines.extend(["", "## Detected Lexical / Slicer Gaps", ""])
        for lq in flagged:
            lines.append(f"### `{lq.language}` ({lq.status.upper()})")
            for r in lq.reasons:
                lines.append(f"- {r}")
            lang_cells = [c for c in self.disagreements if c.language == lq.language]
            if paranoid:
                exts = sorted({Path(p).suffix or "<no_ext>" for p in lq.sample_files})
                if exts:
                    lines.append(f"- Affected file extensions: {', '.join(f'`{e}`' for e in exts)}")
                if lang_cells:
                    by_kind: dict[str, int] = defaultdict(int)
                    for c in lang_cells:
                        by_kind[c.kind] += 1
                    summary = ", ".join(f"`{k}`={v}" for k, v in sorted(by_kind.items()))
                    lines.append(f"- Disagreement breakdown (PII stripped): {summary}")
            else:
                if lq.sample_files:
                    lines.append(f"- Sample affected files: {', '.join(f'`{p}`' for p in lq.sample_files)}")
                for c in lang_cells[:5]:
                    loc_s = f" (~{c.loc} LOC)" if c.loc is not None else ""
                    fn_s = f" `{c.func_name}`" if c.func_name else ""
                    lines.append(f"  - `{c.kind}` via `{c.source_tool}` in `{c.file_path}`{fn_s}{loc_s}")
            lines.append("")

        lines.extend([
            "## Action Checklist for `gitgalaxy`",
            "",
            "- [ ] Tune function-boundary regexes / slicer rules for each `FAIL`/`WARN` language above",
            "- [ ] Verify all 32 keyword-family regexes fire on non-trivial functions (`0-signal rate < 25%`)",
            "- [ ] Re-run `sfs quality --verify` against `ctags` / `tree-sitter` until recall >= 75%",
            "",
        ])
        return "\n".join(lines)


def _extract_ctags_entries(
    source_dir: Path,
    rel_paths: list[str],
) -> list[tuple[str, str, int | None, int | None]] | None:
    ctags_bin = shutil.which("ctags")
    if not ctags_bin:
        return None
    path_map = {str((source_dir / rp).resolve()): rp for rp in rel_paths if (source_dir / rp).is_file()}
    if not path_map:
        return None
    res = subprocess.run(
        [ctags_bin, "-f", "-", "--fields=Kne", "-L", "-"],
        input="\n".join(path_map.keys()) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return None
    entries: list[tuple[str, str, int | None, int | None]] = []
    for line in res.stdout.splitlines():
        if not line or line.startswith("!"):
            continue
        parts = line.split("\t")
        if len(parts) < 4 or parts[3].strip().lower() not in _CTAGS_FUNC_KINDS:
            continue
        name = parts[0].strip()
        rp = path_map.get(str(Path(parts[1]).resolve()), parts[1])
        line_no: int | None = None
        end_no: int | None = None
        for field in parts[4:]:
            if field.startswith("line:"):
                line_no = int(field[5:])
            elif field.startswith("end:"):
                end_no = int(field[4:])
        entries.append((rp, name, line_no, end_no))
    return entries


def _extract_treesitter_entries(
    source_dir: Path,
    rel_paths: list[str],
    language: str,
) -> list[tuple[str, str, int, int]] | None:
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        return None
    ts_lang = _TS_LANG_ALIASES.get(language.lower(), language.lower())
    try:
        parser = get_parser(ts_lang)
    except Exception:
        return None
    entries: list[tuple[str, str, int, int]] = []
    found_any = False
    for rp in rel_paths:
        fp = source_dir / rp
        if not fp.is_file():
            continue
        found_any = True
        try:
            src_bytes = fp.read_bytes()
            tree = parser.parse(src_bytes)
        except Exception:
            continue
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type in _TS_FUNC_NODE_TYPES:
                name_node = node.child_by_field_name("name") or node.child_by_field_name("declarator")
                fn_name = name_node.text.decode("utf-8", errors="replace").split("(")[0].strip() if name_node else ""
                entries.append((rp, fn_name, node.start_point[0] + 1, node.end_point[0] + 1))
            stack.extend(node.children)
    return entries if found_any else None


def assess_quality(
    scan: ScanData,
    *,
    source_dir: Path | None = None,
    verify: bool = False,
) -> QualityReport:
    """Assess GitGalaxy scan quality per language and optionally cross-check with ctags/tree-sitter."""
    tools = ["internal"]
    use_ctags = bool(verify and source_dir is not None and shutil.which("ctags"))
    if use_ctags:
        tools.append("ctags")
    use_ts = False
    if verify and source_dir is not None:
        try:
            import tree_sitter_language_pack  # noqa: F401
            use_ts = True
            tools.append("tree-sitter")
        except ImportError:
            pass

    files_by_lang: dict[str, list[int]] = defaultdict(list)
    for fi, lang in enumerate(scan.file_language):
        files_by_lang[(lang or "unknown").lower()].append(fi)

    funcs_by_file: dict[int, list[int]] = defaultdict(list)
    for fn_i, fi in enumerate(scan.func_file.tolist()):
        funcs_by_file[int(fi)].append(fn_i)

    edge_deg = np.zeros(scan.n_functions, dtype=np.int64)
    if len(scan.call_edges) > 0:
        for u, v in scan.call_edges.tolist():
            edge_deg[u] += 1
            edge_deg[v] += 1

    sig_sums = scan.signals.sum(axis=1) if scan.n_functions > 0 else np.zeros(0, dtype=np.float64)

    lang_reports: list[LanguageQuality] = []
    cells: list[DisagreementCell] = []
    cell_counter = 0

    def _next_id() -> str:
        nonlocal cell_counter
        cell_counter += 1
        return f"cell_{cell_counter:04d}"

    for lang in sorted(files_by_lang):
        f_idxs = files_by_lang[lang]
        coding_loc = int(sum(int(scan.file_coding_loc[fi]) for fi in f_idxs))
        fn_idxs = [fn_i for fi in f_idxs for fn_i in funcs_by_file.get(fi, ())]
        gg_funcs = len(fn_idxs)
        gg_name_set = {(scan.file_path[int(scan.func_file[fn_i])], scan.func_name[fn_i].lower()) for fn_i in fn_idxs}

        zero_func_files = [
            scan.file_path[fi]
            for fi in f_idxs
            if int(scan.file_coding_loc[fi]) >= 20 and not funcs_by_file.get(fi)
        ]
        for zf in zero_func_files:
            cells.append(DisagreementCell(
                cell_id=_next_id(),
                language=lang,
                kind="zero_function_file",
                source_tool="internal",
                file_path=zf,
                func_name="",
                line_start=None,
                line_end=None,
                loc=None,
                note="File with >=20 coding LOC yielded 0 GitGalaxy functions",
            ))

        nontrivial_fns = [fn_i for fn_i in fn_idxs if float(scan.loc[fn_i]) >= 5.0]
        zero_sig_fns = [fn_i for fn_i in nontrivial_fns if float(sig_sums[fn_i]) <= 0.0]
        zero_sig_rate = (len(zero_sig_fns) / len(nontrivial_fns)) if nontrivial_fns else 0.0
        for fn_i in zero_sig_fns[:20]:
            cells.append(DisagreementCell(
                cell_id=_next_id(),
                language=lang,
                kind="zero_signal_function",
                source_tool="internal",
                file_path=scan.file_path[int(scan.func_file[fn_i])],
                func_name=scan.func_name[fn_i],
                line_start=None,
                line_end=None,
                loc=int(scan.loc[fn_i]),
                note="Extracted function (loc>=5) had 0 keyword-family signal hits",
            ))

        edges_per_func = (float(edge_deg[fn_idxs].sum()) / gg_funcs) if gg_funcs > 0 else 0.0

        rel_paths = [scan.file_path[fi] for fi in f_idxs]
        ctags_entries = _extract_ctags_entries(source_dir, rel_paths) if use_ctags and source_dir is not None else None
        ts_entries = _extract_treesitter_entries(source_dir, rel_paths, lang) if use_ts and source_dir is not None else None
        ctags_n = len(ctags_entries) if ctags_entries is not None else None
        ts_n = len(ts_entries) if ts_entries is not None else None
        ctags_rec = (gg_funcs / ctags_n) if ctags_n and ctags_n > 0 else None
        ts_rec = (gg_funcs / ts_n) if ts_n and ts_n > 0 else None

        if ctags_entries:
            for rp, name, l_start, l_end in ctags_entries:
                if (rp, name.lower()) not in gg_name_set:
                    loc_est = (l_end - l_start + 1) if (l_start is not None and l_end is not None) else None
                    cells.append(DisagreementCell(
                        cell_id=_next_id(),
                        language=lang,
                        kind="missed_by_gitgalaxy",
                        source_tool="ctags",
                        file_path=rp,
                        func_name=name,
                        line_start=l_start,
                        line_end=l_end,
                        loc=loc_est,
                        note="Present in ctags, absent from GitGalaxy function table",
                    ))
        if ts_entries:
            for rp, name, l_start, l_end in ts_entries:
                if name and (rp, name.lower()) not in gg_name_set:
                    cells.append(DisagreementCell(
                        cell_id=_next_id(),
                        language=lang,
                        kind="missed_by_gitgalaxy",
                        source_tool="tree-sitter",
                        file_path=rp,
                        func_name=name,
                        line_start=l_start,
                        line_end=l_end,
                        loc=l_end - l_start + 1,
                        note="Present in tree-sitter AST, absent from GitGalaxy function table",
                    ))

        reasons: list[str] = []
        status = "ok"
        if coding_loc >= 50 and gg_funcs == 0:
            status = "fail"
            reasons.append(f"0 functions extracted across {coding_loc} coding LOC")
        if ctags_rec is not None and ctags_n is not None and ctags_n >= 5:
            if ctags_rec < 0.50:
                status = "fail"
                reasons.append(f"ctags recall {ctags_rec:.0%} ({gg_funcs}/{ctags_n} functions)")
            elif ctags_rec < 0.75:
                if status != "fail":
                    status = "warn"
                reasons.append(f"low ctags recall {ctags_rec:.0%} ({gg_funcs}/{ctags_n} functions)")
        if ts_rec is not None and ts_n is not None and ts_n >= 5:
            if ts_rec < 0.50:
                status = "fail"
                reasons.append(f"tree-sitter recall {ts_rec:.0%} ({gg_funcs}/{ts_n} functions)")
            elif ts_rec < 0.75:
                if status != "fail":
                    status = "warn"
                reasons.append(f"low tree-sitter recall {ts_rec:.0%} ({gg_funcs}/{ts_n} functions)")
        if gg_funcs >= 5 and zero_sig_rate > 0.50:
            status = "fail"
            reasons.append(f"{zero_sig_rate:.0%} of loc>=5 functions have 0 keyword-signal hits")
        if coding_loc >= 50 and zero_func_files:
            if status == "ok":
                status = "warn"
            reasons.append(f"{len(zero_func_files)} file(s) with >=20 LOC yielded 0 functions")
        if gg_funcs >= 10 and edges_per_func == 0.0:
            if status == "ok":
                status = "warn"
            reasons.append(f"0 resolved call edges across {gg_funcs} functions")

        sample_files = tuple(
            dict.fromkeys(
                zero_func_files[:5]
                + [scan.file_path[int(scan.func_file[fn_i])] for fn_i in zero_sig_fns[:5]]
            )
        )[:5]

        lang_reports.append(
            LanguageQuality(
                language=lang,
                files=len(f_idxs),
                coding_loc=coding_loc,
                gg_functions=gg_funcs,
                zero_function_files=len(zero_func_files),
                zero_signal_rate=zero_sig_rate,
                call_edges_per_func=edges_per_func,
                ctags_functions=ctags_n,
                treesitter_functions=ts_n,
                ctags_recall=ctags_rec,
                treesitter_recall=ts_rec,
                status=status,
                reasons=tuple(reasons),
                sample_files=sample_files,
            )
        )

    statuses = {lq.status for lq in lang_reports}
    if "fail" in statuses:
        overall = "fail"
    elif "warn" in statuses or scan.missing_signals:
        overall = "warn"
    else:
        overall = "ok"

    return QualityReport(
        repo=scan.name,
        overall_status=overall,
        accurate_numbers=(overall != "fail"),
        missing_signals=scan.missing_signals,
        tools_used=tuple(tools),
        languages=tuple(lang_reports),
        disagreements=tuple(cells),
    )
