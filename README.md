# structural-fingerprint-scanner

Structural fingerprints, lineage and function-level twins for codebases, computed from
[GitGalaxy](https://github.com/squid-protocol/gitgalaxy) scans.

GitGalaxy measures every function it parses: size, decisions, arity, call fan-in and
fan-out, and 32 structural signals (guards, state mutation, I/O, encapsulation,
pointers, macros, and more). This tool places each function in the engine's own
37-dimensional reference frame, fitted on a population of 8.2 million functions, and
builds on it:

- a **fingerprint** of each repository: function-archetype mix, signal densities,
  function-geometry distributions, centre of mass and call-graph shape;
- a **profile distance** between repositories (how alike their designs are), which
  works across languages under the `portable` lens;
- **structural twins**: function pairs measured almost identically, found without
  names by reciprocal best hits and accepted at a calibrated false-twin rate
  (target–decoy, 1% by default). Twins survive renaming, file moves and reformatting;
- **descent**: how much of one codebase's structure survives in another, and a
  descent tree built from it;
- **subway maps**: each repository's call graph collapsed to its subsystems and drawn
  on a reference's fixed layout, with stations coloured by how much of the reference's
  subsystem survives. They follow wiring, which survives translation into another
  language even when every measurement changes. A mapping-free graph-shape check
  confirms the maps aren't an artefact of the subsystem assignment;
- **per-subsystem fingerprints**: each subsystem compared with the reference's same
  subsystem ("this renderer is designed like the original; this sound code isn't"),
  with the features that drifted, a by-language table and engine-calibration
  candidates;
- a **cross-check** that scores the twins function by function against an independent
  source-text verdict (such as the DOOM study's `lineage.py`): precision, recall, and
  what the twins really are;
- **drift timelines**: fingerprint repositories through their git history (yearly or
  any step) and chart how much of a reference survives, how fast code turns over, and
  how far the design drifts, with events marked;
- a **twin browser** in the report: pick a pair or click a descent cell and see
  every twin behind the number, by folder, with search and filters;
- reports in Markdown, HTML (heatmaps with trees), CSV, JSON and Newick.

It reads scan output only. It never sees source code and needs nothing beyond NumPy
and SciPy.

## Install

```bash
pip install .              # Python 3.10+, numpy, scipy
pip install '.[scan]'      # + gitgalaxy auto-scan for raw source repos
pip install '.[verify]'    # + tree-sitter quality verification (uses ctags if on PATH)
sfs --version
```

## Use

Scan each repository with GitGalaxy first. Each scan directory holds a
`<name>_galaxy_master.db`.

```bash
# Fingerprint scans (writes <name>.sfp.json + <name>.sfp.npz)
sfs fingerprint scans/ -o fingerprints/

# Compare a cohort: distances, twins, both trees, report.html
sfs compare fingerprints/ -o report/ --declared-lineage lineage.csv

# Across languages: down-weight what tracks the language, not the design
sfs compare fingerprints/ -o report-portable/ --lens portable --no-twins

# Which functions of A survive in B, as CSV
sfs twins fingerprints/A.sfp.json fingerprints/B.sfp.json -o twins/

# Subway maps on a reference's layout (subsystems from a taxonomy, or top-level directories)
sfs subway fingerprints/ -o subway/ --reference A --taxonomy subsystems.json [--declared-lineage lineage.csv]
# Score the twins against a source-text verdict (variant,file,func,cls CSV)
sfs subway fingerprints/ -o subway/ --reference A --taxonomy subsystems.json --crosscheck survival_functions.csv
# ...or as a section of the compare report
sfs compare fingerprints/ -o report/ --subway-reference A --taxonomy subsystems.json

# Drift through history: snapshots are scanned with the engine, fingerprinted and cached by commit
sfs timeline --repo A=path/to/A --repo B=path/to/B --reference fingerprints/R.sfp.json \
    --engine /path/to/gitgalaxy --python /path/to/engine-venv/bin/python --every 1y \
    [--taxonomy subsystems.json] [--events events.csv] -o timeline/

# What separates two repositories
sfs explain fingerprints/A.sfp.json fingerprints/B.sfp.json

# The reference frame and lens in use
sfs spec [--lens portable] [--json]

# Two-layer comparison: FGW Seed-and-Extend + 32-keyword-family lattice
# Accepts pre-scanned DBs or raw source directories (with [scan] installed)
sfs lattice-compare repo_a/ repo_b/ --lens portable [--matches] [--verify] [--json]

# Assess GitGalaxy scan quality, generate LLM disagreement triage, and emit an Epic issue
sfs quality repo_or_scan --verify --triage-out triage.md --epic-issue --privacy paranoid --issue-out epic.md
```

`compare` accepts scans, scan directories, fingerprints or fingerprint directories in
any mix. The optional `--declared-lineage` CSV (`name,parent`, one row per parent)
checks both trees against what the projects say about their own ancestry.

### Output of `compare`

| path | content |
|---|---|
| `report.html` | descent and profile heatmaps with trees, the twin browser, archetype mix, lineage check, repository table (and subway maps with `--subway-reference`) |
| `report.md` | the same in Markdown |
| `summary.json` | everything, machine-readable, with the reference and lens provenance |
| `pairs.csv` | one row per pair: every component distance, profile distance, survival both ways, shared structure, twin and clone counts, name agreement |
| `matrices/*.csv` | square matrices behind each figure |
| `trees/profile.nwk`, `trees/descent.nwk` | Newick trees |
| `subway.json`, `subway.csv` | with `--subway-reference`: stations, lines, survival per subsystem, wiring scores |
| `subsystems.csv` | with `--subway-reference`: per repository and subsystem, distances to the reference's subsystem and the drift |
| `crosscheck.csv`, `crosscheck.json` | with `--crosscheck`: precision and recall of the twins against the source-text verdict, per port and subsystem |
| `fingerprints/` | the fingerprints used |

## How to read the numbers

- **Profile distance**: 1.0 is a typical pair *in that cohort*, 0.5 twice as close.
  The component distances in `pairs.csv` are absolute.
- **Survival A→B**: share of A's functions (≥6 lines) with a twin in B.
- **Shared structure**: the larger direction's Wilson 95% lower bound on survival.
  Below 2% there is no detectable descent.
- **`--fdr`**: the target–decoy estimate understates the realized false-twin rate.
  Against source text on DOOM, nominal 1% (the default) gave 3.4% twins of unrelated
  functions, 92% exact-or-near precision and 61% recall; nominal 5% gave 12.7%, 76% and
  71%. Keep 1% when each twin must be right; use 5% for port-level survival rankings.
- **Name agreement**: names are not used to match, so when two codebases name their
  functions alike, the share of twins whose names agree is an independent estimate of
  the twin search's precision.

[docs/METHOD.md](docs/METHOD.md) specifies every formula, the calibration and the
limits. [examples/doom/](examples/doom/) validates the tool on 44 DOOM ports with
known ancestry.

## Two-layer Seed-and-Extend + Keyword Lattice & Accuracy Floors

`sfs lattice-compare` (and `structural_fingerprint.lattice`) pairs two complementary layers:

1. **Layer 1 — Macro-Topology (FGW + 3-hop Seed-and-Extend):** Aligns imperative call-graph structure and function geometry without names or paths, recovering vertical ports, cross-language translations (`C → Go`), and horizontally borrowed modules (`textscreen`).
2. **Layer 2 — Micro-Idiomatic Texture (32-Keyword-Family Lattice + `delta_lang`):** Measures the 32-keyword-family distribution (`kw_jsd`, `matched_kw_cos`) and supports additive language-translation centering (`LanguageDeltaModel`) and linear style attribution (`StyleAxis`).

### Measured Degradation Floors (DOOM Ablation)

Because Layer 1 and Layer 2 rely on different parser outputs, they fail asymmetrically:

| Failure Mode | Layer 1 (`FGW Seed-and-Extend`) | Layer 2 (`32-Keyword Lattice`) |
|---|---|---|
| **Missing functions** | Same-lang C holds `>80%` T1 through `50%` loss (cliff at `75%` drop); cross-lang Go drops below `50%` T1 at `50%` loss | **Resilient** (`kw_JSD` flat `0.101–0.109` up to `90%` drop) |
| **Missing call edges** | Same-lang C holds `>70%` T1 through `75%` loss; cross-lang Go drops below `50%` T1 at `50%` loss | **Resilient** |
| **Missing keyword signals** | **Resilient** (`98.7%` C / `91.5%` Go T1 even with `90%` signal columns zeroed) | **Sensitive** (`doomgeneric` `kw_JSD` jumps `0.104 → 0.392` at `25%` loss and `0.748` at `50%` loss) |

### Scan Quality Verification & GitGalaxy Epic Reporting

Run `sfs quality --verify` (or pass `--verify` to `sfs lattice-compare`) to guard against under-tuned language regexes/slicers:

1. **Disagreement Cell List & LLM Triage (`--triage-out`):** Cross-checks GitGalaxy against `ctags` and `tree-sitter` (when available) and emits a cell-validation prompt + slicing instructions so an LLM can separate true GitGalaxy gaps (`TRUE_GG_MISS`, `TRUE_GG_SIGNAL_GAP`) from expected parser noise (header prototypes, `<3` LOC helpers, lambdas).
2. **Outbound `gitgalaxy` Epic Issue (`--epic-issue --privacy standard|paranoid`):**
   - `standard`: keeps repo name, file paths, and function names for exact tracing.
   - `paranoid`: strips all repo, file-path, and function-name PII, leaving only language, extensions, aggregate recall/0-signal rates, and disagreement counts.

## GitHub Action

[`action.yml`](action.yml) provides a composite GitHub Action that installs `structural-fingerprint-scanner[scan]`, runs `sfs lattice-compare` on two repositories or scan directories, and emits `fgw-cost`, `se-coverage`, `kw-jsd`, and `json` outputs plus a step summary.

## Reference frame and reproducibility

The vendored reference model (`src/structural_fingerprint/data/`) records the
GitGalaxy commit it came from. Each fingerprint carries a `spec_hash` over the scaler,
weights, caps, features and lens, and fingerprints with different hashes refuse to
compare. Twin search is deterministic: decoys are seeded from the repository names.

```bash
python tools/refresh_reference.py --engine /path/to/gitgalaxy        # after the engine retrains
python tools/derive_portable_weights.py --rosetta /path/to/keyword-rosetta
```

## Tests

```bash
pip install '.[test]'
pytest
# engine parity on real scans (made by the reference's engine commit):
SFS_REAL_SCANS=/path/to/scans pytest
```
