# structural-fingerprint-scanner

**Blind, two-layer structural homology and stylometry for codebases**, built on [GitGalaxy](https://github.com/squid-protocol/gitgalaxy) scans.

`structural-fingerprint-scanner` (`sfs`) compares repositories **without reading function names, comments, or file paths**. It pairs two complementary layers:

1. **Layer 1 — Macro-Topology (FGW + Seed-and-Extend):** Fused Gromov–Wasserstein (`FGW`) optimal transport over function geometry + hop-distance call graphs, followed by 3-hop call-graph extension. Recovers vertical source ports, cross-language translations, and horizontally borrowed modules.
2. **Layer 2 — Micro-Idiomatic Texture (32-Keyword-Family Lattice + `delta_lang`):** Distributional and per-matched-pair comparison across 32 structural signal families (branching, bitwise ops, pointer/state flux, encapsulation, I/O, etc.), with additive language-translation centering (`LanguageDeltaModel`) and linear style attribution (`StyleAxis`).

It also provides **cohort fingerprinting**, **target–decoy structural twins**, **descent & profile trees**, **subsystem subway maps**, **git-history drift timelines**, and **scan-quality verification** (`ctags` / `tree-sitter` + LLM triage + PII-stripped GitGalaxy Epic issue reporting).

---

## Install

```bash
# Core (Python 3.10+, numpy, scipy — works on pre-scanned *_galaxy_master.db or .sfp.json files)
pip install .

# With GitGalaxy auto-scanning for raw source repositories
pip install '.[scan]'

# With Tree-sitter quality verification (also auto-uses Universal Ctags if on PATH)
pip install '.[verify]'

# Development / tests
pip install '.[test]'

sfs --version
```

> **Note:** `pip install '.[scan]'` requires `gitgalaxy` to be reachable (via PyPI or `pip install git+https://github.com/squid-protocol/gitgalaxy.git`).

---

## Quickstart & Uses

### 1. Two-Layer Comparison (`sfs lattice-compare`)
Compare two pre-scanned DBs or raw source repositories (with `[scan]` installed):

```bash
sfs lattice-compare repo_a/ repo_b/ --lens portable --json
sfs lattice-compare repo_a/ repo_b/ --lens portable --matches --verify --issue-out epic.md
```

### 2. Scan Quality Verification & GitGalaxy Triage (`sfs quality`)
Guard against under-tuned language regexes or slicers before trusting downstream numbers:

```bash
# Check internal yield + optional ctags / tree-sitter recall
sfs quality repo_or_scan --source ./repo --verify

# Emit a disagreement cell list + LLM validation prompt (separates real gaps from parser noise)
sfs quality repo_or_scan --source ./repo --verify --triage-out triage.md

# Emit a copy-pasteable GitGalaxy Epic issue (standard or PII-stripped paranoid mode)
sfs quality repo_or_scan --source ./repo --verify --epic-issue --privacy paranoid --issue-out epic.md
```

### 3. Cohort Fingerprints, Twins, Subway Maps & Timelines

```bash
# Fingerprint scans (writes <name>.sfp.json + <name>.sfp.npz)
sfs fingerprint scans/ -o fingerprints/

# Compare a cohort: distances, twins, descent & profile trees, report.html
sfs compare fingerprints/ -o report/ --declared-lineage lineage.csv
sfs compare fingerprints/ -o report-portable/ --lens portable --no-twins

# Function-level structural twins (CSV)
sfs twins fingerprints/A.sfp.json fingerprints/B.sfp.json -o twins/

# Subsystem subway maps & cross-check
sfs subway fingerprints/ -o subway/ --reference A --taxonomy subsystems.json [--declared-lineage lineage.csv]

# Git-history drift timeline
sfs timeline --repo A=path/to/A --repo B=path/to/B --reference fingerprints/R.sfp.json \
    --engine /path/to/gitgalaxy --python /path/to/engine-venv/bin/python --every 1y -o timeline/

# Explain what separates two repositories & inspect the 37D reference frame
sfs explain fingerprints/A.sfp.json fingerprints/B.sfp.json
sfs spec [--lens portable] [--json]
```

---

## Empirical Evidence

The two-layer pipeline has been validated across three benchmark programs (all blind to identifiers and file paths):

### 1. `doom-dna` — Lineage, Cross-Language Ports & Horizontal Borrowing
*Source study: [`squid-protocol/doom-dna`](https://github.com/squid-protocol/doom-dna)*

- **Clean port vs. scratch/unrelated separation (41 ports + controls):**
  - Unrelated C (`lua` `0.0%`, `quake`/`quake2` `0.6%`) and from-scratch reimplementations (`room4doom`, `rust-doom`, `helion` all `0.0%`) sit at the noise floor.
  - Layout-stripped ports are recovered 1-to-1: single-header `puredoom` (`89.0%`, `590` functions) and flat-directory Go port `gore` (`75.9%`, `426` functions).
- **Horizontal gene transfer:** Unsupervised Seed-and-Extend recovers Chocolate Doom's `textscreen` UI library inside both `eternity` (`76.1%` of Tier-1 hits) and `odamex` (`84.0%`), and directly between `eternity ↔ odamex` (`190` Tier-1 hits) across distinct DOOM branches.
- **Architectural clustering vs. syntax:** Three COBOL raycasters cluster at `FGW 0.670–0.706` vs. enterprise business COBOL (`carddemo-cobol-ctrl`) at `0.748–0.772`.
- **Preregistered `DOOM-E5` held-out predictions:** Confirmed `cydoomgeneric` (`FGW=0.313`, `100.0%` Tier-1 vs. `doomgeneric`), `DOOM-style-Game` (`0.0%`), and `doome` (`0.0%`); falsifications on `iron-doom` (`0.0%`) and `doomarkable` (`0.0%`) pinpointed the paradigm boundary (below).

### 2. `agent-habits` — Model Fingerprinting & Language-Centered Style
- **Same-model runs resemble source forks (56 `confirm` runs):** Independent Sonnet runs (`FGW = 0.154–0.489`, `78.7%` Tier-1 / `68.1%` SE) match or exceed true C DOOM ports even with zero shared codebase ancestry.
- **32-keyword-family lattice (`-kw_JSD`):** Achieves **`0.944` ROC AUC** for same-model vs. different-model attribution within `(task, lang)`, outperforming FGW Seed-and-Extend (`0.874` AUC) and 14-archetype JSD (`0.655` AUC).
- **Additive language-translation centering (`delta_lang`):** Pure 32D translation matches or beats 32×32 Orthogonal Procrustes (`0.698` vs. `0.645` on held-out `chess`), and a language-centered `Claude Code vs. Codex CLI` vector transfers zero-shot to 16 held-out `explore/` scans at **`0.730` ROC AUC** (`6/7` within-`(task, lang)` pairs ordered `cc > codex`).

### 3. `keyword-rosetta` — Zero-Shot Cross-Language Translation Geometry
- Using 24-language centroids from `keyword-rosetta` (`language-crucible` v4.3.1 DB), predicting a port's 32-keyword-family profile as `linuxdoom + (rosetta[lang] - rosetta["c"])` reduces JSD across all 5 tested cross-language DOOM translations:
  - `mochadoom` (`java`): `0.358 → 0.193` (`-0.165`)
  - `doomf77` (`fortran`): `0.353 → 0.253` (`-0.100`)
  - `managed-doom` (`csharp`): `0.332 → 0.273` (`-0.059`)
  - `gore` (`go`): `0.329 → 0.273` (`-0.056`)
  - `dartdoom` (`dart`): `0.350 → 0.302` (`-0.048`)
- Conversely, general-purpose Rosetta centroids do *not* improve subtle cross-language model attribution on `chess`/`sat` (`0.660 → 0.601`), showing that coarse translation shifts transfer universally while fine model-style residuals need domain-matched `delta_lang`.

---

## Limits & Measured Accuracy Floors

### Structural Limits
- **Paradigm shifts:** `C → Go` (`gore`, `75.9%`) preserves imperative call-graph DNA, whereas `C → Bevy ECS` (`iron-doom`, `0.0%`) replaces the call graph with decoupled ECS systems and drops to the from-scratch floor.
- **Externalized wrappers:** Thin platform wrappers around external crates (`doomarkable`, 41 functions) lack the vendored engine call graph (`0.0%`).
- **Uncentered cross-domain attribution:** Comparing `kw_JSD` directly across unrelated domains (e.g. `raycaster` vs. `chess`/`sat` COBOL) is confounded by task structure unless language/domain offsets are centered.

### Degradation & Resilience Asymmetry (DOOM Ablation)
Controlled ablations against `linuxdoom` (`doomgeneric`, `gore`, `rust-doom`) show that Layer 1 and Layer 2 fail in opposite ways:

| Failure Mode | Layer 1 (`FGW Seed-and-Extend`) | Layer 2 (`32-Keyword Lattice`) |
|---|---|---|
| **Missing functions** | Same-lang C holds `>80%` T1 through `50%` loss (cliff at `75%` drop); cross-lang Go drops below `50%` T1 at `50%` loss | **Resilient** (`kw_JSD` flat `0.101–0.109` up to `90%` drop) |
| **Missing call edges** | Same-lang C holds `>70%` T1 through `75%` loss; cross-lang Go drops below `50%` T1 at `50%` loss | **Resilient** |
| **Missing keyword signals** | **Resilient** (`98.7%` C / `91.5%` Go T1 even with `90%` signal columns zeroed) | **Sensitive** (`doomgeneric` `kw_JSD` jumps `0.104 → 0.392` at `25%` loss and `0.748` at `50%` loss) |

This asymmetry is why `sfs quality --verify` gates on **both** parser function recall (`ctags` / `tree-sitter`, protecting Layer 1) **and** zero-signal rate (protecting Layer 2).

---

## GitHub Action

[`action.yml`](action.yml) provides a composite GitHub Action that installs `structural-fingerprint-scanner[scan]`, runs `sfs lattice-compare` on two repositories or scan directories, and emits `fgw-cost`, `se-coverage`, `kw-jsd`, and `json` outputs plus a step summary:

```yaml
- uses: squid-protocol/structural-fingerprint-scanner@v0.1.0
  with:
    repo-a: ./baseline
    repo-b: ./candidate
    lens: portable
```

---

## Reference Frame, Docs & Tests

- [docs/METHOD.md](docs/METHOD.md) specifies formulas, calibration, and limits; [examples/doom/](examples/doom/) contains the 44-port DOOM reference dataset.
- The vendored reference model (`src/structural_fingerprint/data/`) records the GitGalaxy commit (`spec_hash`) used to normalize the 37D feature frame.

```bash
pytest
# Engine parity on real scans:
SFS_REAL_SCANS=/path/to/scans pytest
```

## License

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
