# How sfs finds a port, shown

![How sfs finds a port](seed-extend.gif)

[GIF](seed-extend.gif) · [MP4](seed-extend.mp4) · [final frame](seed-extend.png)

One film of `sfs lattice-compare` (`lattice.seed_and_extend_with_lattice`) at work on two real scans, both scanned by
GitGalaxy for the DOOM study:
- **linuxdoom**: id Software's original C, 777 functions of 3+ lines;
- **gore**: a Go port of it, 996 functions.

**Every number and every function drawn comes from the run.**
- **Data.** `tools/media/seed_extend_data.py` replays the matching step by step to keep the intermediates the library
  discards. It refuses to write anything unless its pairs and seed count equal the library's own result.
- **Verdicts.** Each pair's evidence verdict is the library's own (`MatchedPair.sufficient`).
- **Neighbourhood.** The drawing shows two real anchors, the real matches that grew from them, and some neighbours
  that never got a partner. The zoom-out at the end shows every function.

## The steps

1. **Read.** Each function becomes a shape:
   - its size, arguments and 32 keyword-family counts;
   - where it sits in its call graph.

   Names are hidden, and the port's functions start scrambled.
2. **Match.**
   - Fused Gromov–Wasserstein optimal transport pairs functions whose shape *and* call-graph position agree.
   - 439 linuxdoom functions and their best gore partner pick each other (rings).
   - Only the strongest 15% are kept as **anchors**: 66.
3. **Extend.**
   - **Offset.** The median difference between the anchor pairs is the C → Go offset. It is subtracted before
     comparing shapes. Here, C functions are longer and use more pointers.
   - **Hops.** Three times, a caller or callee of a matched function pairs with its partner's unclaimed caller or
     callee that is closest in shape.
   - **Growth.** Matched functions go 66 → 276 → 482 → 598, 77% of linuxdoom.
4. **Fingerprint check (evidence quality).** Extend always finds partners, even in unrelated code: the same steps
   pair 123 functions of rust-doom, a from-scratch Rust engine, and none of them is the same function. So every pair
   is checked:
   - **Compare.** The pair's 32-family keyword mixes are compared (shown on a log scale), and so is its shape after
     the offset.
   - **Threshold.** The pair's evidence must be stronger than a threshold set by **decoys**: the port's functions
     with each measurement shuffled among functions of the same kind (the target–decoy rule `sfs twins` uses).
   - **Verdict.** It passes with *sufficient evidence* or is held back with *insufficient evidence*. Insufficient is
     not wrong: it is a match the evidence cannot back.

   | | pairs with sufficient evidence |
   |---|---:|
   | gore (Go port) | 121 of 598 |
   | rust-doom (Rust rewrite) | 0 of 123 |
   | random one-to-one pairing (same function) | 0 of 777 |

5. **Answer key and zoom out.** Names, never used to match, grade the result. In the drawn neighbourhood:
   - all 9 pairs with sufficient evidence are the same function;
   - all 4 drawn pairs that are not the same function were held back.

   Then the camera pulls out to all 777 functions on linuxdoom's call graph:

   | final verdict | functions |
   |---|---:|
   | sufficient evidence, same function by name (teal) | 108 |
   | sufficient evidence, not confirmed by name (amber): 11 a different function, 2 with no same-named Go function | 13 |
   | matched, insufficient evidence (slate) | 477 |
   | no partner found (dark) | 179 |

## How much each step helps (same pair, same answer key)

| what is kept | pairs | same function |
|---|---:|---:|
| random one-to-one pairing | 777 | 0 |
| shape alone: reciprocal nearest neighbour, no call graph | 329 | 220 (67%) |
| shape + call-graph position (FGW), all reciprocal best | 439 | 267 (61%) |
| … only the strongest 15% (anchors) | 66 | 62 (94%) |
| anchors + 3 hops of extension | 598 | 364 (61%) |
| … only pairs with sufficient evidence | 121 | 108 (89%) |
| extension without the language offset | 604 | 365 |

The language offset barely matters for this pair (364 vs 365); the film says so.

## Caveats

- **What the false-pass rate covers.** The gate runs at a decoy-estimated false-pass rate of 2%: about 1.4 decoy
  pairs pass per draw. But decoys model "a random function of the same kind", not a near-identical sibling (the
  hold-outs include `G_WriteDemoTiccmd` matched to `g_ReadDemoTiccmd`). Measured by name, 11% of certified pairs are
  not confirmed.
- **How the default was chosen.** The 2% default was picked with this answer key in view. At `twins`' 1%, at least
  100 pairs must pass with no decoy among them, and nothing was certified here.
- **Answer key.** The key is name-based:
  - same name ignoring case and underscores;
  - also after dropping one short module prefix the port added (gore writes `IdentifyVersion` as
    `d_IdentifyVersion`).

  A pair whose linuxdoom name has no counterpart in gore is "not confirmed", not wrong.
- **One pair.** This is one pair of codebases, with one rewrite as the contrast. The film measures code shape, not
  behaviour.
- **Layout.** In the full map, linked functions are laid out by a force layout of the call graph (layout only, no
  data). Functions with no resolved call sit in a band at the bottom.
- **Licences.** linuxdoom and gore are GPL-2.0 projects. Only function names and counts appear here, no code.

## Regenerate

The data step needs only the scanner's dependencies. The film needs Pillow and ffmpeg (or the `imageio-ffmpeg`
wheel, whose bundled binary is used when `ffmpeg` is not on PATH).

```bash
python tools/media/seed_extend_data.py --a <scans>/linuxdoom --b <scans>/gore --out docs/media/seed-extend-data.json
python tools/media/seed_extend_data.py --a <scans>/linuxdoom --b <scans>/rust-doom --summary-only \
    --out docs/media/seed-extend-contrast.json
python tools/media/seed_extend_explainer.py --data docs/media/seed-extend-data.json \
    --contrast docs/media/seed-extend-contrast.json --out docs/media/seed-extend
```

`<scans>` is the DOOM study's scan directory at engine `9f213ab83`. The rendered data is committed beside the film,
so the film re-renders without the scans.
