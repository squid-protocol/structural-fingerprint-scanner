#!/usr/bin/env python3
"""Render "How sfs finds a port": Seed-and-Extend + the keyword lattice, step by step, on real scans.

Every number and every function drawn comes from ``seed_extend_data.py`` output (which refuses to write unless its
step-by-step replay equals ``seed_and_extend_with_lattice``). The film:

1. READ: every function becomes a shape (size, arguments, 32 keyword families) at a place in its call graph; names
   are hidden;
2. MATCH: optimal transport (FGW) pairs functions whose shape and call-graph position agree; pairs that pick each
   other are reciprocal best; the strongest 15% become anchors;
3. EXTEND: the C-to-Go offset is measured on the anchors; then, three times, a neighbour of a matched function is
   paired with the closest unclaimed neighbour of its partner;
4. FINGERPRINT CHECK: each matched pair's keyword fingerprint and shape must beat decoys (the library's evidence
   gate, ``MatchedPair.sufficient``); two pairs in full on a log scale, then a sweep over every drawn pair, then the
   totals next to a rewrite that shares no code and a random pairing;
5. ANSWER KEY: names, never used to match, grade every pair; the camera zooms out to every original function,
   coloured by its final verdict.

    python tools/media/seed_extend_data.py --a <linuxdoom scan> --b <gore scan> --out <work>/se-gore.json
    python tools/media/seed_extend_data.py --a <linuxdoom scan> --b <rust-doom scan> --summary-only --out <work>/se-rust.json
    python tools/media/seed_extend_explainer.py --data <work>/se-gore.json --contrast <work>/se-rust.json \\
        --out docs/media/seed-extend

Needs Pillow, numpy and ffmpeg (or the imageio-ffmpeg wheel); docs tooling only.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 12
FD = pathlib.Path("/usr/share/fonts/truetype/dejavu")


def font(name, size):
    return ImageFont.truetype(str(FD / name), size)


f_title = font("DejaVuSans-Bold.ttf", 24)
f_chip = font("DejaVuSans-Bold.ttf", 14)
f_head = font("DejaVuSans-Bold.ttf", 17)
f_cap = font("DejaVuSans.ttf", 17)
f_capb = font("DejaVuSans-Bold.ttf", 17)
f_small = font("DejaVuSans.ttf", 14)
f_smallb = font("DejaVuSans-Bold.ttf", 14)
f_tiny = font("DejaVuSans.ttf", 12)
f_name = font("DejaVuSansMono.ttf", 11)
f_big = font("DejaVuSans-Bold.ttf", 30)
f_huge = font("DejaVuSans-Bold.ttf", 52)
f_mono = font("DejaVuSansMono.ttf", 14)

BG, PANEL, PANEL2 = (12, 15, 30), (20, 24, 43), (28, 33, 58)
INK, INK2, MUTED, FAINT = (240, 239, 236), (195, 194, 183), (130, 132, 150), (62, 66, 92)
NODE = (128, 142, 186)  # not matched yet: muted, so each hop colour stands out
GOLD = (250, 204, 21)
HOP = [(250, 204, 21), (56, 189, 248), (129, 140, 248), (192, 132, 252)]  # anchors, hop 1, 2, 3
A_COL, B_COL = (96, 165, 250), (244, 114, 182)
B_NODE = (186, 128, 162)
# verdicts: neutral, not traffic-light
PASS, FAIL, AMBER, DARK = (45, 212, 191), (104, 112, 140), (245, 180, 60), (52, 57, 78)
STATUS_COL = {"confirmed": PASS, "wrong": AMBER, "unchecked": AMBER, "insufficient": FAIL, "unmatched": DARK}
STATUS_Z = {"unmatched": 0, "insufficient": 1, "confirmed": 2, "wrong": 3, "unchecked": 3}
LANG = {"c": "C", "go": "Go", "rust": "Rust", "cpp": "C++"}

STEPS = ["1 Read", "2 Match", "3 Extend", "4 Fingerprint check", "5 Answer key"]
# seconds: read, match, offset, each hop, fingerprint check, answer key + zoom out, hold
T_READ, T_MATCH, T_OFF, T_HOP, T_VER, T_KEY, T_HOLD = 7.0, 9.0, 6.0, 4.5, 21.0, 12.0, 6.0


def lerp(a, b, t):
    return a + (b - a) * t


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def mix(c1, c2, t):
    return tuple(int(lerp(x, y, max(0.0, min(1.0, t)))) for x, y in zip(c1, c2))


def fade(c, a):
    return mix(BG, c, a)


# ---------------------------------------------------------------------------------------------------------------------
def layout(d):
    """Radial tree: anchors at the centre, hop k on ring k, each subtree in its own angular span."""
    na = len(d["nodes_a"])
    tier = [n["tier"] for n in d["nodes_a"]]
    adj = {i: set() for i in range(na)}
    for u, v in d["edges_a"]:
        adj[u].add(v)
        adj[v].add(u)
    parent = {}
    for p in d["pairs"]:
        via = p["why"].get("via") or []
        cands = [u for u in via if tier[u] is not None and tier[u] == p["tier"] - 1]
        if cands:
            parent[p["a"]] = cands[0]
    for i in range(na):
        if tier[i] is None:  # never matched: hang off its deepest drawn neighbour
            nb = [u for u in adj[i] if tier[u] is not None]
            parent[i] = max(nb, key=lambda u: tier[u])
    kids = {i: [] for i in range(na)}
    for c, p in parent.items():
        kids[p].append(c)
    depth = {}

    def dep(i):
        if i not in depth:
            depth[i] = 0 if tier[i] == 0 else dep(parent[i]) + 1
        return depth[i]

    for i in range(na):
        dep(i)

    def leaves(i):
        return max(1, sum(leaves(c) for c in kids[i]))

    roots = [i for i in range(na) if tier[i] == 0]
    ang = {}

    def assign(i, a0, a1):
        ang[i] = (a0 + a1) / 2
        tot = sum(leaves(c) for c in kids[i]) or 1
        x = a0
        for c in sorted(kids[i], key=lambda c: (tier[c] is None, c)):
            w = (a1 - a0) * leaves(c) / tot
            assign(c, x, x + w)
            x += w

    tot = sum(leaves(r) for r in roots)
    x = -math.pi / 2 - 0.35
    for r in roots:
        w = 2 * math.pi * leaves(r) / tot
        assign(r, x, x + w)
        x += w
    gaps = {1: 0.42, 2: 0.24, 3: 0.17, 4: 0.15}
    for k, gap in gaps.items():  # keep the DFS order, push neighbours on a ring apart to a minimum gap
        ring = sorted((i for i in range(na) if depth[i] == k), key=lambda i: ang[i])
        for _ in range(300):
            moved = False
            for x, y in zip(ring, ring[1:] + ring[:1]):
                g = (ang[y] - ang[x]) % (2 * math.pi) if len(ring) > 1 else 9
                if g < gap:
                    ang[x] -= (gap - g) / 2
                    ang[y] += (gap - g) / 2
                    moved = True
            if not moved:
                break
    rings = [0, 92, 160, 218, 262]
    pos = {}
    for i in range(na):
        if depth[i] == 0:
            k = roots.index(i)
            pos[i] = (-34 + 68 * k, 0.0)
        else:
            r = rings[depth[i]]
            pos[i] = (1.04 * r * math.cos(ang[i]), 0.86 * r * math.sin(ang[i]))
    return pos, parent, depth


class Film:
    def __init__(self, d, contrast):
        self.d, self.c = d, contrast
        self.pos_a, self.parent, self.depth = layout(d)
        self.na, self.nb = len(d["nodes_a"]), len(d["nodes_b"])
        self.pair_of_a = {p["a"]: p for p in d["pairs"]}
        self.pair_of_b = {p["b"]: p for p in d["pairs"]}
        self.cxA, self.cxB, self.cy = 322, 958, 352
        rng = random.Random(5)
        # the port's functions start scrambled: before matching, nothing says which is which
        self.scr = {}
        for j in range(self.nb):
            for _ in range(200):
                x, y = rng.uniform(-255, 255), rng.uniform(-200, 200)
                if all((x - a) ** 2 + (y - b) ** 2 > 34**2 for a, b in self.scr.values()):
                    break
            self.scr[j] = (x, y)
        self.final_b = {}
        for p in d["pairs"]:
            self.final_b[p["b"]] = self.pos_a[p["a"]]
        badj = {j: set() for j in range(self.nb)}
        for u, v in d["edges_b"]:
            badj[u].add(v)
            badj[v].add(u)
        extra_on = {}
        for j in range(self.nb):
            if j not in self.final_b:  # an unmatched port function: next to a drawn neighbour, one ring further out
                nb = [u for u in badj[j] if u in self.final_b]
                host = nb[0] if nb else 0
                k = extra_on.get(host, 0)
                extra_on[host] = k + 1
                bx, by = self.final_b[host]
                r = math.hypot(bx / 1.04, by / 0.86)
                a = math.atan2(by / 0.86, bx / 1.04) if r > 1 else math.pi / 2
                a += 0.30 * (k - 0.5) if r > 1 else 0.6 * k
                r = min(r + 62, 240)
                self.final_b[j] = (1.04 * r * math.cos(a), 0.86 * r * math.sin(a))
        s = d["summary"]
        self.tiers = s["tiers"]
        cum, acc = [], 0
        for t in self.tiers:
            acc += t["n"]
            cum.append(acc)
        self.cum = cum
        self.sweep = sorted(d["pairs"], key=lambda p: (p["tier"], p["a"]))
        self.sweep_rank = {id(p): k for k, p in enumerate(self.sweep)}
        self.labels = {
            "a": self.place_labels(
                [n["name"] for n in d["nodes_a"]], [self.pos_a[i] for i in range(self.na)], self.cxA
            ),
            "b": self.place_labels(
                [n["name"] for n in d["nodes_b"]], [self.final_b[j] for j in range(self.nb)], self.cxB
            ),
        }

    def place_labels(self, names, rel, cx):
        """Greedy: each label takes the first candidate spot that stays in the panel and clears the nodes and the
        labels already placed. Outward from the centre is tried first."""
        probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        nodes = [(cx + x, self.cy + y) for x, y in rel]
        boxes, out = [], []
        order = sorted(range(len(names)), key=lambda k: math.hypot(*rel[k]))
        res = {}
        for k in order:
            x, y = nodes[k]
            rx, ry = rel[k]
            out_ang = math.atan2(ry, rx) if (abs(rx) > 1 or abs(ry) > 1) else -math.pi / 2
            if abs(rx) < 40 and abs(ry) < 5:
                out_ang = -math.pi / 2 if rx < 0 else math.pi / 2
            cands = []
            for dist in (12, 20, 30, 42):
                for da in (0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.4, -2.4, math.pi):
                    a = out_ang + da
                    cands.append((dist * math.cos(a), dist * 0.95 * math.sin(a)))
            best, best_hits = None, 1 << 30
            for dx, dy in cands:  # in the panel always; fewest collisions, first (most outward) on a tie
                anc = ("l" if dx > 4 else "r" if dx < -4 else "m") + ("s" if dy < -5 else "t" if dy > 5 else "m")
                bb = probe.textbbox((x + dx, y + dy), names[k], font=f_name, anchor=anc)
                bb = (bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1)
                if bb[0] < cx - 304 or bb[2] > cx + 304 or bb[1] < 112 or bb[3] > 560:
                    continue
                hits = sum(not (bb[2] < o[0] or bb[0] > o[2] or bb[3] < o[1] or bb[1] > o[3]) for o in boxes)
                hits += sum(
                    bb[0] - 8 < nx < bb[2] + 8 and bb[1] - 8 < ny < bb[3] + 8
                    for j, (nx, ny) in enumerate(nodes)
                    if j != k
                )
                if hits < best_hits:
                    best, best_hits = (x + dx, y + dy, anc, bb), hits
                    if hits == 0:
                        break
            if best is None:
                raise SystemExit(f"refusing: no place inside the panel for the label {names[k]}")
            boxes.append(best[3])
            res[k] = best[:3]
        out = [res[k] for k in range(len(names))]
        return out

    def shape_popup(self, d, xy, node, col, f):
        if f <= 0:
            return
        x, y = xy
        bx0 = x - 120
        by0 = y - 160 if y > 280 else y + 24
        w, h = 250, 112
        d.rounded_rectangle(
            (bx0, by0, bx0 + w, by0 + h), 8, fill=(30, 35, 62, int(240 * f)), outline=(*col, int(255 * f)), width=1
        )
        d.text((bx0 + 8, by0 + 6), "name hidden · its shape:", font=f_tiny, fill=(*INK2, int(255 * f)))
        kw = node["kw"]
        mx = max(kw) or 1
        bw = (w - 16) / len(kw)
        for k, v in enumerate(kw):
            hh = 56 * v / mx
            d.rectangle(
                (bx0 + 8 + k * bw, by0 + 82 - hh, bx0 + 8 + (k + 1) * bw - 1.5, by0 + 82), fill=(*col, int(230 * f))
            )
        d.line((bx0 + 8, by0 + 83, bx0 + w - 8, by0 + 83), fill=(*MUTED, int(200 * f)))
        d.text(
            (bx0 + 8, by0 + 90), "32 keyword families (+ size, args, calls)", font=f_tiny, fill=(*MUTED, int(255 * f))
        )
        d.line((x, y - 9, x, by0 + h) if by0 < y else (x, y + 9, x, by0), fill=(*col, int(160 * f)), width=1)

    def offset_card(self, d, t):
        f = ease((t - self.t0("off")) / 0.6) * (1 - ease((t - self.t0("hop1") + 0.5) / 0.5))
        x0, y0, w, h = W // 2 - 300, 170, 600, 300
        d.rounded_rectangle(
            (x0, y0, x0 + w, y0 + h), 14, fill=(26, 30, 54, int(245 * f)), outline=(*GOLD, int(200 * f)), width=2
        )
        a = int(255 * f)
        d.text((x0 + 20, y0 + 16), "Before extending: the language offset", font=f_head, fill=(*GOLD, a))
        d.text(
            (x0 + 20, y0 + 42),
            "The median difference between the 66 anchor pairs is how C and Go",
            font=f_small,
            fill=(*INK2, a),
        )
        d.text(
            (x0 + 20, y0 + 60),
            "write the same code differently. It is subtracted before comparing.",
            font=f_small,
            fill=(*INK2, a),
        )
        rows = [("size (log lines)", self.d["delta_size"])] + [
            (x["label"], x["shift"]) for x in sorted(self.d["delta"], key=lambda x: -abs(x["shift"]))[:4]
        ]
        mx = max(abs(v) for _, v in rows)
        for k, (lab, v) in enumerate(rows):
            yy = y0 + 100 + k * 30
            d.text((x0 + 200, yy), lab, font=f_small, fill=(*INK, a), anchor="rm")
            cx = x0 + 390
            bw = 170 * abs(v) / mx
            d.line((cx, yy - 11, cx, yy + 11), fill=(*MUTED, a))
            d.rectangle(
                (cx, yy - 8, cx + bw, yy + 8) if v > 0 else (cx - bw, yy - 8, cx, yy + 8),
                fill=(*(A_COL if v > 0 else B_COL), a),
            )
            d.text(
                (cx + (bw + 8 if v > 0 else -bw - 8), yy),
                f"{v:+.2f}",
                font=f_tiny,
                fill=(*INK2, a),
                anchor="lm" if v > 0 else "rm",
            )
        d.text((x0 + 390 - 8, y0 + 252), "◀ more in Go", font=f_tiny, fill=(*B_COL, a), anchor="rm")
        d.text((x0 + 390 + 8, y0 + 252), "more in C ▶", font=f_tiny, fill=(*A_COL, a), anchor="lm")
        ab = self.d["ablation"]
        d.text(
            (x0 + 20, y0 + 272),
            f"Small for this pair: {ab['extend_right']} right with it, {ab['extend_no_offset_right']} without.",
            font=f_small,
            fill=(*INK2, a),
        )

    # time of each part
    ORDER = ("read", "match", "off", "hop1", "hop2", "hop3", "ver", "key", "hold")
    DUR = (T_READ, T_MATCH, T_OFF, T_HOP, T_HOP, T_HOP, T_VER, T_KEY, T_HOLD)

    def t0(self, part):
        return sum(self.DUR[: self.ORDER.index(part)])

    @property
    def total(self):
        return self.t0("hold") + T_HOLD

    def matched_at(self, t):
        """Tier reached by time t (-1: nothing matched) and progress inside the current reveal."""
        if t < self.t0("match") + 4.5:
            return -1, 0.0
        if t < self.t0("hop1"):
            return 0, ease((t - self.t0("match") - 4.5) / 2.0)
        for k in (1, 2, 3):
            s = self.t0(f"hop{k}")
            if t < s + T_HOP:
                return k, ease((t - s - 0.6) / 2.6)
        return 3, 1.0

    def node_b_pos(self, j, t):
        p = self.pair_of_b.get(j)
        tier, prog = self.matched_at(t)
        if p is not None:
            if p["tier"] < tier or (p["tier"] == tier and prog >= 1):
                f = 1.0
            elif p["tier"] == tier:
                f = prog
            else:
                f = 0.0
        else:
            f = ease((t - self.t0("hop3") - 2.6) / 1.5)
        a, b = self.scr[j], self.final_b[j]
        return lerp(a[0], b[0], ease(f)), lerp(a[1], b[1], ease(f))

    def step(self, t):
        for k, part in enumerate(("match", "off", "ver", "key")):
            if t < self.t0(part):
                return k if part != "ver" else 2
        return 4

    # the per-pair check sweeps through the drawn pairs, anchors first
    SWEEP_START, SWEEP_EACH = 7.6, 0.34

    def sweep_t(self, k):
        return self.t0("ver") + self.SWEEP_START + self.SWEEP_EACH * k

    def checked(self, p, t):
        """0..1: how far the drawn pair's verdict has been revealed."""
        return ease((t - self.sweep_t(self.sweep_rank[id(p)]) - 0.35) / 0.4)

    def zoom(self, t):
        return ease((t - self.t0("key") - 4.0) / 3.5)

    # -----------------------------------------------------------------------------------------------------------
    def frame(self, t):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im, "RGBA")
        st = self.step(t)
        d.text((W // 2, 22), "How sfs finds a port: match, extend, fingerprint", font=f_title, fill=INK, anchor="mm")
        cw = 200
        x0 = W // 2 - cw * len(STEPS) // 2
        for k, s in enumerate(STEPS):
            on = k == st
            done = k < st
            col = GOLD if on else (INK2 if done else MUTED)
            d.rounded_rectangle(
                (x0 + k * cw + 6, 42, x0 + (k + 1) * cw - 6, 68),
                13,
                fill=PANEL2 if on else PANEL,
                outline=GOLD if on else FAINT,
                width=2 if on else 1,
            )
            d.text((x0 + k * cw + cw // 2, 55), ("✓ " if done else "") + s, font=f_chip, fill=col, anchor="mm")
        z = self.zoom(t)
        if z < 1:
            pa = 1 - z
            src = self.d["source"]
            for cx, side, col in ((self.cxA, "a", A_COL), (self.cxB, "b", B_COL)):
                d.rounded_rectangle((cx - 308, 80, cx + 308, 588), 14, fill=fade(PANEL, pa))
                s = src[side]
                d.text((cx - 292, 92), f"{s['name']}", font=f_head, fill=fade(col, pa))
                lang = LANG.get(s["language"], s["language"])
                d.text(
                    (cx + 292, 94),
                    f"{lang} · {s['functions']} functions",
                    font=f_small,
                    fill=fade(INK2, pa),
                    anchor="ra",
                )
            d.text((self.cxA - 292, 566), "the original", font=f_tiny, fill=fade(MUTED, pa))
            d.text((self.cxB - 292, 566), "a port into another language", font=f_tiny, fill=fade(MUTED, pa))
            d.text(
                (self.cxB + 292, 566),
                "a few of each codebase's functions are drawn",
                font=f_tiny,
                fill=fade(MUTED, pa),
                anchor="ra",
            )
            self.graphs(d, t, pa)
        if z > 0:
            self.full_map(d, t, z)
        if self.t0("off") <= t < self.t0("hop1"):
            self.offset_card(d, t)
        v0 = self.t0("ver")
        if v0 <= t < v0 + self.SWEEP_START:
            self.check_cards(d, t)
        if v0 + T_VER - 5.6 <= t < self.t0("key") + 0.4:
            self.verdict_card(d, t)
        self.caption(d, t)
        self.counter(d, t)
        d.text((W - 18, H - 8), "GitGalaxy · sfs", font=f_smallb, fill=(240, 239, 236, 210), anchor="rd")
        return im

    def pair_colour(self, p, t, base):
        """Hop colour once matched; teal or slate once checked; teal or amber once graded."""
        c = self.checked(p, t)
        col = mix(base, PASS if p["sufficient"] else FAIL, c)
        key = ease((t - self.t0("key")) / 1.2)
        if p["sufficient"] and key > 0 and p["grade"] != "right":
            col = mix(col, AMBER, key)
        return col

    def graphs(self, d, t, pa=1.0):
        appear = ease(t / 1.5) * pa
        tier, prog = self.matched_at(t)
        PA = {i: (self.cxA + x, self.cy + y) for i, (x, y) in self.pos_a.items()}
        PB = {j: (self.cxB + x, self.cy + y) for j, (x, y) in ((j, self.node_b_pos(j, t)) for j in range(self.nb))}
        self.PA = PA

        def a_state(i):
            p = self.pair_of_a.get(i)
            if p is None:
                return None, 0.0
            if p["tier"] < tier or (p["tier"] == tier and prog >= 1):
                return p, 1.0
            if p["tier"] == tier:
                return p, prog
            return None, 0.0

        # hop wave: the calls the extension walks light up while that hop runs
        for edges, P, side in ((self.d["edges_a"], PA, "a"), (self.d["edges_b"], PB, "b")):
            for u, v in edges:
                col = fade(FAINT, appear)
                w = 1
                if side == "a" and tier >= 1 and 0 < prog < 1:
                    pu, pv = self.pair_of_a.get(u), self.pair_of_a.get(v)
                    if pu and pv and {pu["tier"], pv["tier"]} == {tier - 1, tier}:
                        col, w = mix(FAINT, HOP[tier], 0.9), 2
                d.line((P[u], P[v]), fill=col, width=w)
        lab = ease((t - self.t0("key") - 0.8) / 1.2) * pa
        dim_after = self.t0("hop3") + 3
        for i in range(self.na):
            x, y = PA[i]
            p, f = a_state(i)
            col = NODE
            if p is not None:
                col = self.pair_colour(p, t, mix(NODE, HOP[p["tier"]], f))
            elif self.d["nodes_a"][i]["tier"] is None and t > dim_after:
                col = mix(NODE, DARK, ease((t - dim_after) / 1.0))
            big = p is not None and p["tier"] == 0 and f > 0
            r = 9 if big else 7
            if big and t < self.t0("ver"):
                d.ellipse((x - r - 5, y - r - 5, x + r + 5, y + r + 5), outline=(*GOLD, int(120 * f * pa)), width=2)
            d.ellipse((x - r, y - r, x + r, y + r), fill=fade(col, appear))
        for j in range(self.nb):
            x, y = PB[j]
            p = self.pair_of_b.get(j)
            col = B_NODE
            if p is not None:
                _, f = a_state(p["a"])
                col = self.pair_colour(p, t, mix(B_NODE, HOP[p["tier"]], f))
            elif t > dim_after:
                col = mix(B_NODE, DARK, ease((t - dim_after) / 1.0))
            r = 9 if (p is not None and p["tier"] == 0 and a_state(p["a"])[1] > 0) else 7
            d.ellipse((x - r, y - r, x + r, y + r), fill=fade(col, appear))
        if lab > 0:
            for side, nodes in (("a", self.d["nodes_a"]), ("b", self.d["nodes_b"])):
                for k, n in enumerate(nodes):
                    lx, ly, anc = self.labels[side][k]
                    d.text((lx, ly), n["name"], font=f_name, fill=(*INK, int(235 * lab)), anchor=anc)
        m0 = self.t0("match")
        # match: every drawn function whose transport partner picks it back gets a ring; most fade, anchors stay
        if m0 + 0.4 <= t < self.t0("off"):
            rec = [i for i, n in enumerate(self.d["nodes_a"]) if n["reciprocal"]]
            for k, i in enumerate(rec):
                on = ease((t - m0 - 0.4 - 0.12 * k) / 0.4)
                p = self.pair_of_a.get(i)
                off = 0.0 if (p is not None and p["tier"] == 0) else ease((t - m0 - 4.6) / 0.8)
                a = on * (1 - off)
                if a > 0:
                    x, y = PA[i]
                    d.ellipse((x - 13, y - 13, x + 13, y + 13), outline=(*INK, int(200 * a)), width=2)
        # anchor links across the gap while matching
        if m0 + 4.5 <= t < self.t0("off") + 0.5:
            f = ease((t - m0 - 4.5) / 1.5) * (1 - ease((t - self.t0("off")) / 0.5))
            for p in self.d["pairs"]:
                if p["tier"] == 0:
                    a, b = PA[p["a"]], PB[p["b"]]
                    mid = ((a[0] + b[0]) / 2, min(a[1], b[1]) - 120)
                    pts = [
                        (
                            (1 - s) ** 2 * a[0] + 2 * (1 - s) * s * mid[0] + s * s * b[0],
                            (1 - s) ** 2 * a[1] + 2 * (1 - s) * s * mid[1] + s * s * b[1],
                        )
                        for s in [k / 30 for k in range(31)]
                    ]
                    d.line(pts, fill=(*GOLD, int(220 * f)), width=2)
        # read: one function from each side opens into its shape
        r0 = 2.0
        if r0 <= t < self.t0("match") + 0.3:
            f = ease((t - r0) / 1.0) * (1 - ease((t - self.t0("match")) / 0.3))
            self.shape_popup(d, PA[0], self.d["nodes_a"][0], A_COL, f)
            self.shape_popup(d, PB[0], self.d["nodes_b"][0], B_COL, f)
        # check: each pair's fingerprint pops up beside it with its verdict
        for p in self.sweep:
            s = self.sweep_t(self.sweep_rank[id(p)])
            if s <= t < s + 1.1:
                f = ease((t - s) / 0.2) * (1 - ease((t - s - 0.8) / 0.3))
                self.mini_print(d, PA[p["a"]], p, f, self.checked(p, t))
                self.mini_print(d, PB[p["b"]], p, f, self.checked(p, t))

    def mini_print(self, d, xy, p, f, verdict):
        """A small fingerprint: the pair's two 32-family mixes, one above the other, and the verdict."""
        if f <= 0:
            return
        x, y = xy
        w, h = 74, 40
        x0, y0 = x - w / 2, y - h - 16
        ka, kb = self.d["nodes_a"][p["a"]]["kw"], self.d["nodes_b"][p["b"]]["kw"]
        a = int(255 * f)
        col = mix(INK2, PASS if p["sufficient"] else FAIL, verdict)
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), 5, fill=(26, 30, 54, int(235 * f)), outline=(*col, a), width=2)
        bw = (w - 8) / len(ka)
        for k in range(len(ka)):
            ha, hb = 13 * logshare(ka[k]), 13 * logshare(kb[k])
            xx = x0 + 4 + k * bw
            d.rectangle((xx, y0 + 18 - ha, xx + bw - 0.6, y0 + 18), fill=(*A_COL, a))
            d.rectangle((xx, y0 + 21, xx + bw - 0.6, y0 + 21 + hb), fill=(*B_COL, a))
        if verdict > 0.5:
            mark = "✓" if p["sufficient"] else "✗"
            d.text((x0 + w + 3, y0 + h / 2), mark, font=f_smallb, fill=(*col, a), anchor="lm")

    def full_map(self, d, t, z):
        """Zoom out: every original function on its own call graph, coloured by its final verdict."""
        fm = self.d["full_map"]
        cx, cy, sx, sy = W / 2, 338, 590, 238
        P = [(cx + x * sx, cy + y * sy) for x, y in fm["xy"]]
        drawn = {q: i for i, q in enumerate(fm["drawn"])}
        if getattr(self, "PA", None):
            for q, i in drawn.items():
                a, b = self.PA[i], P[q]
                P[q] = (lerp(a[0], b[0], z), lerp(a[1], b[1], z))
        ea = int(70 * z)
        for u, v in fm["edges"]:
            d.line((P[u], P[v]), fill=(*FAINT, ea), width=1)
        order = sorted(range(len(P)), key=lambda q: STATUS_Z[fm["status"][q]])
        for q in order:
            x, y = P[q]
            st = fm["status"][q]
            col = STATUS_COL[st]
            r = 4.2 if st in ("confirmed", "wrong", "unchecked") else 3.2
            a = z if q not in drawn else 1.0
            d.ellipse((x - r, y - r, x + r, y + r), fill=(*col, int(255 * a)))
        la = ease((t - self.t0("key") - 7.0) / 1.0)
        if la <= 0:
            return
        d.text(
            (cx - sx, cy + 0.80 * sy),
            f"{fm['no_calls']} functions with no resolved call",
            font=f_tiny,
            fill=(*MUTED, int(255 * la)),
        )
        c = fm["counts"]
        rows = [
            ("confirmed", "sufficient evidence, same function by name", c.get("confirmed", 0)),
            ("wrong", "sufficient evidence, not confirmed by name", c.get("wrong", 0) + c.get("unchecked", 0)),
            ("insufficient", "matched, insufficient evidence", c.get("insufficient", 0)),
            ("unmatched", "no partner found", c.get("unmatched", 0)),
        ]
        x0, y0 = 880, 600
        for k, (st, lab, n) in enumerate(rows):
            yy = y0 + k * 22
            d.ellipse((x0, yy - 6, x0 + 12, yy + 6), fill=(*STATUS_COL[st], int(255 * la)))
            d.text((x0 + 20, yy), f"{n:>3}", font=f_smallb, fill=(*INK, int(255 * la)), anchor="lm")
            d.text((x0 + 56, yy), lab, font=f_small, fill=(*INK2, int(255 * la)), anchor="lm")

    def check_cards(self, d, t):
        """Two real pairs checked in full: keyword families on a log scale, and the evidence against decoys."""
        v0 = self.t0("ver")
        f = ease((t - v0 - 0.2) / 0.5) * (1 - ease((t - v0 - self.SWEEP_START + 0.6) / 0.5))
        if f <= 0:
            return
        a = int(255 * f)
        ev = self.d["summary"]["evidence"]
        for k, p in enumerate((self.d["pairs"][self.d["pass_pair"]], self.d["pairs"][self.d["fail_pair"]])):
            x0, y0, w, h = 46 + k * 600, 96, 588, 484
            ok = p["sufficient"]
            col = PASS if ok else FAIL
            d.rounded_rectangle(
                (x0, y0, x0 + w, y0 + h), 14, fill=(22, 26, 48, int(248 * f)), outline=(*col, a), width=2
            )
            d.text(
                (x0 + 20, y0 + 14),
                f"Matched pair {'A' if k == 0 else 'B'} (hop {p['tier']})",
                font=f_head,
                fill=(*INK, a),
            )
            d.text((x0 + w - 20, y0 + 16), "names still hidden", font=f_tiny, fill=(*MUTED, a), anchor="ra")
            d.text((x0 + 20, y0 + 40), "■ original (C)", font=f_tiny, fill=(*A_COL, a))
            d.text((x0 + 120, y0 + 40), "■ port (Go)", font=f_tiny, fill=(*B_COL, a))
            ka, kb = self.d["nodes_a"][p["a"]]["kw"], self.d["nodes_b"][p["b"]]["kw"]
            fam = sorted((j for j in range(len(ka)) if ka[j] > 0 or kb[j] > 0), key=lambda j: -max(ka[j], kb[j]))[:8]
            bx, bl = x0 + 170, w - 220
            grow = ease((t - v0 - 0.6) / 0.6)
            for r, j in enumerate(fam):
                yy = y0 + 66 + r * 34
                d.text((bx - 10, yy + 7), self.d["signals"][j], font=f_small, fill=(*INK, a), anchor="rm")
                la, lb = bl * logshare(ka[j]) * grow, bl * logshare(kb[j]) * grow
                d.rectangle((bx, yy, bx + max(1, la), yy + 6), fill=(*A_COL, a))
                d.rectangle((bx, yy + 8, bx + max(1, lb), yy + 14), fill=(*B_COL, a))
                d.text((bx + la + 6, yy + 3), f"{ka[j]:.0%}", font=f_tiny, fill=(*A_COL, a), anchor="lm")
                d.text((bx + lb + 6, yy + 12), f"{kb[j]:.0%}", font=f_tiny, fill=(*B_COL, a), anchor="lm")
            ya = y0 + 66 + len(fam) * 34 + 4
            for share in (0.01, 0.1, 1.0):
                xx = bx + bl * logshare(share)
                d.line((xx, ya - 4, xx, ya + 2), fill=(*MUTED, a))
                d.text((xx, ya + 4), f"{share:.0%}", font=f_tiny, fill=(*MUTED, a), anchor="ma")
            d.text((bx, ya + 20), "share of the function's keyword hits · log scale", font=f_tiny, fill=(*MUTED, a))
            # evidence meter: lower is stronger; the threshold comes from decoys
            my, mx0, mw, top = y0 + h - 70, x0 + 20, w - 40, 2.0
            ef = ease((t - v0 - 2.2) / 0.8)
            ea = int(255 * f * ef)
            d.text(
                (mx0, my - 26),
                f"keyword-mix cosine {p['kw_cos']:.2f} · evidence distance {p['evidence']:.2f}",
                font=f_small,
                fill=(*INK2, ea),
            )
            d.rounded_rectangle((mx0, my, mx0 + mw, my + 10), 5, fill=(*PANEL2, ea))
            tx = mx0 + mw * min(ev["threshold"], top) / top
            d.rounded_rectangle((mx0, my, tx, my + 10), 5, fill=(*PASS, int(90 * f * ef)))
            d.line((tx, my - 6, tx, my + 16), fill=(*INK, ea), width=2)
            d.text((tx, my + 18), "decoy threshold", font=f_tiny, fill=(*INK2, ea), anchor="ma")
            px = mx0 + mw * min(p["evidence"], top) / top
            d.ellipse((px - 8, my - 3, px + 8, my + 13), fill=(*col, ea))
            vf = ease((t - v0 - 3.4) / 0.6)
            verdict = "SUFFICIENT EVIDENCE" if ok else "INSUFFICIENT EVIDENCE"
            d.text((x0 + w - 20, y0 + 14 + 26), verdict, font=f_smallb, fill=(*col, int(255 * f * vf)), anchor="ra")

    def verdict_card(self, d, t):
        s0 = self.t0("ver") + T_VER - 5.6
        f = ease((t - s0) / 0.5) * (1 - ease((t - self.t0("key")) / 0.4))
        if f <= 0:
            return
        a = int(255 * f)
        g, r = self.d["summary"], self.c["summary"]
        x0, y0, w, h = 140, 150, 1000, 330
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), 14, fill=(22, 26, 48, int(248 * f)), outline=(*PASS, a), width=2)
        d.text((x0 + 24, y0 + 18), "Every matched pair, checked the same way", font=f_head, fill=(*INK, a))
        tiles = [
            (
                f"{g['b']} (Go port)",
                f"{g['evidence']['sufficient']}",
                f"of {g['matched']} pairs: sufficient evidence",
                PASS,
            ),
            (
                f"{r['b']} (Rust rewrite)",
                f"{r['evidence']['sufficient']}",
                f"of {r['matched']} pairs: sufficient evidence",
                FAIL,
            ),
            ("random pairing", "0", f"of {g['functions_a']} pairs the same function", FAIL),
        ]
        for k, (lab, big, sub, col) in enumerate(tiles):
            tx = x0 + 24 + k * 325
            tf = ease((t - s0 - 0.4 - 0.5 * k) / 0.5)
            ta = int(255 * f * tf)
            d.rounded_rectangle((tx, y0 + 56, tx + 300, y0 + 196), 10, fill=(*PANEL2, int(240 * f * tf)))
            d.text((tx + 16, y0 + 70), lab, font=f_smallb, fill=(*INK2, ta))
            d.text((tx + 16, y0 + 96), big, font=f_huge, fill=(*col, ta))
            d.text((tx + 16, y0 + 164), sub, font=f_small, fill=(*INK2, ta))
        ev = g["evidence"]
        lines = [
            (
                "The threshold is set by decoys: functions of the same kind with their measurements shuffled. At it, "
                f"about {ev['decoy_passing_per_draw']:.1f} decoy"
            ),
            (
                f"pairs would pass per {g['matched']}, a false-pass rate under {ev['fdr']:.0%}. Unrelated code makes "
                "pairs too; none of them pass."
            ),
        ]
        la = int(255 * f * ease((t - s0 - 2.0) / 0.6))
        for k, line in enumerate(lines):
            d.text((x0 + 24, y0 + 222 + 24 * k), line, font=f_small, fill=(*INK2, la))

    def counter(self, d, t):
        tier, prog = self.matched_at(t)
        if tier < 0 or t >= self.t0("ver"):
            return
        n = self.cum[tier] if prog >= 1 else int(lerp(self.cum[tier - 1] if tier else 0, self.cum[tier], prog))
        tot = self.d["source"]["a"]["functions"]
        x0, x1, y = 900, 1234, 660
        col = HOP[max(0, tier)]
        d.text((x0, y - 30), f"{n}", font=f_big, fill=col, anchor="ls")
        d.text(
            (x0 + d.textlength(f"{n}", font=f_big) + 8, y - 32),
            f"of {tot} matched",
            font=f_small,
            fill=INK2,
            anchor="ls",
        )
        d.rounded_rectangle((x0, y - 16, x1, y - 2), 7, fill=PANEL2)
        d.rounded_rectangle((x0, y - 16, x0 + max(14, (x1 - x0) * n / tot), y - 2), 7, fill=col)

    def caption(self, d, t):
        s, ab = self.d["summary"], self.d["ablation"]
        ev, r = s["evidence"], self.c["summary"]
        v0, k0 = self.t0("ver"), self.t0("key")
        lines = []
        if t < self.t0("match"):
            lines = [
                ("Each function becomes a shape: ", "its size, arguments and 32 keyword-family counts, plus"),
                ("", "where it sits in the call graph. Names are hidden: the port's functions start scrambled."),
            ]
        elif t < self.t0("match") + 4.5:
            lines = [
                ("Match: ", "optimal transport (FGW) pairs functions whose shape AND"),
                ("", f"call-graph position agree. {s['reciprocal_best']} pairs pick each other."),
            ]
        elif t < self.t0("off"):
            lines = [
                ("Anchors: ", f"only the strongest 15% are kept: {s['anchors']} of {s['reciprocal_best']}."),
                ("", "Two of them (gold) and their partners snapping into place."),
            ]
        elif t < self.t0("hop1"):
            lines = [("Extend, first: ", "measure how the two languages differ,"), ("", "from the anchors only.")]
        elif t < v0:
            k = 1 if t < self.t0("hop2") else 2 if t < self.t0("hop3") else 3
            lines = [
                (f"Extend, hop {k}: ", "a caller or callee of a matched function pairs"),
                ("", "with the unclaimed neighbour of its partner closest in shape."),
                ("", "More shared matched neighbours, more weight."),
            ]
            if k == 3 and t > self.t0("hop3") + 3:
                lines = [
                    ("Extend stops ", "after 3 hops. Dark: functions whose"),
                    ("", "neighbours offered no partner."),
                ]
        elif t < v0 + self.SWEEP_START:
            lines = [
                ("Fingerprint check: ", "extend always finds partners, even in unrelated code. So each pair's"),
                ("", "keyword fingerprint is compared, and its evidence must beat decoys to count."),
            ]
        elif t < v0 + T_VER - 5.6:
            lines = [
                ("Every pair is checked: ", "teal = sufficient evidence, slate = insufficient."),
                ("", "Insufficient is not wrong: it is a match the evidence cannot back."),
            ]
        elif t < k0:
            lines = [
                ("Evidence quality ", "is what sfs adds: each match carries a verdict, and the"),
                ("", "verdict is calibrated, so unrelated code and random pairs score zero."),
            ]
        elif t < k0 + 4.0:
            ok = [p for p in self.d["pairs"] if p["sufficient"]]
            n_ok_right = sum(p["grade"] == "right" for p in ok)
            bad = [p for p in self.d["pairs"] if p["grade"] != "right"]
            held = sum(not p["sufficient"] for p in bad)
            lines = [
                ("Answer key: ", "names, never used, now shown. Drawn pairs with sufficient evidence:"),
                (
                    "",
                    (
                        f"{n_ok_right} of {len(ok)} are the same function. Drawn pairs that are not: "
                        f"{held} of {len(bad)} held back."
                    ),
                ),
            ]
        else:
            nc = ev["right"]
            lines = [
                ("All ", f"{s['functions_a']} functions. {ev['sufficient']} pairs had sufficient evidence: {nc} are"),
                (
                    "",
                    (
                        f"the same function by name; {ev['wrong'] + ev['unkeyed']} are not confirmed (amber). "
                        f"Rewrite: {r['evidence']['sufficient']}."
                    ),
                ),
                (
                    "",
                    (
                        f"Without the check: {s['right']} of {s['matched']} matches right. "
                        f"Shape alone: {ab['shape_only_right']} of {ab['shape_only_pairs']}."
                    ),
                ),
            ]
        y = 600
        for b, rest in lines:
            x = 46
            if b:
                d.text((x, y), b, font=f_capb, fill=GOLD)
                x += d.textlength(b, font=f_capb)
            d.text((x, y), rest, font=f_cap, fill=INK)
            y += 25


def logshare(v):
    """0..1 position of a share (0..1) on a log scale from 0.1% to 100%."""
    return math.log10(1 + 1000 * max(0.0, v)) / math.log10(1001)


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="seed_extend_data.py output for the port")
    ap.add_argument("--contrast", required=True, help="seed_extend_data.py --summary-only output for a non-port")
    ap.add_argument("--pass-pair", default="S_StartSound", help="original-side name of the sufficient pair in step 4")
    ap.add_argument("--fail-pair", default="G_WriteDemoTiccmd", help="original-side name of the insufficient pair")
    ap.add_argument("--out", required=True, help="output stem: writes <out>.gif, <out>.mp4, <out>.png")
    ap.add_argument("--still", type=float, help="write one frame at this second to <out>-still.png and stop")
    a = ap.parse_args(argv)
    d = json.loads(pathlib.Path(a.data).read_text())
    c = json.loads(pathlib.Path(a.contrast).read_text())
    if c["summary"]["a"] != d["source"]["a"]["name"]:
        raise SystemExit("refusing: the contrast pair has a different original")
    names = [n["name"] for n in d["nodes_a"]]
    d["pass_pair"] = next(k for k, p in enumerate(d["pairs"]) if names[p["a"]] == a.pass_pair)
    d["fail_pair"] = next(k for k, p in enumerate(d["pairs"]) if names[p["a"]] == a.fail_pair)
    if not d["pairs"][d["pass_pair"]]["sufficient"] or d["pairs"][d["fail_pair"]]["sufficient"]:
        raise SystemExit("refusing: --pass-pair must be a sufficient pair and --fail-pair an insufficient one")
    fm = Film(d, c)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.still is not None:
        fm.frame(a.still).save(str(out) + "-still.png")
        return 0
    n = int(fm.total * FPS)
    fdir = pathlib.Path(tempfile.mkdtemp(prefix="sfs-seedext-"))
    for i in range(n):
        fm.frame(i / FPS).save(fdir / f"f{i:04d}.png")
    ff = [ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(fdir / "f%04d.png")]
    subprocess.run(ff + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out) + ".mp4"], check=True)
    gif_filter = (
        "fps=10,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer"
    )
    subprocess.run([*ff, "-vf", gif_filter, str(out) + ".gif"], check=True)
    shutil.copy(fdir / f"f{n - 1:04d}.png", str(out) + ".png")
    shutil.rmtree(fdir, ignore_errors=True)
    print(f"{out}: {n} frames, {fm.total:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
