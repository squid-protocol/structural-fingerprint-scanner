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
4. FINGERPRINT: each matched pair's 32-family keyword mix is compared, against a rewrite that shares no code;
5. ANSWER KEY: names, never used to match, grade every pair.

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
f_mono = font("DejaVuSansMono.ttf", 14)

BG, PANEL, PANEL2 = (12, 15, 30), (20, 24, 43), (28, 33, 58)
INK, INK2, MUTED, FAINT = (240, 239, 236), (195, 194, 183), (130, 132, 150), (62, 66, 92)
NODE = (128, 142, 186)  # not matched yet: muted, so each hop colour stands out
GOLD = (250, 204, 21)
HOP = [(250, 204, 21), (45, 212, 191), (56, 189, 248), (167, 139, 250)]  # anchors, hop 1, 2, 3
RIGHT, WRONG, UNKEYED = (74, 222, 128), (248, 113, 113), (148, 150, 168)
A_COL, B_COL = (96, 165, 250), (244, 114, 182)

STEPS = ["1 Read", "2 Match", "3 Extend", "4 Fingerprint", "5 Answer key"]
# seconds: read, match, offset, hop1, hop2, hop3, fingerprint, key, hold
T_READ, T_MATCH, T_OFF, T_HOP, T_FP, T_KEY, T_HOLD = 7.0, 9.0, 6.0, 4.5, 13.0, 9.0, 6.0


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
        self.labels = {
            "a": self.place_labels(
                [n["name"] for n in d["nodes_a"]], [self.pos_a[i] for i in range(self.na)], self.cxA
            ),
            "b": self.place_labels(
                [n["name"] for n in d["nodes_b"]], [self.final_b[j] for j in range(self.nb)], self.cxB
            ),
        }

    # time of each part
    def t0(self, part):
        order = ["read", "match", "off", "hop1", "hop2", "hop3", "fp", "key", "hold"]
        dur = [T_READ, T_MATCH, T_OFF, T_HOP, T_HOP, T_HOP, T_FP, T_KEY, T_HOLD]
        return sum(dur[: order.index(part)])

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
        if t < self.t0("match"):
            return 0
        if t < self.t0("off"):
            return 1
        if t < self.t0("fp"):
            return 2
        if t < self.t0("key"):
            return 3
        return 4

    # -----------------------------------------------------------------------------------------------------------
    def frame(self, t):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im, "RGBA")
        st = self.step(t)
        d.text((W // 2, 22), "How sfs finds a port: match, extend, fingerprint", font=f_title, fill=INK, anchor="mm")
        cw = 190
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
        src = self.d["source"]
        for cx, side, col in ((self.cxA, "a", A_COL), (self.cxB, "b", B_COL)):
            d.rounded_rectangle((cx - 308, 80, cx + 308, 588), 14, fill=PANEL)
            s = src[side]
            d.text((cx - 292, 92), f"{s['name']}", font=f_head, fill=col)
            lang = {"c": "C", "go": "Go", "rust": "Rust", "cpp": "C++"}.get(s["language"], s["language"])
            d.text((cx + 292, 94), f"{lang} · {s['functions']} functions", font=f_small, fill=INK2, anchor="ra")
        d.text((self.cxA - 292, 566), "the original", font=f_tiny, fill=MUTED)
        d.text((self.cxB - 292, 566), "a port into another language", font=f_tiny, fill=MUTED)
        d.text(
            (self.cxB + 292, 566), "a few of each codebase's functions are drawn", font=f_tiny, fill=MUTED, anchor="ra"
        )
        self.graphs(d, t)
        if self.t0("off") <= t < self.t0("hop1"):
            self.offset_card(d, t)
        if self.t0("fp") <= t < self.t0("key") + 0.8:
            self.fingerprint_card(d, t)
        self.caption(d, t)
        self.counter(d, t)
        d.text((W - 18, H - 8), "GitGalaxy · sfs", font=f_smallb, fill=(240, 239, 236, 210), anchor="rd")
        return im

    def graphs(self, d, t):
        appear = ease(t / 1.5)
        tier, prog = self.matched_at(t)
        key = ease((t - self.t0("key")) / 1.5)
        PA = {i: (self.cxA + x, self.cy + y) for i, (x, y) in self.pos_a.items()}
        PB = {j: (self.cxB + x, self.cy + y) for j, (x, y) in ((j, self.node_b_pos(j, t)) for j in range(self.nb))}

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
        lab = ease((t - self.t0("key") - 1.0) / 1.5)
        for i in range(self.na):
            x, y = PA[i]
            p, f = a_state(i)
            col = NODE
            if p is not None:
                col = mix(NODE, HOP[p["tier"]], f)
                if key > 0:
                    col = mix(col, {"right": RIGHT, "wrong": WRONG}.get(p["grade"], UNKEYED), key)
            elif self.d["nodes_a"][i]["tier"] is None and t > self.t0("hop3") + 3:
                col = mix(NODE, (90, 94, 118), ease((t - self.t0("hop3") - 3) / 1.0))
            r = 9 if (p is not None and p["tier"] == 0 and f > 0) else 7
            if p is not None and p["tier"] == 0 and f > 0:
                d.ellipse((x - r - 5, y - r - 5, x + r + 5, y + r + 5), outline=(*GOLD, int(120 * f)), width=2)
            d.ellipse((x - r, y - r, x + r, y + r), fill=fade(col, appear))
        for j in range(self.nb):
            x, y = PB[j]
            p = self.pair_of_b.get(j)
            col = (186, 128, 162)
            if p is not None:
                _, f = a_state(p["a"])
                col = mix(col, HOP[p["tier"]], f)
                if key > 0:
                    col = mix(col, {"right": RIGHT, "wrong": WRONG}.get(p["grade"], UNKEYED), key)
            elif t > self.t0("hop3") + 3:
                col = mix(col, (90, 94, 118), ease((t - self.t0("hop3") - 3) / 1.0))
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

    def fingerprint_card(self, d, t):
        s0 = self.t0("fp")
        f = ease((t - s0) / 0.6) * (1 - ease((t - self.t0("key") - 0.2) / 0.6))
        a = int(255 * f)
        x0, y0, w, h = 46, 104, W - 92, 470
        d.rounded_rectangle(
            (x0, y0, x0 + w, y0 + h), 14, fill=(22, 26, 48, int(248 * f)), outline=(*HOP[2], int(200 * f)), width=2
        )
        d.text(
            (x0 + 20, y0 + 16),
            "Fingerprint every matched pair: the mix of its 32 keyword families",
            font=f_head,
            fill=(*INK, a),
        )
        # one real pair, bars mirrored
        p = self.d["pairs"][self.d["fp_pair"]]
        ka, kb = self.d["nodes_a"][p["a"]]["kw"], self.d["nodes_b"][p["b"]]["kw"]
        mx = max(max(ka), max(kb)) or 1
        bx, by, bw = x0 + 24, y0 + 60, 15
        grow = ease((t - s0 - 0.6) / 1.2)
        for k in range(len(ka)):
            xx = bx + k * bw
            ha, hb = 80 * ka[k] / mx * grow, 80 * kb[k] / mx * grow
            d.rectangle((xx, by + 90 - ha, xx + bw - 3, by + 90), fill=(*A_COL, a))
            d.rectangle((xx, by + 94, xx + bw - 3, by + 94 + hb), fill=(*B_COL, a))
        d.text(
            (bx, by + 192),
            "one matched pair (hop " + str(p["tier"]) + "): original above, port below",
            font=f_small,
            fill=(*INK2, a),
        )
        top = sorted(range(len(ka)), key=lambda k: -(ka[k] + kb[k]))[:3]
        d.text(
            (bx, by + 212),
            "biggest families: " + ", ".join(self.d["signals"][k] for k in top),
            font=f_tiny,
            fill=(*MUTED, a),
        )
        g = self.d["summary"]
        notes = [
            "Each bar: the share of the function's keyword hits in one family.",
            "Cosine 1.0 means the same mix. Names play no part.",
            f"The median over all {g['matched']} matched pairs is the 'pair keyword",
            "cosine' in the table, next to the same check on a rewrite.",
        ]
        for k, line in enumerate(notes):
            d.text((bx, by + 262 + 22 * k), line, font=f_small, fill=(*INK2, a))
        d.text((bx + 32 * bw + 18, by + 70), f"{p['kw_cos']:.2f}", font=f_big, fill=(*HOP[2], a))
        d.text((bx + 32 * bw + 18, by + 108), "cosine of", font=f_tiny, fill=(*INK2, a))
        d.text((bx + 32 * bw + 18, by + 122), "the two mixes", font=f_tiny, fill=(*INK2, a))
        # the table: port vs a rewrite that shares no code
        g, r = self.d["summary"], self.c["summary"]
        tf = ease((t - s0 - 3.5) / 1.0)
        ta = int(255 * f * tf)
        tx, ty = x0 + 680, y0 + 56
        d.text((tx, ty), "", font=f_small)
        d.text((tx + 290, ty), g["b"], font=f_smallb, fill=(*B_COL, ta), anchor="ma")
        d.text((tx + 400, ty), r["b"], font=f_smallb, fill=(*UNKEYED, ta), anchor="ma")
        d.text((tx + 290, ty + 18), "Go port", font=f_tiny, fill=(*MUTED, ta), anchor="ma")
        d.text((tx + 400, ty + 18), "Rust rewrite", font=f_tiny, fill=(*MUTED, ta), anchor="ma")
        rows = [
            ("anchors", f"{g['anchors']}", f"{r['anchors']}"),
            ("functions matched", f"{g['coverage']:.0%}", f"{r['coverage']:.0%}"),
            ("pair keyword cosine", f"{g['matched_kw_cos']:.2f}", f"{r['matched_kw_cos']:.2f}"),
            ("same archetype", f"{g['matched_arch_agree']:.0%}", f"{r['matched_arch_agree']:.0%}"),
            ("same top family", f"{g['matched_kw_dom_agree']:.0%}", f"{r['matched_kw_dom_agree']:.0%}"),
        ]
        for k, (lab, v1, v2) in enumerate(rows):
            yy = ty + 50 + k * 32
            d.text((tx, yy), lab, font=f_small, fill=(*INK, ta))
            d.text((tx + 290, yy), v1, font=f_smallb, fill=(*INK, ta), anchor="ma")
            d.text((tx + 400, yy), v2, font=f_smallb, fill=(*INK2, ta), anchor="ma")
        uf = ease((t - s0 - 7.0) / 1.0)
        ua = int(255 * f * uf)
        yy = ty + 50 + 5 * 32 + 14
        d.line((tx, yy - 6, tx + 450, yy - 6), fill=(*FAINT, ua))
        d.text((tx, yy), "whole-codebase mix (JSD)", font=f_small, fill=(*INK, ua))
        d.text((tx + 290, yy), f"{g['kw_jsd']:.2f}", font=f_smallb, fill=(*WRONG, ua), anchor="ma")
        d.text((tx + 400, yy), f"{r['kw_jsd']:.2f}", font=f_smallb, fill=(*WRONG, ua), anchor="ma")
        d.text((tx, yy + 26), "The codebase-wide keyword mix cannot tell a port", font=f_tiny, fill=(*INK2, ua))
        d.text((tx, yy + 42), "from a rewrite. Compared pair by pair along the", font=f_tiny, fill=(*INK2, ua))
        d.text((tx, yy + 58), "matches, it can.", font=f_tiny, fill=(*INK2, ua))
        d.text(
            (x0 + 24, y0 + h - 30),
            f"{r['a']} vs {r['b']}, a from-scratch engine with no declared lineage, run through the same steps.",
            font=f_tiny,
            fill=(*MUTED, ta),
        )

    def counter(self, d, t):
        tier, prog = self.matched_at(t)
        if tier < 0 or t >= self.t0("fp"):
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
        s, g, ab = self.d["summary"], self.d["summary"]["tiers"], self.d["ablation"]
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
        elif t < self.t0("fp"):
            k = 1 if t < self.t0("hop2") else 2 if t < self.t0("hop3") else 3
            lines = [
                (f"Extend, hop {k}: ", "a caller or callee of a matched function pairs"),
                ("", "with the unclaimed neighbour of its partner closest in shape."),
                ("", "More shared matched neighbours, more weight."),
            ]
            if k == 3 and t > self.t0("hop3") + 3:
                lines = [
                    ("Extend stops ", "after 3 hops. Grey: functions whose"),
                    ("", "neighbours offered no partner."),
                ]
        elif t < self.t0("key"):
            lines = [
                ("Fingerprint: ", "matching says which functions correspond; the keyword mix of each"),
                ("", "matched pair says how alike they still are, and whether the match is a port at all."),
            ]
        else:
            right = sum(x["right"] for x in g)
            lines = [
                ("Answer key: ", f"names, never used, grade it: {right} of {s['matched']} pairs"),
                ("", "are the same function. Red: wrong. Grey: no same-named function."),
                ("", f"Shape alone, without the call graph: {ab['shape_only_right']} of {ab['shape_only_pairs']}."),
            ]
        y = 600
        for b, rest in lines:
            x = 46
            if b:
                d.text((x, y), b, font=f_capb, fill=GOLD)
                x += d.textlength(b, font=f_capb)
            d.text((x, y), rest, font=f_cap, fill=INK)
            y += 25
        if t >= self.t0("key"):
            self.key_bars(d, t)

    def key_bars(self, d, t):
        f = ease((t - self.t0("key") - 2.0) / 1.2)
        if f <= 0:
            return
        a = int(255 * f)
        names = ["anchors", "hop 1", "hop 2", "hop 3"]
        d.text((900, 598), "right, by step", font=f_smallb, fill=(*INK, a))
        for k, tr in enumerate(self.d["summary"]["tiers"]):
            yy = 628 + k * 19
            d.text((900, yy), names[k], font=f_small, fill=(*HOP[k], a), anchor="lm")
            bx, bw = 975, 180
            n = tr["n"]
            r_w = bw * tr["right"] / n
            w_w = bw * tr["wrong"] / n
            d.rectangle((bx, yy - 7, bx + r_w, yy + 7), fill=(*RIGHT, a))
            d.rectangle((bx + r_w, yy - 7, bx + r_w + w_w, yy + 7), fill=(*WRONG, a))
            d.rectangle((bx + r_w + w_w, yy - 7, bx + bw, yy + 7), fill=(*UNKEYED, a))
            d.text((bx + bw + 6, yy), f"{tr['right']}/{n}", font=f_tiny, fill=(*INK2, a), anchor="lm")


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
    ap.add_argument("--fp-pair", default="G_Ticker", help="original-side name of the pair shown in step 4")
    ap.add_argument("--out", required=True, help="output stem: writes <out>.gif, <out>.mp4, <out>.png")
    ap.add_argument("--still", type=float, help="write one frame at this second to <out>-still.png and stop")
    a = ap.parse_args(argv)
    d = json.loads(pathlib.Path(a.data).read_text())
    c = json.loads(pathlib.Path(a.contrast).read_text())
    if c["summary"]["a"] != d["source"]["a"]["name"]:
        raise SystemExit("refusing: the contrast pair has a different original")
    names = [n["name"] for n in d["nodes_a"]]
    d["fp_pair"] = next(k for k, p in enumerate(d["pairs"]) if names[p["a"]] == a.fp_pair)
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
    gif_filter = "scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer"
    subprocess.run([*ff, "-vf", gif_filter, str(out) + ".gif"], check=True)
    shutil.copy(fdir / f"f{n - 1:04d}.png", str(out) + ".png")
    shutil.rmtree(fdir, ignore_errors=True)
    print(f"{out}: {n} frames, {fm.total:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
