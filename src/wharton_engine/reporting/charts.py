"""Dependency-free inline-SVG charts for the review packet.

Colours are CSS roles (classes) so light/dark themes swap in one place:
equity=blue, treasury=aqua, cash/reserve=yellow, comparison series=orange,
shortfall/fail=critical red, muted=gray. Every mark carries a data-tip for the
hover tooltip; every chart is accompanied by a table in the packet.
Annual values are drawn as points/steps at beginning-of-year dates - straight
segments between them are labelled as connecting annual observations only.
"""

from __future__ import annotations

import html
import math

W = 640


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def money(v: float, compact: bool = True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    a = abs(v)
    sign = "−" if v < 0 else ""
    if compact and a >= 1e6:
        return f"{sign}${a / 1e6:,.2f}M"
    if compact and a >= 1e3:
        return f"{sign}${a / 1e3:,.0f}K"
    return f"{sign}${a:,.0f}"


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    start = math.floor(lo / step) * step
    ticks = []
    t = start
    while t <= hi + 1e-9 * step:
        ticks.append(round(t, 10))
        t += step
    return ticks


class Scale:
    def __init__(self, d0, d1, r0, r1):
        self.d0, self.d1, self.r0, self.r1 = d0, d1, r0, r1

    def __call__(self, v):
        if self.d1 == self.d0:
            return (self.r0 + self.r1) / 2
        return self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


def _svg(h: int, body: str, label: str, w: int = W) -> str:
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}" '
            f'preserveAspectRatio="xMidYMid meet">{body}</svg>')


def _rounded_col(x, y_base, y_top, w, cls, tip, r=4):
    """Column with a 4px rounded data end, square at the baseline (works for negative columns too)."""
    h = abs(y_base - y_top)
    r = min(r, h / 2, w / 2)
    if y_top <= y_base:   # grows up
        d = (f"M{x},{y_base} V{y_top + r} Q{x},{y_top} {x + r},{y_top} H{x + w - r} "
             f"Q{x + w},{y_top} {x + w},{y_top + r} V{y_base} Z")
    else:                 # grows down
        d = (f"M{x},{y_base} V{y_top - r} Q{x},{y_top} {x + r},{y_top} H{x + w - r} "
             f"Q{x + w},{y_top} {x + w},{y_top - r} V{y_base} Z")
    return f'<path class="{cls}" d="{d}" data-tip="{esc(tip)}"><title>{esc(tip)}</title></path>'


def _rounded_hbar(x0, x1, y, h, cls, tip, r=4):
    w = x1 - x0
    r = min(r, w / 2, h / 2)
    d = f"M{x0},{y} H{x1 - r} Q{x1},{y} {x1},{y + r} V{y + h - r} Q{x1},{y + h} {x1 - r},{y + h} H{x0} Z"
    return f'<path class="{cls}" d="{d}" data-tip="{esc(tip)}"><title>{esc(tip)}</title></path>'


def cashflow_timeline(years, contributions, payments) -> str:
    h, ml, mr, mt, mb = 260, 64, 16, 20, 36
    xs = Scale(0, len(years), ml, W - mr)
    vmax = max(max(contributions), 1)
    vmin = -max(max(payments), 1)
    ys = Scale(vmin * 1.1, vmax * 1.1, h - mb, mt)
    body = []
    for t in [t for t in nice_ticks(vmin, vmax, 6) if vmin * 1.1 <= t <= vmax * 1.1]:
        y = ys(t)
        body.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{y:.1f}" y2="{y:.1f}"/>'
                    f'<text class="tick" x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end">{money(t)}</text>')
    bw = min(24, (W - ml - mr) / len(years) * 0.6)
    y0 = ys(0)
    for i, yr in enumerate(years):
        cx = xs(i + 0.5)
        if contributions[i]:
            body.append(_rounded_col(cx - bw / 2, y0, ys(contributions[i]), bw, "m-eq",
                                     f"{yr}: contribution {money(contributions[i], False)}"))
            body.append(f'<text class="lbl" x="{cx:.1f}" y="{ys(contributions[i]) - 6:.1f}" text-anchor="middle">'
                        f'{money(contributions[i])}</text>')
        if payments[i]:
            body.append(_rounded_col(cx - bw / 2, y0, ys(-payments[i]), bw, "m-cmp",
                                     f"{yr}: operating payment {money(payments[i], False)} (fixed nominal)"))
        if i % 2 == 0 or len(years) <= 10:
            body.append(f'<text class="tick" x="{cx:.1f}" y="{h - mb + 16}" text-anchor="middle">{yr}</text>')
    body.append(f'<line class="axis" x1="{ml}" x2="{W - mr}" y1="{y0:.1f}" y2="{y0:.1f}"/>')
    i33 = list(years).index(2033)
    x33 = xs(i33 + 0.5)
    body.append(f'<line class="ref" x1="{x33:.1f}" x2="{x33:.1f}" y1="{mt}" y2="{h - mb}"/>'
                f'<text class="lbl" x="{x33 + 6:.1f}" y="{mt + 10}">2033: reserve -> facility -> payment</text>')
    return _svg(h, "".join(body), "Client cash-flow timeline")


def stacked_glide(title: str, years, weights) -> str:
    """weights: list of (eq, tsy, cash) per year."""
    w, h, ml, mr, mt, mb = 300, 170, 34, 8, 24, 26
    xs = Scale(0, len(years), ml, w - mr)
    ys = Scale(0, 1, h - mb, mt)
    body = [f'<text class="ttl" x="{ml}" y="14">{esc(title)}</text>']
    for t in (0, 0.5, 1):
        body.append(f'<line class="grid" x1="{ml}" x2="{w - mr}" y1="{ys(t):.1f}" y2="{ys(t):.1f}"/>'
                    f'<text class="tick" x="{ml - 4}" y="{ys(t) + 4:.1f}" text-anchor="end">{t:.0%}</text>')
    bw = min(24, (w - ml - mr) / len(years) - 2)
    for i, (yr, wt) in enumerate(zip(years, weights)):
        x = xs(i + 0.5) - bw / 2
        acc = 0.0
        for v, cls, nm in zip(wt, ("m-eq", "m-tsy", "m-cash"), ("equity", "Treasury", "cash")):
            if v <= 0:
                continue
            y_top, y_bot = ys(acc + v), ys(acc)
            gap = 1 if acc > 0 else 0
            body.append(f'<rect class="{cls}" x="{x:.1f}" y="{y_top:.1f}" width="{bw:.1f}" '
                        f'height="{max(0.0, y_bot - y_top - gap):.1f}" data-tip="{esc(f"{yr} {nm} {v:.0%}")}">'
                        f'<title>{esc(f"{yr} {nm} {v:.0%}")}</title></rect>')
            acc += v
        lab = "2033+" if yr == 2033 else str(yr) if i == 0 else str(yr)[2:]
        body.append(f'<text class="tick" x="{xs(i + 0.5):.1f}" y="{h - mb + 14}" text-anchor="middle">{lab}</text>')
    return _svg(h, "".join(body), title, w)


def fan_chart(title: str, years, bands: dict, ref_line=None, ref_label="") -> str:
    """bands: p5,p25,p50,p75,p95 lists (beginning-of-year nominal wealth)."""
    w, h, ml, mr, mt, mb = 460, 250, 56, 16, 26, 28
    xs = Scale(years[0], years[-1], ml, w - mr)
    top = max(bands["p95"]) * 1.05
    ys = Scale(0, top, h - mb, mt)
    body = [f'<text class="ttl" x="{ml}" y="14">{esc(title)}</text>']
    for t in nice_ticks(0, top, 4):
        body.append(f'<line class="grid" x1="{ml}" x2="{w - mr}" y1="{ys(t):.1f}" y2="{ys(t):.1f}"/>'
                    f'<text class="tick" x="{ml - 4}" y="{ys(t) + 4:.1f}" text-anchor="end">{money(t)}</text>')

    def area(lo, hi, cls, tip):
        pts = [f"{xs(y):.1f},{ys(v):.1f}" for y, v in zip(years, hi)]
        pts += [f"{xs(y):.1f},{ys(v):.1f}" for y, v in zip(reversed(years), list(reversed(lo)))]
        return f'<polygon class="{cls}" points="{" ".join(pts)}"><title>{esc(tip)}</title></polygon>'

    body.append(area(bands["p5"], bands["p95"], "band-outer", "P5-P95 range"))
    body.append(area(bands["p25"], bands["p75"], "band-inner", "P25-P75 range"))
    line = " ".join(f"{xs(y):.1f},{ys(v):.1f}" for y, v in zip(years, bands["p50"]))
    body.append(f'<polyline class="line-eq" points="{line}"/>')
    for y, a, b, c, d, e in zip(years, bands["p5"], bands["p25"], bands["p50"], bands["p75"], bands["p95"]):
        tip = f"{y}: P5 {money(a)} | P25 {money(b)} | median {money(c)} | P75 {money(d)} | P95 {money(e)}"
        body.append(f'<circle class="hit" cx="{xs(y):.1f}" cy="{ys(c):.1f}" r="7" data-tip="{esc(tip)}">'
                    f'<title>{esc(tip)}</title></circle>')
    if ref_line:
        yy = ys(ref_line)
        body.append(f'<line class="ref" x1="{ml}" x2="{w - mr}" y1="{yy:.1f}" y2="{yy:.1f}"/>'
                    f'<text class="lbl" x="{w - mr}" y="{yy - 4:.1f}" text-anchor="end">{esc(ref_label)}</text>')
    for y in (years[0], 2031, 2033, years[-1]):
        if years[0] <= y <= years[-1]:
            anc = "end" if y == years[-1] else "start" if y == years[0] else "middle"
            body.append(f'<text class="tick" x="{xs(y):.1f}" y="{h - mb + 16}" text-anchor="{anc}">{y}</text>')
    return _svg(h, "".join(body), title, w)


def interval_dots(rows, threshold=None, xlabel="", lo_dom=None, hi_dom=1.0, pct_fmt=True) -> str:
    """rows: {label, p, lo, hi, cls}"""
    rh = 26
    ml, mr, mt, mb = 190, 24, 22, 34
    h = mt + mb + rh * len(rows)
    lo_dom = lo_dom if lo_dom is not None else min([r["lo"] for r in rows] + [threshold or 1]) - 0.02
    xs = Scale(lo_dom, hi_dom, ml, W - mr)
    body = []
    for t in nice_ticks(lo_dom, hi_dom, 5):
        if lo_dom <= t <= hi_dom:
            body.append(f'<line class="grid" x1="{xs(t):.1f}" x2="{xs(t):.1f}" y1="{mt}" y2="{h - mb}"/>'
                        f'<text class="tick" x="{xs(t):.1f}" y="{h - mb + 14}" text-anchor="middle">'
                        f'{t:.0%}</text>' if pct_fmt else '')
    if threshold is not None:
        body.append(f'<line class="ref-crit" x1="{xs(threshold):.1f}" x2="{xs(threshold):.1f}" y1="{mt - 4}" '
                    f'y2="{h - mb}"/><text class="lbl" x="{xs(threshold) - 4:.1f}" y="{mt + 2}" text-anchor="end">'
                    f'target {threshold:.0%}</text>')
    for i, r in enumerate(rows):
        y = mt + rh * i + rh / 2
        tip = f"{r['label']}: {r['p']:.2%} (95% MC interval {r['lo']:.2%} - {r['hi']:.2%})"
        body.append(f'<text class="rowlbl" x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(r["label"])}</text>'
                    f'<line class="whisker" x1="{xs(max(r["lo"], lo_dom)):.1f}" x2="{xs(r["hi"]):.1f}" '
                    f'y1="{y:.1f}" y2="{y:.1f}"/>'
                    f'<circle class="{r.get("cls", "m-eq")} ring" cx="{xs(max(r["p"], lo_dom)):.1f}" cy="{y:.1f}" r="5" '
                    f'data-tip="{esc(tip)}"><title>{esc(tip)}</title></circle>')
    body.append(f'<text class="tick" x="{(ml + W - mr) / 2:.0f}" y="{h - 4}" text-anchor="middle">{esc(xlabel)}</text>')
    return _svg(h, "".join(body), xlabel)


def hbars(rows, fmt, xlabel="", threshold=None, thr_label="") -> str:
    """rows: {label, value, cls}"""
    rh = 24
    ml, mr, mt, mb = 190, 64, 10, 30
    h = mt + mb + rh * len(rows)
    vmax = max([r["value"] for r in rows] + [threshold or 0]) * 1.1 or 1
    xs = Scale(0, vmax, ml, W - mr)
    body = []
    for t in nice_ticks(0, vmax, 4):
        body.append(f'<line class="grid" x1="{xs(t):.1f}" x2="{xs(t):.1f}" y1="{mt}" y2="{h - mb}"/>'
                    f'<text class="tick" x="{xs(t):.1f}" y="{h - mb + 14}" text-anchor="middle">{fmt(t)}</text>')
    for i, r in enumerate(rows):
        y = mt + rh * i + 4
        bh = min(16, rh - 6)
        body.append(f'<text class="rowlbl" x="{ml - 8}" y="{y + bh / 2 + 4:.1f}" text-anchor="end">{esc(r["label"])}</text>')
        if r["value"] > 0:
            body.append(_rounded_hbar(ml, xs(r["value"]), y, bh, r.get("cls", "m-eq"), f"{r['label']}: {fmt(r['value'])}"))
        body.append(f'<text class="lbl" x="{xs(r["value"]) + 5:.1f}" y="{y + bh / 2 + 4:.1f}">{fmt(r["value"])}</text>')
    if threshold is not None:
        body.append(f'<line class="ref-crit" x1="{xs(threshold):.1f}" x2="{xs(threshold):.1f}" y1="{mt}" y2="{h - mb}"/>'
                    f'<text class="lbl" x="{xs(threshold) + 4:.1f}" y="{mt + 8}">{esc(thr_label)}</text>')
    body.append(f'<text class="tick" x="{(ml + W - mr) / 2:.0f}" y="{h - 2}" text-anchor="middle">{esc(xlabel)}</text>')
    return _svg(h, "".join(body), xlabel)


def year_columns(years, values, fmt, title, cls="m-crit", w: int = 460) -> str:
    W = w
    h, ml, mr, mt, mb = 200, 48, 10, 24, 26
    xs = Scale(0, len(years), ml, W - mr)
    vmax = max(max(values), 1e-9) * 1.15
    ys = Scale(0, vmax, h - mb, mt)
    body = [f'<text class="ttl" x="{ml}" y="14">{esc(title)}</text>']
    for t in nice_ticks(0, vmax, 4):
        body.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{ys(t):.1f}" y2="{ys(t):.1f}"/>'
                    f'<text class="tick" x="{ml - 4}" y="{ys(t) + 4:.1f}" text-anchor="end">{fmt(t)}</text>')
    bw = min(24, (W - ml - mr) / len(years) * 0.6)
    for i, (y, v) in enumerate(zip(years, values)):
        cx = xs(i + 0.5)
        if v > 0:
            body.append(_rounded_col(cx - bw / 2, ys(0), ys(v), bw, cls, f"{y}: {fmt(v)}"))
        if i % 2 == 0 or i == len(years) - 1:
            body.append(f'<text class="tick" x="{cx:.1f}" y="{h - mb + 14}" text-anchor="middle">{y}</text>')
    body.append(f'<line class="axis" x1="{ml}" x2="{W - mr}" y1="{ys(0):.1f}" y2="{ys(0):.1f}"/>')
    return _svg(h, "".join(body), title, W)


def frontier(points, xlabel, ylabel) -> str:
    """points: {x, y, label, group: pass|other, leader: bool}"""
    h, ml, mr, mt, mb = 340, 70, 20, 16, 44
    xmax = max(p["x"] for p in points) * 1.12 or 1
    ymax = max(p["y"] for p in points) * 1.12 or 1
    xs = Scale(0, xmax, ml, W - mr)
    ys = Scale(0, ymax, h - mb, mt)
    body = []
    for t in nice_ticks(0, xmax, 5):
        body.append(f'<line class="grid" x1="{xs(t):.1f}" x2="{xs(t):.1f}" y1="{mt}" y2="{h - mb}"/>'
                    f'<text class="tick" x="{xs(t):.1f}" y="{h - mb + 14}" text-anchor="middle">{money(t)}</text>')
    for t in nice_ticks(0, ymax, 5):
        body.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{ys(t):.1f}" y2="{ys(t):.1f}"/>'
                    f'<text class="tick" x="{ml - 6}" y="{ys(t) + 4:.1f}" text-anchor="end">{money(t)}</text>')
    for p in sorted(points, key=lambda p: p["group"] == "pass"):
        cls = "m-eq ring" if p["group"] == "pass" else "m-muted ring"
        r = 7 if p.get("leader") else 5
        tip = f"{p['label']}: facility median {money(p['x'], False)}, retained median {money(p['y'], False)}"
        body.append(f'<circle class="{cls}" cx="{xs(p["x"]):.1f}" cy="{ys(p["y"]):.1f}" r="{r}" data-tip="{esc(tip)}">'
                    f'<title>{esc(tip)}</title></circle>')
        if p.get("label_on"):
            right = xs(p["x"]) > (ml + W - mr) * 0.6
            lx = xs(p["x"]) - 10 if right else xs(p["x"]) + 10
            body.append(f'<text class="lbl" x="{lx:.1f}" y="{ys(p["y"]) + 4:.1f}" '
                        f'text-anchor="{"end" if right else "start"}">{esc(p["label"])}</text>')
    body.append(f'<text class="tick" x="{(ml + W - mr) / 2:.0f}" y="{h - 6}" text-anchor="middle">{esc(xlabel)}</text>'
                f'<text class="tick" transform="translate(14,{(mt + h - mb) / 2:.0f}) rotate(-90)" '
                f'text-anchor="middle">{esc(ylabel)}</text>')
    return _svg(h, "".join(body), "facility vs retained flexibility")


def range_bars(rows, xlabel) -> str:
    """rows: {label, lo, hi, mid}"""
    rh = 30
    ml, mr, mt, mb = 200, 70, 10, 34
    h = mt + mb + rh * len(rows)
    vmax = max(r["hi"] for r in rows) * 1.1 or 1
    xs = Scale(0, vmax, ml, W - mr)
    body = []
    for t in nice_ticks(0, vmax, 5):
        body.append(f'<line class="grid" x1="{xs(t):.1f}" x2="{xs(t):.1f}" y1="{mt}" y2="{h - mb}"/>'
                    f'<text class="tick" x="{xs(t):.1f}" y="{h - mb + 14}" text-anchor="middle">{money(t)}</text>')
    for i, r in enumerate(rows):
        y = mt + rh * i + 8
        tip = f"{r['label']}: {money(r['lo'], False)} - {money(r['hi'], False)} (median {money(r['mid'], False)})"
        x0, x1 = xs(r["lo"]), max(xs(r["hi"]), xs(r["lo"]) + 2)
        body.append(f'<text class="rowlbl" x="{ml - 8}" y="{y + 11:.1f}" text-anchor="end">{esc(r["label"])}</text>'
                    f'<rect class="m-cash" x="{x0:.1f}" y="{y:.1f}" width="{x1 - x0:.1f}" height="14" rx="4" '
                    f'data-tip="{esc(tip)}"><title>{esc(tip)}</title></rect>'
                    f'<circle class="m-eq ring" cx="{xs(r["mid"]):.1f}" cy="{y + 7:.1f}" r="5"/>'
                    f'<text class="lbl" x="{x1 + 6:.1f}" y="{y + 11:.1f}">{money(r["lo"])}-{money(r["hi"])}</text>')
    body.append(f'<text class="tick" x="{(ml + W - mr) / 2:.0f}" y="{h - 4}" text-anchor="middle">{esc(xlabel)}</text>')
    return _svg(h, "".join(body), xlabel)


def histogram(edges, counts, n_zero, n, title) -> str:
    h, ml, mr, mt, mb = 200, 52, 10, 24, 30
    xs = Scale(edges[0], edges[-1], ml, W - mr)
    share = [c / n for c in counts]
    vmax = max(share) * 1.15 or 1
    ys = Scale(0, vmax, h - mb, mt)
    body = [f'<text class="ttl" x="{ml}" y="14">{esc(title)}</text>']
    for t in nice_ticks(0, vmax, 4):
        body.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{ys(t):.1f}" y2="{ys(t):.1f}"/>'
                    f'<text class="tick" x="{ml - 4}" y="{ys(t) + 4:.1f}" text-anchor="end">{t:.0%}</text>')
    for i, s in enumerate(share):
        x0, x1 = xs(edges[i]) + 1, xs(edges[i + 1]) - 1
        if s > 0:
            tip = f"{money(edges[i], False)} - {money(edges[i + 1], False)}: {s:.1%} of paths"
            body.append(_rounded_col(x0, ys(0), ys(s), max(1.0, x1 - x0), "m-eq", tip, r=2))
    for t in nice_ticks(edges[0], edges[-1], 5):
        if edges[0] <= t <= edges[-1]:
            body.append(f'<text class="tick" x="{xs(t):.1f}" y="{h - mb + 14}" text-anchor="middle">{money(t)}</text>')
    body.append(f'<text class="lbl" x="{W - mr}" y="{mt + 4}" text-anchor="end">'
                f'mass at $0: {n_zero / n:.1%} of paths (first bin includes it; top bin includes values above P99)</text>')
    return _svg(h, "".join(body), title)
