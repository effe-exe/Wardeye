# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""Turn spike CSVs into report material: Markdown tables and an SVG accuracy-vs-height chart.

Reports carry numbers only, never images of cards or streams (decision D-006).

    python -m rifteye_ml.report --csv ../docs/reports/m0-synthetic.csv \
        --svg ../docs/reports/m0-synthetic.svg --bitrate 4000 --rotation search
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

PALETTE = ["#6153CC", "#E0795A", "#2E9E8F", "#C9A227", "#8A8F98", "#D14D8B"]
LABELS = {
    "colorgrid16": "colour grid 16×16",
    "dhash16": "dHash 16",
    "timm:vit_small_patch14_dinov2.lvd142m@224": "DINOv2 ViT-S/14, frozen",
    "timm:vit_pe_core_small_patch16_384.fb@224": "PE Core S16, frozen",
}


def label(encoder: str) -> str:
    """A short display name for an encoder tag."""
    if encoder in LABELS:
        return LABELS[encoder]
    return encoder.replace("timm:", "")[:28]


def load(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _height(r: dict) -> int:
    return int(str(r["card_h"]).split("-")[0])


def _keep(r: dict, bitrate: str, realism: str | None, queries: str | None = None) -> bool:
    return (str(r["bitrate_kbps"]) == str(bitrate) and (realism is None or r.get("realism", "") == realism)
            and (queries is None or r.get("queries", "") == queries))


def _series(rows: Sequence[dict], bitrate: str, rotation: str, metric: str,
            realism: str | None = None) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        if _keep(r, bitrate, realism) and r["rotation"] == rotation:
            out[r["encoder"]].append((_height(r), float(r[metric])))
    return {k: sorted(v) for k, v in out.items()}


def markdown_table(rows: Sequence[dict], bitrate: str, metric: str = "top1_card", realism: str | None = None) -> str:
    """One row per encoder and rotation mode, one column per card height."""
    sel = [r for r in rows if _keep(r, bitrate, realism)]
    heights = sorted({_height(r) for r in sel})
    cells: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in sel:
        cells[(r["encoder"], r["rotation"])][_height(r)] = float(r[metric])
    head = "| Encoder | Rotation | " + " | ".join(f"{h} px" for h in heights) + " |"
    rule = "|---|---|" + "---:|" * len(heights)
    body = [f"| `{enc}` | {rot} | " + " | ".join(f"{v[h]:.3f}" if h in v else "" for h in heights) + " |"
            for (enc, rot), v in sorted(cells.items())]
    return "\n".join([head, rule, *body])


def svg_chart(rows: Sequence[dict], bitrate: str, rotation: str = "search", metric: str = "top1_card",
              title: str = "", width: int = 720, height: int = 420, realism: str | None = None) -> str:
    series = _series(rows, bitrate, rotation, metric, realism)
    xs = sorted({x for pts in series.values() for x, _ in pts}) or [0, 1]
    left, right, top, bottom = 64, 210, 48, 56
    pw, ph = width - left - right, height - top - bottom
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1

    def px(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * pw

    def py(y: float) -> float:
        return top + (1 - y) * ph

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif" font-size="12">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    if title:
        parts.append(f'<text x="{left}" y="28" font-size="15" font-weight="600" fill="#1f2328">{escape(title)}</text>')
    for i in range(0, 11, 2):
        y = py(i / 10)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#57606a">{i * 10}%</text>')
    for x in xs:
        parts.append(f'<text x="{px(x):.1f}" y="{top + ph + 20}" text-anchor="middle" fill="#57606a">{x}</text>')
    frames = {r["frame"] for r in rows if r.get("frame")}
    xlabel = f"card height in a {frames.pop()} frame (px)" if len(frames) == 1 else "card height (px)"
    parts.append(f'<text x="{left + pw / 2:.1f}" y="{height - 14}" text-anchor="middle" fill="#1f2328">'
                 f'{escape(xlabel)}</text>')
    for i, (name, pts) in enumerate(sorted(series.items())):
        color = PALETTE[i % len(PALETTE)]
        path = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        parts.extend(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{color}"/>' for x, y in pts)
        ly = top + 8 + i * 22
        parts.append(f'<line x1="{left + pw + 16}" y1="{ly}" x2="{left + pw + 36}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{left + pw + 42}" y="{ly + 4}" fill="#1f2328">{escape(label(name))}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.report", description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--svg", help="write an accuracy-vs-height chart here")
    ap.add_argument("--bitrate", default="4000", help="bitrate (kbps) for the chart")
    ap.add_argument("--rotation", default="search", choices=["search", "oracle"])
    ap.add_argument("--metric", default="top1_card")
    ap.add_argument("--realism", help="only rows at this realism level (codec, camera)")
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)
    rows = load(a.csv)
    levels = [a.realism] if a.realism else sorted({r.get("realism", "") for r in rows})
    for level in levels:
        for br in sorted({r["bitrate_kbps"] for r in rows}, key=lambda b: int(b or 0)):
            print(f"\n{a.metric}, realism {level or 'n/a'}, {br} kbps\n")
            print(markdown_table(rows, br, a.metric, level))
    if a.svg:
        Path(a.svg).write_text(svg_chart(rows, a.bitrate, a.rotation, a.metric, a.title, realism=a.realism),
                               encoding="utf-8")
        print(f"\nchart -> {a.svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
