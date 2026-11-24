"""PISN abundance signature and how much of it survives dilution.

Panel A: the pure PISN pattern in [X/Fe] (Fe excess collapses every ratio).
Panel B: the PISN minus diluted-Pop-II contrast per yield table, in dex.

The point of the figure: Zn and Co are the loudest markers, but dilution
removes most of the signature, which is why a single element is not enough.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ("Na", "Mg", "Si", "Ca", "Ti", "Cr", "Mn", "Co", "Ni", "Zn")

# short labels for the axes so the deck figure stays readable
LABELS = {
    "Limongi18_clamp": "Limongi18\nclamp",
    "WW95_native": "WW95\nnative",
    "Nomoto13_blend": "Nomoto13\nblend",
    "Nomoto13_clamp": "Nomoto13\nclamp",
    "Nomoto13_CK20_blend": "CK20\nblend",
    "Nomoto13_CK20_clamp": "CK20\nclamp",
}


def run(
    source: str | Path = "src/potluck/tests/v20/pisn_contrast.json",
    outdir: str | Path = "src/potluck/tests/v20/processed/pisn_contrast",
) -> dict[str, Any]:
    source = Path(source)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = json.loads(source.read_text())
    pisn = data["pisn"]
    policies = data["policies"]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(14.5, 5.6), gridspec_kw={"width_ratios": [1.0, 1.35]}
    )

    # ---- panel A: the pure PISN pattern -------------------------------
    values = np.array([pisn[e] for e in ORDER])
    positions = np.arange(len(ORDER))
    colors = ["tab:red" if v < -1.0 else "tab:blue" for v in values]
    left.bar(positions, values, color=colors, alpha=0.85)
    left.axhline(0.0, color="0.4", lw=0.9)
    left.set_xticks(positions, labels=ORDER, rotation=45, ha="right")
    left.set_ylabel(r"$[\mathrm{X/Fe}]$  (pure PISN)")
    left.set_title("A  What a PISN looks like when undiluted", loc="left", fontsize=11)
    left.grid(axis="y", alpha=0.25)
    for position, value in zip(positions, values):
        left.text(
            position,
            value + (-0.35 if value < 0 else 0.20),
            f"{value:+.1f}",
            ha="center",
            va="top" if value < 0 else "bottom",
            fontsize=7.5,
        )
    left.set_ylim(min(values) * 1.18, max(values) * 1.9 + 0.4)
    left.annotate(
        "huge Fe excess from $^{56}$Ni decay\ndrives every ratio down",
        xy=(ORDER.index("Zn"), pisn["Zn"]),
        xytext=(4.4, pisn["Zn"] * 0.55),
        fontsize=8,
        color="0.25",
        arrowprops={"arrowstyle": "->", "color": "0.5", "lw": 0.9},
    )

    # ---- panel B: what dilution leaves behind ------------------------
    # The stored contrast is (pure PISN - diluted model), so the model's
    # own [X/Fe] is recovered by subtracting it back off.
    names = [
        p for p in ("Limongi18_clamp", "WW95_native", "Nomoto13_blend", "Nomoto13_clamp")
        if p in policies
    ]
    width = 0.8 / max(len(names), 1)
    for offset, policy in enumerate(names):
        row = np.array(
            [pisn[e] - policies[policy].get(e, np.nan) for e in ORDER]
        )
        right.bar(
            positions + (offset - (len(names) - 1) / 2) * width,
            row,
            width=width,
            label=LABELS.get(policy, policy),
            alpha=0.85,
        )
    right.plot(
        positions,
        values,
        color="black",
        lw=1.4,
        ls="--",
        marker="o",
        ms=3.5,
        label="pure PISN",
        zorder=5,
    )
    right.axhline(0.0, color="0.4", lw=0.9)
    right.set_xticks(positions, labels=ORDER, rotation=45, ha="right")
    right.set_ylabel(r"$[\mathrm{X/Fe}]$")
    right.set_title(
        "B  What survives after dilution  (dashed = undiluted PISN)",
        loc="left",
        fontsize=11,
    )
    right.legend(fontsize=7.5, frameon=False, ncol=3)
    right.grid(axis="y", alpha=0.25)

    # highlight the two loudest markers
    for element in ("Co", "Zn"):
        right.axvspan(
            ORDER.index(element) - 0.42, ORDER.index(element) + 0.42,
            color="tab:orange", alpha=0.10, zorder=0,
        )

    fig.suptitle(
        "The PISN signature and its dilution  "
        r"($Z_* = 10^{-5}$, $t = 10$ Myr; marker elements shaded)",
        fontsize=12,
    )
    fig.tight_layout()
    path = outdir / "pisn_contrast.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "figure": str(path),
        "pure_pisn": {e: float(pisn[e]) for e in ORDER},
        "policy_contrast": {
            p: {e: float(policies[p].get(e, float("nan"))) for e in ORDER} for p in names
        },
        "strongest_markers": sorted(
            ((e, float(pisn[e])) for e in ORDER), key=lambda item: item[1]
        )[:3],
    }
    with (outdir / "pisn_contrast_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote {path}")
    for element, value in summary["strongest_markers"]:
        print(f"  strongest: {element} {value:+.2f} dex")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="src/potluck/tests/v20/pisn_contrast.json")
    parser.add_argument(
        "--outdir", default="src/potluck/tests/v20/processed/pisn_contrast"
    )
    args = parser.parse_args(argv)
    run(args.source, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
