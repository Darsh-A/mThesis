"""Element coverage and abundance-error tables for the APOGEE DR17 catalogue.

Two questions, one pass over the FITS file:

* how often is each element recorded, including elements never recorded at all;
* how large is the quoted uncertainty on each element.

The whole allStarLite table is used, not the potluck quality cut, so the counts
describe the catalogue rather than the analysis sample.

Outputs (``--outdir``):
    apogee_element_counts.png      bar chart, one bar per element, zeros visible
    apogee_element_counts.csv      counts, fractions, flag breakdown
    apogee_element_errors.png      rendered error table
    apogee_element_errors.csv      percentile table
    apogee_element_error_dist.png  error distributions
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
from astropy.io import fits

# Every element column in allStarLite DR17, in the order ASPCAP writes them.
ELEMENT_COLUMNS: tuple[str, ...] = (
    "C_FE",
    "CI_FE",
    "N_FE",
    "O_FE",
    "NA_FE",
    "MG_FE",
    "AL_FE",
    "SI_FE",
    "P_FE",
    "S_FE",
    "K_FE",
    "CA_FE",
    "TI_FE",
    "TIII_FE",
    "V_FE",
    "CR_FE",
    "MN_FE",
    "CO_FE",
    "NI_FE",
    "CU_FE",
    "CE_FE",
)

# Display labels that keep the ion stage visible instead of merging it away.
LABELS: dict[str, str] = {
    "CI_FE": "C I",
    "TIII_FE": "Ti II",
}

PERCENTILES = (5.0, 16.0, 50.0, 84.0, 95.0)


def _label(column: str) -> str:
    if column in LABELS:
        return LABELS[column]
    return column[: -len("_FE")].capitalize()


def read_columns(path: Path) -> dict[str, np.ndarray]:
    """Read only the element and error columns, with the per-element flags."""
    wanted = list(ELEMENT_COLUMNS)
    wanted += [f"{column}_ERR" for column in ELEMENT_COLUMNS]
    wanted += [f"{column}_FLAG" for column in ELEMENT_COLUMNS]
    out: dict[str, np.ndarray] = {}
    with fits.open(path, memmap=True) as handle:
        data = handle[1].data
        available = set(data.names)
        for name in wanted:
            if name not in available:
                out[name] = np.full(len(data), np.nan, dtype=np.float32)
                continue
            values = np.array(data[name])
            if values.dtype.kind == "f":
                values = values.astype(np.float32)
            out[name] = values
    return out


def summarise(columns: dict[str, np.ndarray]) -> dict[str, Any]:
    """Per-element counts, flag breakdown, and error percentiles."""
    rows: dict[str, Any] = {}
    for column in ELEMENT_COLUMNS:
        value = np.asarray(columns[column], dtype=float)
        error = np.asarray(columns[f"{column}_ERR"], dtype=float)
        flag = np.asarray(columns[f"{column}_FLAG"], dtype=float)

        finite = np.isfinite(value)
        positive_error = finite & np.isfinite(error) & (error > 0.0)
        flagged_zero = np.isfinite(flag) & (flag.astype(np.int64) & 1 == 0)

        errors = error[positive_error]
        if errors.size:
            low, p16, median, p84, high = np.percentile(errors, PERCENTILES)
        else:
            low = p16 = median = p84 = high = float("nan")

        rows[column] = {
            "label": _label(column),
            "recorded": int(finite.sum()),
            "fraction": float(finite.mean()),
            "flag_zero": int(flagged_zero.sum()),
            "n_error": int(positive_error.sum()),
            "error_p5": float(low),
            "error_p16": float(p16),
            "error_median": float(median),
            "error_p84": float(p84),
            "error_p95": float(high),
        }
    return {"n_rows": int(len(columns[ELEMENT_COLUMNS[0]])), "elements": rows}


def plot_counts(summary: dict[str, Any], path: Path) -> None:
    """One bar per element on a linear axis; zero-count elements stay visible."""
    elements = summary["elements"]
    total = summary["n_rows"]

    counts = np.array([elements[column]["recorded"] for column in ELEMENT_COLUMNS])
    order = np.argsort(-counts)
    names_sorted = [_label(ELEMENT_COLUMNS[i]) for i in order]
    counts_sorted = counts[order]
    fractions = 100.0 * counts_sorted / total

    fig, axis = plt.subplots(figsize=(13.0, 6.0))
    colours = ["tab:red" if value == 0 else "tab:blue" for value in counts_sorted]
    axis.bar(np.arange(len(counts_sorted)), counts_sorted, color=colours)
    axis.set_xticks(np.arange(len(counts_sorted)))
    axis.set_xticklabels(names_sorted, rotation=45, ha="right", fontsize=9)
    axis.set_ylabel("stars with a recorded value")
    axis.set_title(f"APOGEE DR17 element coverage, {total:,} stars")
    axis.set_ylim(0, total * 1.12)
    axis.grid(axis="y", alpha=0.25)

    for position, (name, value, fraction) in enumerate(
        zip(names_sorted, counts_sorted, fractions)
    ):
        if value == 0:
            axis.text(
                position, total * 0.02, "never\nrecorded",
                ha="center", va="bottom", fontsize=8, color="tab:red",
            )
        else:
            axis.text(
                position, value + total * 0.015, f"{fraction:.1f}%",
                ha="center", va="bottom", fontsize=7.5,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_errors(summary: dict[str, Any], path: Path, table_path: Path) -> None:
    """Rendered table of the error percentiles."""
    elements = summary["elements"]
    columns = [c for c in ELEMENT_COLUMNS]
    # Largest uncertainty first; elements with no recorded value last.
    columns.sort(
        key=lambda c: (
            -elements[c]["error_median"]
            if np.isfinite(elements[c]["error_median"])
            else np.inf
        )
    )

    header = ["element", "n", "p16", "median", "p84", "p95"]
    cells = []
    for column in columns:
        row = elements[column]
        if not np.isfinite(row["error_median"]):
            cells.append([row["label"], "0", "-", "-", "-", "-"])
        else:
            cells.append([
                row["label"],
                f"{row['n_error']:,}",
                f"{row['error_p16']:.3f}",
                f"{row['error_median']:.3f}",
                f"{row['error_p84']:.3f}",
                f"{row['error_p95']:.3f}",
            ])

    height = 0.32 * len(cells) + 1.5
    fig, axis = plt.subplots(figsize=(7.5, height))
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=header, loc="center",
                       cellLoc="right", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    for (row_index, _), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(weight="bold")
    axis.set_title("Quoted [X/Fe] uncertainty (dex)", fontsize=12, pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    table_path.write_text(
        "element,n,error_p5,error_p16,error_median,error_p84,error_p95\n"
        + "\n".join(
            f"{elements[c]['label']},{elements[c]['n_error']},"
            f"{elements[c]['error_p5']:.4f},{elements[c]['error_p16']:.4f},"
            f"{elements[c]['error_median']:.4f},{elements[c]['error_p84']:.4f},"
            f"{elements[c]['error_p95']:.4f}"
            for c in columns
            if np.isfinite(elements[c]["error_median"])
        )
        + "\n"
    )


def plot_error_distributions(
    columns: dict[str, np.ndarray], summary: dict[str, Any], path: Path
) -> None:
    """Violin-style spread of the errors, so the table has a visual partner."""
    present = [
        column
        for column in ELEMENT_COLUMNS
        if summary["elements"][column]["n_error"] > 0
    ]
    present.sort(key=lambda c: -summary["elements"][c]["error_median"])
    data = []
    labels = []
    for column in present:
        error = np.asarray(columns[f"{column}_ERR"], dtype=float)
        good = np.isfinite(error) & (error > 0.0)
        data.append(np.clip(error[good], 0.0, 1.2))
        labels.append(_label(column))

    fig, axis = plt.subplots(figsize=(15.0, 5.5))
    parts = axis.violinplot(data, showextrema=False, widths=0.85)
    for body in parts["bodies"]:
        body.set_facecolor("tab:blue")
        body.set_alpha(0.45)
    for position, values in enumerate(data, start=1):
        median = float(np.median(values))
        axis.plot([position], [median], "o", color="black", ms=3, zorder=3)
        axis.hlines(median, position - 0.32, position + 0.32, color="black", lw=1.0)
    axis.set_xticks(np.arange(1, len(labels) + 1))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axis.set_ylabel(r"$\sigma$([X/Fe]) [dex]")
    axis.set_ylim(0, 1.0)
    axis.set_title("Error distribution per element (clipped at 1.2 dex)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(
    fits_path: str | Path,
    outdir: str | Path = "src/potluck/tests/v20/processed/apogee",
) -> dict[str, Any]:
    fits_path = Path(fits_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    columns = read_columns(fits_path)
    summary = summarise(columns)

    plot_counts(summary, outdir / "apogee_element_counts.png")
    plot_errors(
        summary,
        outdir / "apogee_element_errors.png",
        outdir / "apogee_element_errors.csv",
    )
    plot_error_distributions(
        columns, summary, outdir / "apogee_element_error_dist.png"
    )

    lines = ["element,recorded,fraction,flag_zero,n_error"]
    for column in ELEMENT_COLUMNS:
        row = summary["elements"][column]
        lines.append(
            f"{row['label']},{row['recorded']},{row['fraction']:.4f},"
            f"{row['flag_zero']},{row['n_error']}"
        )
    (outdir / "apogee_element_counts.csv").write_text("\n".join(lines) + "\n")
    (outdir / "apogee_element_coverage.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fits",
        nargs="?",
        default="data/raw/allStarLite-dr17-synspec_rev1.fits",
    )
    parser.add_argument(
        "--outdir", default="src/potluck/tests/v20/processed/apogee"
    )
    args = parser.parse_args(argv)
    summary = run(args.fits, args.outdir)
    total = summary["n_rows"]
    print(f"{total:,} stars")
    for column in ELEMENT_COLUMNS:
        row = summary["elements"][column]
        print(
            f"  {row['label']:6s} recorded {row['recorded']:7,d} "
            f"({100 * row['fraction']:5.1f}%)  "
            f"median error {row['error_median']:.3f} dex"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
