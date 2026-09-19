"""Figure styling and small annotation helpers shared by all plots.

The helpers here replace the boiler-plate that used to be copied into every figure
cell: millimetre figure sizes, journal rcParams, colour palette, significance star
lookup, significance brackets and consistent ``savefig`` settings.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "MM_PER_INCH",
    "mm_to_inches",
    "apply_style",
    "PALETTE",
    "condition_colors",
    "despine",
    "sig_stars",
    "significance_bar",
    "annotate_significance",
    "save_figure",
]

MM_PER_INCH = 25.4

#: Default colour scheme (condition / band names used across the project figures).
PALETTE = {
    "sem": "#000080",
    "semantic": "#000080",
    "str": "#B22222",
    "structural": "#B22222",
    "ambg": "#FF8C00",
    "reg": "#4D4D4D",
    "om": "#32037D",
    "rev": "#7C1A97",
    "theta": "#EA8379",
    "alpha": "#7DAEE0",
    "beta": "#B395BD",
    "gamma": "#F3A332",
    "blue": "#074C9B",
    "grey": "#808080",
}


def mm_to_inches(width_mm: float, height_mm: float | None = None):
    """Convert millimetre figure sizes to inches.

    ``mm_to_inches(60, 50)`` -> ``(2.36, 1.97)``; with ``height_mm=None`` a single
    number is returned.
    """
    if height_mm is None:
        return width_mm / MM_PER_INCH
    return width_mm / MM_PER_INCH, height_mm / MM_PER_INCH


def apply_style(*, font: str = "Arial", mathtext: str = "cm", svg_fonttype: str = "none",
                fontsize: float = 8.0, linewidth: float = 1.0, dpi: int = 600,
                transparent: bool = True) -> None:
    """Set matplotlib rcParams for publication figures (idempotent)."""
    mpl.rcParams.update({
        "font.family": font,
        "mathtext.fontset": mathtext,
        "svg.fonttype": svg_fonttype,
        "font.size": fontsize,
        "axes.labelsize": fontsize + 2,
        "axes.titlesize": fontsize + 2,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "axes.linewidth": linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": linewidth,
        "ytick.major.width": linewidth,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": linewidth + 1,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": dpi,
        "savefig.transparent": transparent,
        "savefig.bbox": "tight",
    })


def condition_colors(conditions, *, palette: dict | None = None, cmap: str | None = None):
    """Map condition names to colours.

    Names present in :data:`PALETTE` (or a user ``palette``) keep their colour; the
    remaining ones are spread over ``cmap``.
    """
    palette = {**PALETTE, **(palette or {})}
    colors = {}
    missing = [c for c in conditions if str(c).lower() not in palette]
    if missing and cmap is not None:
        cm = plt.get_cmap(cmap)
        for i, name in enumerate(missing):
            colors[name] = cm(i / max(len(missing) - 1, 1))
    for name in conditions:
        if name not in colors:
            colors[name] = palette.get(str(name).lower(), "black")
    return colors


def despine(ax, *, left: bool = False, bottom: bool = False) -> None:
    """Hide the top / right spines (and optionally left / bottom)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


def sig_stars(pvalue, *, thresholds=(0.001, 0.01, 0.05), symbols=("***", "**", "*"),
              non_significant: str = "n.s.") -> str:
    """Significance star string for a p-value."""
    if pvalue is None or (isinstance(pvalue, float) and np.isnan(pvalue)):
        return non_significant
    for threshold, symbol in zip(thresholds, symbols):
        if pvalue < threshold:
            return symbol
    return non_significant


def significance_bar(ax, x1: float, x2: float, y: float, *, height=None, text=None,
                     color: str = "k", linewidth: float = 1.2, fontsize: float = 6,
                     text_offset: float = 0.2) -> None:
    """Draw a significance bracket between ``x1`` and ``x2`` at height ``y``."""
    span = abs(x2 - x1)
    if height is None:
        ymin, ymax = ax.get_ylim()
        height = 0.02 * (ymax - ymin)
    height = max(height, 1e-12)
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=linewidth, c=color,
            clip_on=False)
    if text:
        ax.text((x1 + x2) / 2, y + height * (1 + text_offset), text, ha="center",
                va="bottom", fontsize=fontsize, color=color)


def annotate_significance(ax, pairs, pvalues, *, values=None, y_start=None, step=None,
                          color: str = "k", linewidth: float = 1.2,
                          fontsize: float = 6) -> None:
    """Stack significance brackets above a set of distributions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    pairs : sequence of (int, int)
        Positions on the x axis (categorical positions 0, 1, ...).
    pvalues : sequence of float
    values : sequence of array_like | None
        Data behind each x position, used to place the brackets automatically.
    y_start, step : float | None
        Explicit bottom bracket height and vertical spacing.
    """
    if values is not None:
        all_values = np.concatenate([np.ravel(v) for v in values])
        y_min, y_max = np.nanmin(all_values), np.nanmax(all_values)
        span = (y_max - y_min) or 1.0
        y_start = y_start if y_start is not None else y_max + 0.12 * span
        step = step if step is not None else 0.12 * span
    else:
        ymin, ymax = ax.get_ylim()
        span = (ymax - ymin) or 1.0
        y_start = y_start if y_start is not None else ymin + 0.85 * span
        step = step if step is not None else 0.08 * span

    for i, ((x1, x2), pvalue) in enumerate(zip(pairs, pvalues)):
        significance_bar(ax, x1, x2, y_start + i * step, height=0.02 * span,
                         text=sig_stars(pvalue), color=color, linewidth=linewidth,
                         fontsize=fontsize)


def save_figure(fig, path, *, dpi: int = 600, transparent: bool = True,
                also_png: bool = False, close: bool = False, **kwargs) -> None:
    """Save a figure with the project defaults; optionally also as PNG."""
    fig.savefig(path, dpi=dpi, transparent=transparent, bbox_inches="tight", **kwargs)
    if also_png:
        fig.savefig(str(path).rsplit(".", 1)[0] + ".png", dpi=dpi,
                    transparent=transparent, bbox_inches="tight", **kwargs)
    if close:
        plt.close(fig)
