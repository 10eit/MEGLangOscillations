"""Group distribution plots: box / violin with subject points, KDEs and fits.

The functions take plain arrays (or a ``{label: array}`` mapping) and draw into an
existing axes when ``ax`` is given, so several conditions can be overlaid.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .style import annotate_significance, condition_colors, despine, mm_to_inches

__all__ = [
    "plot_group_distribution",
    "annotate_pairs",
    "plot_kde",
    "plot_scatter_fit",
]


def _as_mapping(values):
    if isinstance(values, dict):
        return list(values.keys()), [np.asarray(v, dtype=float) for v in values.values()]
    values = [np.asarray(v, dtype=float) for v in values]
    return list(range(len(values))), values


def _orientation_kwargs(horizontal: bool) -> dict:
    """Orientation keyword across matplotlib versions (``vert`` -> ``orientation``)."""
    import inspect

    if "orientation" in inspect.signature(plt.Axes.boxplot).parameters:
        return {"orientation": "horizontal" if horizontal else "vertical"}
    return {"vert": not horizontal}


def _jitter(n: int, width: float, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)


def plot_group_distribution(values, *, labels=None, ax=None, kind: str = "box",
                            colors=None, points: str | None = "strip", paired: bool = False,
                            width: float = 0.5, jitter: float = 0.08, point_size: float = 3,
                            point_alpha: float = 0.7, point_color: str = "black",
                            positions=None, showfliers: bool = False, orient: str = "v",
                            xlabel=None, ylabel=None, ylim=None, linewidth: float = 1.0,
                            seed=0, figsize_mm=(60, 50), palette=None, cmap=None,
                            point_edge: str | None = "white", **kwargs):
    """Box or violin plot with individual observations overlaid.

    Parameters
    ----------
    values : dict | sequence of array_like
        One array per condition.
    labels : sequence | None
        Condition names (defaults to the keys of ``values``).
    kind : {'box', 'violin', 'point'}
        ``'point'`` draws only the observations.
    colors : dict | sequence | None
        Per-condition colours; names are resolved with
        :func:`visualization.style.condition_colors`.
    points : {'strip', 'swarm', None}
        How to overlay single observations (``paired`` implies connecting them).
    paired : bool
        Connect the i-th observation of neighbouring conditions (repeated measures).
    positions : sequence | None
        Explicit x positions (defaults to ``0..n-1``).
    orient : {'v', 'h'}
        Vertical (conditions on x) or horizontal (conditions on y).
    seed : int
        Jitter seed.
    figsize_mm : (float, float)
        Used only when ``ax`` is None.

    Returns
    -------
    fig, ax
    """
    names, arrays = _as_mapping(values)
    if labels is not None:
        names = list(labels)
    if positions is None:
        positions = np.arange(len(arrays))
    else:
        positions = np.asarray(positions, dtype=float)

    if colors is None:
        color_map = condition_colors([str(n) for n in names], palette=palette, cmap=cmap)
    elif isinstance(colors, dict):
        color_map = colors
    else:
        color_map = dict(zip(names, colors))
    face_colors = [color_map.get(name, color_map.get(str(name), "grey")) for name in names]

    if ax is None:
        fig, ax = plt.subplots(figsize=mm_to_inches(*figsize_mm))
    else:
        fig = ax.figure

    horizontal = orient == "h"
    plot_data = [a[~np.isnan(a)] for a in arrays]
    if kind in ("box", "violin"):
        common = dict(positions=positions, widths=width, **_orientation_kwargs(horizontal))
        if kind == "box":
            artists = ax.boxplot(plot_data, showfliers=showfliers, patch_artist=True,
                                 **common)
            bodies = artists.get("boxes")
        else:
            artists = ax.violinplot(plot_data, showextrema=False, showmedians=True,
                                    **common)
            bodies = artists.get("bodies")
        for body, face in zip(bodies, face_colors):
            body.set_facecolor(face)
            body.set_edgecolor("black")
            body.set_linewidth(linewidth)
            if kind == "violin":
                body.set_alpha(0.8)
        for key in ("whiskers", "caps", "medians", "cbars", "cmins", "cmaxes"):
            for item in artists.get(key, []) or []:
                item.set_color("black")
                item.set_linewidth(linewidth)

    if points:
        for position, array, face in zip(positions, arrays, face_colors):
            array = np.asarray(array, dtype=float)
            mask = ~np.isnan(array)
            offsets = _jitter(int(mask.sum()), jitter, seed)
            if horizontal:
                ax.scatter(array[mask], position + offsets, s=point_size,
                           color=point_color, alpha=point_alpha, edgecolors=point_edge,
                           linewidths=0.5, zorder=3)
            else:
                ax.scatter(position + offsets, array[mask], s=point_size,
                           color=point_color, alpha=point_alpha, edgecolors=point_edge,
                           linewidths=0.5, zorder=3)

    if paired and len(arrays) > 1:
        for left, right in zip(range(len(arrays) - 1), range(1, len(arrays))):
            a, b = arrays[left], arrays[right]
            for i in range(min(len(a), len(b))):
                if np.isnan(a[i]) or np.isnan(b[i]):
                    continue
                if horizontal:
                    ax.plot([a[i], b[i]], [positions[left], positions[right]],
                            color="grey", linewidth=0.5, alpha=0.5, zorder=2)
                else:
                    ax.plot([positions[left], positions[right]], [a[i], b[i]],
                            color="grey", linewidth=0.5, alpha=0.5, zorder=2)

    if horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels([str(n) for n in names])
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels([str(n) for n in names])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    despine(ax)
    return fig, ax


def annotate_pairs(ax, pairs, pvalues, *, values=None, y_start=None, step=None,
                   **kwargs):
    """Significance brackets above group distributions (see :mod:`visualization.style`)."""
    annotate_significance(ax, pairs, pvalues, values=values, y_start=y_start, step=step,
                          **kwargs)
    return ax


def plot_kde(samples, *, ax=None, colors=None, fill: bool = True,
            common_norm: bool = False, bandwidth=None, grid: int = 1000,
            reference: float | None = None, xlabel=None, ylabel: str = "Density",
            legend: bool = True, xlim=None, linewidth: float = 2,
            figsize_mm=(60, 50), palette=None, cmap=None, **kwargs):
    """Kernel density estimate for one or several sample sets.

    Parameters
    ----------
    samples : dict | sequence of array_like
    common_norm : bool
        Normalise each density by the number of samples of its group (``True``) or
        scale every density to the same total area (``False``, default).
    reference : float | None
        Vertical reference line (e.g. 0).
    bandwidth : float | None
        Gaussian kernel bandwidth (default: Scott's rule).
    """
    names, arrays = _as_mapping(samples)
    if colors is None:
        color_map = condition_colors([str(n) for n in names], palette=palette, cmap=cmap)
    elif isinstance(colors, dict):
        color_map = colors
    else:
        color_map = dict(zip(names, colors))

    if ax is None:
        fig, ax = plt.subplots(figsize=mm_to_inches(*figsize_mm))
    else:
        fig = ax.figure

    for name, array in zip(names, arrays):
        array = np.asarray(array, dtype=float)
        array = array[~np.isnan(array)]
        if array.size < 2:
            continue
        density = gaussian_kde(array, bw_method=bandwidth)
        x = np.linspace(array.min(), array.max(), grid)
        y = density(x)
        if common_norm:
            y = y / array.size
        color = color_map.get(name, color_map.get(str(name), "black"))
        ax.plot(x, y, color=color, linewidth=linewidth, label=str(name), **kwargs)
        if fill:
            ax.fill_between(x, y, color=color, alpha=0.2, linewidth=0)

    if reference is not None:
        ax.axvline(reference, color="k", linestyle="--", linewidth=1)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend and len(names) > 1:
        ax.legend()
    despine(ax)
    return fig, ax


def plot_scatter_fit(x, y, *, ax=None, fit: str | None = "linear", color: str = "grey",
                     fit_color: str = "darkred", point_size: float = 30,
                     point_alpha: float = 0.7, n_grid: int = 200, xlabel=None,
                     ylabel=None, legend: bool = False, figsize_mm=(60, 45), **kwargs):
    """Scatter plot with an optional linear or logistic fit.

    Parameters
    ----------
    x, y : array_like
        ``y`` may be binary when ``fit='logistic'``.
    fit : {'linear', 'logistic', None}
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=mm_to_inches(*figsize_mm))
    else:
        fig = ax.figure

    ax.scatter(x, y, s=point_size, color=color, alpha=point_alpha, **kwargs)
    if fit is not None:
        grid = np.linspace(np.nanmin(x), np.nanmax(x), n_grid)
        if fit == "linear":
            coefficients = np.polyfit(x, y, 1)
            ax.plot(grid, np.polyval(coefficients, grid), color=fit_color, linewidth=2)
        elif fit == "logistic":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression().fit(x[:, None], y.astype(int))
            probability = model.predict_proba(grid[:, None])[:, 1]
            ax.plot(grid, probability, color=fit_color, linewidth=2, label="Logistic fit")
        else:
            raise ValueError(f"unknown fit {fit!r}")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    despine(ax)
    return fig, ax
