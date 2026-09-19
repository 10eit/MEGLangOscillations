"""Posterior and information-criterion plots.

Used for drift-diffusion / Bayesian modelling output:

* :func:`plot_posterior_kde` - posterior densities of regression coefficients with a
  reference line and optional HDI shading.
* :func:`plot_ic_ranking` - ranked model comparison (WAIC / LOO, point estimate +/- SE).
* :func:`plot_pointwise_ic` - per-observation information criterion per model.
* :func:`plot_delta_ic` - difference of the pointwise criterion against the best model.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import condition_colors, despine, mm_to_inches

__all__ = ["plot_posterior_kde", "plot_ic_ranking", "plot_pointwise_ic", "plot_delta_ic"]


def _as_mapping(samples):
    if isinstance(samples, dict):
        return dict(samples)
    try:
        import pandas as pd

        if isinstance(samples, pd.DataFrame):
            return {column: samples[column].to_numpy() for column in samples.columns}
    except ImportError:  # pragma: no cover
        pass
    array = np.asarray(samples, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    return {f"param_{i}": array[:, i] for i in range(array.shape[1])}


def plot_posterior_kde(samples, *, reference: float | None = 0.0, hdi_prob: float | None = None,
                       colors=None, titles=None, xlabels=None, ncols: int | None = None,
                       share_x: bool = False, figsize_mm=(120, 50), palette=None,
                       cmap: str = "tab10", bandwidth=None, grid: int = 1000,
                       legend: bool = True, **kwargs):
    """Posterior densities of one or more coefficients.

    Parameters
    ----------
    samples : dict | pandas.DataFrame | 2-D array
        ``name -> samples``; a DataFrame is interpreted column-wise.
    reference : float | None
        Vertical reference line (e.g. 0 for regression coefficients).
    hdi_prob : float | None
        When given, the highest-density interval is shaded and its bounds returned.
    colors : dict | sequence | None
    titles, xlabels : sequence | None
        Per-panel title / x label.
    ncols : int | None
        Number of panels per row (default: as many as parameters, max 4).
    figsize_mm : (float, float)
        Size of the whole figure.

    Returns
    -------
    fig, axes
        ``axes`` is always a 1-D array of axes.
    """
    from scipy.stats import gaussian_kde

    from kit import stats as kit_stats

    samples = _as_mapping(samples)
    names = list(samples)
    ncols = ncols or min(len(names), 4)
    nrows = int(np.ceil(len(names) / ncols))
    figsize = mm_to_inches(figsize_mm[0], figsize_mm[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             sharex=share_x)
    axes = axes.ravel()

    if colors is None:
        color_map = condition_colors([str(n) for n in names], palette=palette, cmap=cmap)
    elif isinstance(colors, dict):
        color_map = colors
    else:
        color_map = dict(zip(names, colors))

    intervals = {}
    for i, (name, values) in enumerate(samples.items()):
        ax = axes[i]
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if values.size >= 2:
            density = gaussian_kde(values, bw_method=bandwidth)
            x = np.linspace(values.min(), values.max(), grid)
            ax.plot(x, density(x), color=color_map.get(name, "black"), linewidth=2)
            ax.fill_between(x, density(x), color=color_map.get(name, "black"), alpha=0.2,
                            linewidth=0)
            if hdi_prob is not None:
                low, high = kit_stats.hdi(values, prob=hdi_prob)
                intervals[name] = (low, high)
                ax.hdi_interval = (low, high)
                ax.axvspan(low, high, color=color_map.get(name, "black"), alpha=0.12,
                           linewidth=0)
        if reference is not None:
            ax.axvline(reference, color="grey", linestyle="--", linewidth=1)
        ax.set_yticks([])
        ax.set_title(str(name) if titles is None else str(titles[i]), fontsize=8)
        if xlabels is not None:
            ax.set_xlabel(str(xlabels[i]))
        despine(ax)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    if legend and len(names) > 1 and ncols == 1:
        axes[0].legend()
    fig.tight_layout()
    fig.hdi_intervals = intervals  # convenience: HDI bounds per parameter
    return fig, axes


def plot_ic_ranking(comp, *, ic: str = "waic", model_col: str = "model",
                    value_col: str | None = None, se_col: str | None = None,
                    scale: float = 1.0, color: str = "#4C72B0", annotate_best: str = "best",
                    markercolor: str = "white", xlabel: str | None = None,
                    figsize_mm=(60, 50), invert_y: bool = True, ax=None, **kwargs):
    """Ranked model comparison (point estimate +/- standard error).

    Parameters
    ----------
    comp : pandas.DataFrame
        Model comparison table with one row per model; typically the output of
        :func:`kit.hddm.compare` (``az.compare``).
    ic : {'waic', 'loo'}
        Used to auto-detect the value / SE columns.
    value_col, se_col : str | None
        Column names; when ``None`` the criterion column is detected from ``elpd_<ic>``
        / ``elpd`` (higher is better) or ``<ic>`` / ``ic`` (deviance, lower is better),
        and the error column from ``se`` / ``ic_se``.
    scale : float
        Divisor applied to the values (e.g. ``1e4`` for readability).
    invert_y : bool
        Put the best model at the top.

    Returns
    -------
    fig, ax
    """
    frame = comp.copy()
    if model_col not in frame.columns:
        frame = frame.reset_index().rename(columns={"index": model_col})

    if value_col is None:
        if f"elpd_{ic}" in frame.columns:
            value_col, higher_is_better = f"elpd_{ic}", True
        elif "elpd" in frame.columns:
            value_col, higher_is_better = "elpd", True
        elif f"{ic}" in frame.columns:
            value_col, higher_is_better = f"{ic}", False
        elif "ic" in frame.columns:
            value_col, higher_is_better = "ic", False
        else:
            raise KeyError(f"no information-criterion column in {list(frame.columns)}")
    else:
        higher_is_better = "elpd" in value_col

    if se_col is None:
        for candidate in ("se", "ic_se", f"se_{ic}"):
            if candidate in frame.columns:
                se_col = candidate
                break

    # best model first
    frame = frame.sort_values(value_col, ascending=not higher_is_better).reset_index(drop=True)
    if not invert_y:
        frame = frame.iloc[::-1].reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)

    values = frame[value_col].to_numpy(dtype=float) / scale
    errors = (frame[se_col].to_numpy(dtype=float) / scale
              if se_col and se_col in frame.columns else None)

    if ax is None:
        fig, ax = plt.subplots(figsize=mm_to_inches(*figsize_mm))
    else:
        fig = ax.figure

    y = frame["rank"].to_numpy()
    ax.errorbar(values, y, xerr=errors, fmt="o", markersize=5,
                markerfacecolor=markercolor, markeredgecolor="black", markeredgewidth=1,
                ecolor="black", elinewidth=1, capsize=3, zorder=3, **kwargs)
    if annotate_best:
        ax.annotate(annotate_best, xy=(values[0], y[0]), xytext=(0, -10),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([str(int(r)) for r in frame["rank"]])
    if invert_y:
        ax.invert_yaxis()
    if xlabel is None:
        xlabel = ("elpd (log scale)" if higher_is_better else f"{ic.upper()} (deviance)")
        if scale != 1:
            xlabel += f" / {scale:g}"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ranked models")
    span = np.nanmax(values) - np.nanmin(values)
    padding = 0.15 * span if span > 0 else 0.1
    ax.set_xlim(np.nanmin(values) - padding, np.nanmax(values) + padding)
    despine(ax)
    fig.tight_layout()
    return fig, ax


def plot_pointwise_ic(pointwise, *, value_col: str = "value", model_col: str = "model",
                      observation_col: str = "observation", order=None,
                      clip_percentile: float | None = None, max_points: int = 300,
                      color: str = "#4C72B0", point_color: str = "black",
                      figsize_mm=(60, 50), xlabel: str = "Ranked models",
                      ylabel: str = "Pointwise criterion", seed=0, ax=None, **kwargs):
    """Box plot (with subsampled points) of a pointwise information criterion.

    Parameters
    ----------
    pointwise : pandas.DataFrame
        Long format: one row per model and observation.
    order : sequence | None
        Model order; defaults to the sorted order of appearance.
    clip_percentile : float | None
        Display cut-off (percentile) for the y axis.  Values above it are *kept* and
        marked with triangles labelled ``n=<count>``.
    max_points : int
        Number of points drawn per model (random subsample for readability).

    Returns
    -------
    fig, ax
    """
    rng = np.random.default_rng(seed)
    models = list(order) if order is not None else list(dict.fromkeys(pointwise[model_col]))
    arrays = [pointwise.loc[pointwise[model_col] == m, value_col].to_numpy(dtype=float)
              for m in models]

    if ax is None:
        fig, ax = plt.subplots(figsize=mm_to_inches(*figsize_mm))
    else:
        fig = ax.figure

    cut = np.percentile(np.concatenate(arrays), clip_percentile) if clip_percentile else None
    box_data = [a if cut is None else a[a <= cut] for a in arrays]
    artists = ax.boxplot(box_data, widths=0.6, patch_artist=True, showfliers=False,
                         **kwargs)
    for patch in artists["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1)
    for key in ("whiskers", "caps", "medians"):
        for item in artists[key]:
            item.set_color("black")
            item.set_linewidth(1)

    for position, array in enumerate(arrays, start=1):
        pool = array if cut is None else array[array <= cut]
        if pool.size > max_points:
            pool = rng.choice(pool, max_points, replace=False)
        ax.scatter(position + rng.uniform(-0.18, 0.18, pool.size), pool, s=1.5,
                   color=point_color, alpha=0.25, linewidths=0)
        if cut is not None:
            n_clipped = int(np.sum(array > cut))
            if n_clipped:
                ax.scatter(position, cut * 0.985, marker="^", s=22, facecolor="white",
                           edgecolor="black", linewidth=0.8, zorder=5)
                ax.annotate(f"n={n_clipped}", xy=(position, cut * 0.985), xytext=(0, -11),
                            textcoords="offset points", ha="center", va="top", fontsize=6)

    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels([str(i) for i in range(1, len(models) + 1)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    despine(ax)
    fig.tight_layout()
    return fig, ax


def plot_delta_ic(delta, *, value_col: str = "delta", model_col: str = "model",
                  reference_label: str = "best", color: str = "#4C72B0",
                  point_color: str = "grey", show_points: bool = True,
                  figsize_mm=(60, 50), xlabel=None, ylabel: str = "Ranked models",
                  ax=None, **kwargs):
    """Dot-and-IQR plot of the pointwise criterion difference to the reference model.

    Parameters
    ----------
    delta : pandas.DataFrame
        Output of :func:`kit.hddm.delta_ic` (columns ``model`` and ``delta``).
    reference_label : str
        Text annotated at ``delta = 0``.

    Returns
    -------
    fig, ax
    """
    models = list(dict.fromkeys(delta[model_col]))
    if ax is None:
        fig, ax = plt.subplots(figsize=mm_to_inches(*figsize_mm))
    else:
        fig = ax.figure

    rng = np.random.default_rng(0)
    for row, model in enumerate(models):
        values = delta.loc[delta[model_col] == model, value_col].to_numpy(dtype=float)
        if show_points:
            ax.scatter(values, row + rng.uniform(-0.16, 0.16, values.size), s=8,
                       facecolors="none", edgecolors=point_color, linewidths=0.5,
                       alpha=0.55, rasterized=True)
        q1, median, q3 = np.nanpercentile(values, [25, 50, 75])
        ax.hlines(row, q1, q3, color=color, linewidth=3, zorder=3)
        ax.scatter([median], [row], s=28, color=color, edgecolors="black", linewidths=0.7,
                   zorder=4)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    if reference_label:
        ax.text(0, -0.55, reference_label, ha="center", va="bottom", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([str(i + 2) for i in range(len(models))])
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$IC" if xlabel is None else xlabel)
    ax.set_ylabel(ylabel)
    despine(ax)
    fig.tight_layout()
    return fig, ax
