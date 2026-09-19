"""Time-course plots: mean +/- error curves and temporal-generalization matrices.

Both functions accept an existing axes (``ax=``) so that panels can be composed, and
create a figure of the requested millimetre size otherwise.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .style import mm_to_inches

__all__ = ["mean_error", "cluster_bar", "plot_series", "plot_tg_matrix"]


def mean_error(data, *, error: str | None = "se", axis: int = 0):
    """Mean and (optionally) error of ``data`` along ``axis``.

    ``error`` is ``'se'`` (standard error), ``'sd'`` (standard deviation) or ``None``.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        return data, None
    mean = np.nanmean(data, axis=axis)
    if error is None:
        return mean, None
    if error == "se":
        n = np.sum(~np.isnan(data), axis=axis)
        err = np.nanstd(data, axis=axis, ddof=1) / np.sqrt(n)
    elif error == "sd":
        err = np.nanstd(data, axis=axis, ddof=1)
    else:
        raise ValueError(f"unknown error {error!r}")
    return mean, err


def cluster_bar(ax, intervals, y: float, *, color="k", linewidth: float = 2, **kwargs):
    """Draw horizontal significance bars for ``[(start, end), ...]`` intervals.

    Intended for ``ClusterResult.intervals``.
    """
    for start, end in intervals:
        ax.hlines(y=y, xmin=start, xmax=end, colors=color, linewidth=linewidth,
                  **kwargs)
    return ax


def plot_series(data, times=None, *, ax=None, color=None, label=None, chance=None,
                smooth_sigma: float | None = None, error: str | None = "se",
                linewidth: float = 2, alpha: float = 0.2, intervals=None,
                interval_y=None, interval_color=None, events=None, event_labels=None,
                xlabel=None, ylabel=None, xlim=None, ylim=None, legend: bool = False,
                figsize_mm=(120, 30), **kwargs):
    """Plot a mean time course with an error band and optional significance bars.

    Parameters
    ----------
    data : array_like
        ``(n_observations, n_times)`` or ``(n_times,)``.
    times : array_like | None
        x axis; defaults to sample indices.
    ax : matplotlib.axes.Axes | None
    color : str | tuple | None
    label : str | None
    chance : float | None
        Reference value drawn as a dashed line.
    smooth_sigma : float | None
        Gaussian smoothing applied to the plotted mean (samples).
    error : {'se', 'sd', None}
    intervals : sequence of (float, float) | None
        Significant intervals (e.g. ``ClusterResult.intervals``); requires
        ``interval_y``.
    interval_y : float | None
        Height of the significance bars.
    events : sequence of float | None
        Vertical event markers.
    event_labels : sequence of str | None
        Tick labels used together with ``events``.
    figsize_mm : (float, float)

    Returns
    -------
    fig, ax
    """
    data = np.asarray(data, dtype=float)
    if times is None:
        times = np.arange(data.shape[-1])
    times = np.asarray(times, dtype=float)
    mean, err = mean_error(data, error=error)
    if smooth_sigma:
        mean = gaussian_filter1d(mean, sigma=smooth_sigma)
    color = color if color is not None else "black"

    if ax is None:
        fig, ax = plt.subplots(figsize=mm_to_inches(*figsize_mm))
    else:
        fig = ax.figure

    ax.plot(times, mean, color=color, linewidth=linewidth, label=label, **kwargs)
    if err is not None:
        ax.fill_between(times, mean - err, mean + err, color=color, alpha=alpha,
                        linewidth=0)
    if chance is not None:
        ax.axhline(chance, color="k", linestyle="--", linewidth=1.2)
    if intervals is not None:
        if interval_y is None:
            raise ValueError("interval_y is required when intervals are given")
        cluster_bar(ax, intervals, interval_y,
                    color=interval_color or color, linewidth=linewidth)
    if events is not None:
        for event in events:
            ax.axvline(event, color="grey", linestyle="--", linewidth=1.2, alpha=0.6)
        if event_labels is not None:
            ax.set_xticks(list(events))
            ax.set_xticklabels(list(event_labels))
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend and label is not None:
        ax.legend()
    return fig, ax


def plot_tg_matrix(matrix, *, ax=None, times=None, cmap: str = "RdBu_r", vmin=None,
                   vmax=None, sig_mask=None, contour_color: str = "gray",
                   contour_linewidth: float = 0.5, rectangles=None, cbar: bool = False,
                   cbar_label: str = "ROC-AUC", xlabel: str | None = None,
                   ylabel: str | None = None, ticks=None, tick_labels=None,
                   figsize_mm=(40, 40), **kwargs):
    """Plot a temporal-generalization matrix with optional significance contours.

    Parameters
    ----------
    matrix : array_like, shape (n_train, n_test)
    times : array_like | None
        Axis values of train and test; defaults to sample indices.
    sig_mask : array_like of bool | None
        Significant train/test pairs, drawn as contours.
    rectangles : sequence of (x0, y0, width, height) | None
        Reference boxes (e.g. the diagonal blocks used to mark train==test regions).
    cbar, cbar_label : bool / str
    ticks, tick_labels : sequence | None
        Tick positions / labels of the axis.
    figsize_mm : (float, float)

    Returns
    -------
    fig, ax
    """
    matrix = np.asarray(matrix, dtype=float)
    if times is None:
        times = np.arange(matrix.shape[0])
    times = np.asarray(times, dtype=float)
    extent = [times[0], times[-1], times[0], times[-1]]

    if ax is None:
        fig, ax = plt.subplots(figsize=mm_to_inches(*figsize_mm))
    else:
        fig = ax.figure

    image = ax.imshow(matrix, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=extent, aspect="auto", **kwargs)
    if sig_mask is not None:
        grid_x, grid_y = np.meshgrid(times, times)
        ax.contour(grid_x, grid_y, np.asarray(sig_mask, dtype=float),
                   colors=contour_color, linewidths=contour_linewidth, levels=[0.5])
    for x0, y0, width, height in (rectangles or []):
        ax.add_patch(plt.Rectangle((x0, y0), width, height, fill=False,
                                   edgecolor="black", linewidth=0.5, zorder=10))
    if ticks is not None:
        ax.set_xticks(list(ticks))
        ax.set_yticks(list(ticks))
        if tick_labels is not None:
            ax.set_xticklabels(list(tick_labels))
            ax.set_yticklabels(list(tick_labels))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if cbar:
        bar = fig.colorbar(image, ax=ax, shrink=0.8)
        bar.set_label(cbar_label)
    return fig, ax
