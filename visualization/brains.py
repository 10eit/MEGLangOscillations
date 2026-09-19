"""Source-space surface plots and standalone colour bars.

The functions wrap the pattern that was repeated in every figure notebook:
build a :class:`mne.SourceEstimate`, plot it on a fsaverage surface with a chosen
colour map / limit, optionally add label borders, and finally turn the 3-D
screenshot into a transparent matplotlib figure that can be embedded in a panel.

``mne`` is imported lazily inside the functions.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .style import mm_to_inches

__all__ = [
    "source_estimate",
    "as_clim",
    "plot_source_map",
    "add_label_borders",
    "crop_screenshot",
    "screenshot_figure",
    "source_map_figure",
    "truncate_colormap",
    "colorbar_figure",
]


def source_estimate(values, vertices, *, subject: str = "fsaverage", tmin: float = 0.0,
                    tstep: float = 0.01):
    """Wrap a vertex array into an :class:`mne.SourceEstimate`.

    Parameters
    ----------
    values : array_like
        ``(n_vertices,)`` or ``(n_vertices, n_times)``.
    vertices : sequence
        ``(lh_vertno, rh_vertno)`` of the source space.
    """
    import mne

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    return mne.SourceEstimate(values, vertices=vertices, tmin=tmin, tstep=tstep,
                              subject=subject)


def as_clim(clim):
    """Normalise colour limits to the MNE ``clim`` dictionary.

    Accepts ``None``, a ready-made dict, ``(vmin, vmax)`` or ``(vmin, midpoint, vmax)``.
    Two-sided limits (any value below zero) use ``lims``, one-sided limits use
    ``pos_lims``, matching MNE's conventions.
    """
    if clim is None or isinstance(clim, dict):
        return clim
    values = list(clim)
    if len(values) == 2:
        values = [values[0], (values[0] + values[1]) / 2, values[1]]
    if len(values) != 3:
        raise ValueError("clim must be a dict, (vmin, vmax) or (vmin, mid, vmax)")
    key = "lims" if min(values) < 0 else "pos_lims"
    return {"kind": "value", key: [float(v) for v in values]}


def plot_source_map(values, vertices=None, *, subjects_dir: str | None = None,
                    surf: str = "pial", views=("lateral", "medial"), hemi: str = "split",
                    cmap: str = "bwr", clim=None, cortex=None, colorbar: bool = False,
                    size=(800, 800), smoothing_steps: float = 10, background: str = "white",
                    alpha: float = 1.0, view_layout: str = "vertical",
                    borders=None, border_linewidth: float | None = None,
                    border_color: str = "black", subject: str = "fsaverage",
                    stc=None, **kwargs):
    """Plot a vertex map (or an existing source estimate) on a fsaverage surface.

    Parameters
    ----------
    values : array_like
        ``(n_vertices,)`` vertex map; ignored when ``stc`` is given.
    vertices : sequence | None
        ``(lh_vertno, rh_vertno)``; ignored when ``stc`` is given.
    subjects_dir : str | None
        FreeSurfer subjects directory (must contain ``fsaverage``).
    surf : {'pial', 'inflated', 'white', ...}
    views : sequence of str
    hemi : {'split', 'lh', 'rh', 'both'}
    cmap : str
    clim : dict | tuple | None
        See :func:`as_clim`.
    cortex : str | tuple | None
        Background cortex colour, e.g. ``"#CDBFB1"``.
    colorbar : bool
    size, smoothing_steps, background, alpha, view_layout, kwargs : forwarded to
        ``SourceEstimate.plot``.
    borders : sequence of mne.Label | None
        Labels drawn on top (e.g. language ROIs).
    border_linewidth : float | None
        Line width of the label borders when the backend exposes them.

    Returns
    -------
    mne.viz.Brain
    """
    if stc is None:
        if vertices is None:
            raise ValueError("provide `vertices` (or pass a ready-made `stc`)")
        stc = source_estimate(values, vertices, subject=subject)
    if cortex is not None and not isinstance(cortex, tuple):
        cortex = mpl.colors.to_rgb(cortex)

    brain = stc.plot(hemi=hemi, surface=surf, views=list(views),
                     subjects_dir=subjects_dir, size=size, smoothing_steps=smoothing_steps,
                     colormap=cmap, colorbar=colorbar, cortex=cortex,
                     background=background, alpha=alpha, view_layout=view_layout,
                     clim=as_clim(clim), **kwargs)
    if borders is not None:
        add_label_borders(brain, borders, color=border_color, linewidth=border_linewidth)
    return brain


def add_label_borders(brain, labels, *, color: str = "black", linewidth: float | None = None):
    """Draw label borders on a :class:`mne.viz.Brain` (thickened when possible)."""
    for label in labels:
        brain.add_label(label, borders=True, color=color)
    if linewidth:
        for hemi in getattr(brain, "geo", {}):
            geometry = brain.geo[hemi]
            for actor in getattr(geometry, "_borders", []) or []:
                try:
                    actor.GetProperty().SetLineWidth(linewidth)
                except AttributeError:  # pragma: no cover - backend specific
                    pass
    return brain


def crop_screenshot(image) -> np.ndarray:
    """Crop the white margins of a brain screenshot and add a transparent background.

    Parameters
    ----------
    image : np.ndarray
        ``(height, width)`` grey or ``(height, width, 3/4)`` RGB(A) array.

    Returns
    -------
    np.ndarray RGBA
    """
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.dstack([image] * 3)
    rgb = image[..., :3]
    non_white = (rgb != 255).any(axis=-1)
    if not non_white.any():
        return np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    rows, cols = np.where(non_white)
    cropped = rgb[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    alpha = np.where(np.all(cropped == 255, axis=-1), 0, 255).astype(np.uint8)
    return np.dstack([cropped, alpha])


def screenshot_figure(brain, *, transparent: bool = True, dpi: int = 600):
    """Turn a brain screenshot into a matplotlib figure with a transparent background."""
    image = brain.screenshot()
    if image is None:
        raise RuntimeError("brain.screenshot() returned None; is a 3-D backend available?")
    rgba = crop_screenshot(image)
    plt.ioff()
    fig, ax = plt.subplots()
    if transparent:
        fig.patch.set_alpha(0.0)
    ax.imshow(rgba)
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.set_dpi(dpi)
    return fig


def source_map_figure(values, vertices=None, *, screenshot: bool = True, **kwargs):
    """Convenience wrapper: plot a vertex map and return the cropped figure.

    Extra keyword arguments are forwarded to :func:`plot_source_map`.
    """
    brain = plot_source_map(values, vertices, **kwargs)
    return screenshot_figure(brain) if screenshot else brain


def truncate_colormap(cmap, lo: float = 0.0, hi: float = 1.0, n: int = 256):
    """Build a colormap from the ``[lo, hi]`` sub-range of ``cmap``.

    Useful for one-sided colour bars that reuse only half of a diverging map.
    """
    base = plt.get_cmap(cmap)
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"{base.name}_{lo}_{hi}", base(np.linspace(lo, hi, n)))


def colorbar_figure(cmap, vmin: float, vmax: float, *, label: str | None = None,
                    orientation: str = "horizontal", ticks=None, tick_labels=None,
                    figsize_mm=(45, 2.5), fontsize: float = 6,
                    label_fontsize: float = 8, truncate=None):
    """Standalone colour bar figure (for panels that need it separately).

    Parameters
    ----------
    cmap : str | Colormap
    vmin, vmax : float
    label : str | None
    orientation : {'horizontal', 'vertical'}
    ticks : sequence | None
    truncate : (float, float) | None
        Sub-range of ``cmap`` to use, e.g. ``(0.0, 0.5)`` for the negative half of
        ``bwr``.
    figsize_mm : (float, float)
        Figure size in millimetres; swap the values for a vertical bar.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if truncate is not None:
        cmap = truncate_colormap(cmap, *truncate)
    fig, ax = plt.subplots(figsize=mm_to_inches(*figsize_mm))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    bar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation=orientation)
    if ticks is not None:
        bar.set_ticks(ticks)
    if tick_labels is not None:
        bar.set_ticklabels(tick_labels)
    if label:
        bar.set_label(label, fontsize=label_fontsize)
    bar.ax.tick_params(labelsize=fontsize, width=0.2, length=2)
    bar.outline.set_linewidth(0.2)
    return fig
