"""Cluster-based permutation testing for time courses and vertex maps.

Entry points
------------
:func:`cluster_permutation_1d`
    One-sample (or paired) test on a parameter axis such as time or frequency.
:func:`cluster_permutation_vertices`
    One-sample test on one value per observation (e.g. searchlight scores) with a
    spatial adjacency matrix.
:func:`cluster_permutation_spatiotemporal`
    One-sample test on ``(time, vertex)`` maps with a spatial adjacency matrix.
:func:`cluster_permutation_surrogate`
    Cluster correction when the null is an *empirical* surrogate distribution rather
    than a permutation of the data.
:func:`cluster_permutation_signflip`
    Cluster correction from a summary statistic map only (no per-subject data),
    randomising by sign flipping.

Conventions
-----------
* The cluster-forming alpha is converted into a statistic threshold in exactly one
  place, :func:`cluster_forming_threshold`, which always returns a **positive
  magnitude**.  Internally the data are re-oriented for left-tailed tests and MNE is
  always called with ``tail=1`` / ``tail=0``, so no sign convention of any MNE
  version can leak into the results.  ``ClusterResult.stat`` is always expressed in
  the orientation of the input data.
* ``cluster_forming`` and ``alpha`` are two different things: the former defines the
  threshold that forms candidate clusters, the latter decides which clusters are
  reported (``sig_mask``).
* ``min_cluster_size`` / ``min_cluster_duration`` only decide *which* clusters end up
  in ``sig_mask``; they do not change the cluster p-values.

Every threshold is an argument; nothing study-specific is hard-coded.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy import stats as ss

__all__ = [
    "ClusterResult",
    "cluster_forming_threshold",
    "cluster_permutation_1d",
    "cluster_permutation_vertices",
    "cluster_permutation_spatiotemporal",
    "cluster_permutation_surrogate",
    "cluster_permutation_signflip",
    "cluster_recurrence",
]


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class ClusterResult:
    """Outcome of a cluster-based permutation test.

    Attributes
    ----------
    stat : np.ndarray
        Observed statistic map, same shape as the averaged input.
    clusters : list of np.ndarray
        Boolean masks, one per cluster (same shape as ``stat``).
    cluster_pvals : np.ndarray
        Cluster-level p-value per entry of ``clusters``.
    sig_mask : np.ndarray
        Boolean mask of the clusters that survived ``alpha`` and the size / duration
        filters.
    alpha : float
        Cluster-level significance level that was applied.
    null_max : np.ndarray | None
        Null distribution of the maximal cluster statistic (when available).
    times : np.ndarray | None
        Time axis of the tested dimension, if given.
    tstep : float | None
        Sampling step of the tested dimension, if known.
    """

    stat: np.ndarray
    clusters: list = field(default_factory=list)
    cluster_pvals: np.ndarray = field(default_factory=lambda: np.array([]))
    sig_mask: np.ndarray = None
    alpha: float = 0.05
    null_max: np.ndarray | None = None
    times: np.ndarray | None = None
    tstep: float | None = None

    def __len__(self) -> int:
        return len(self.clusters)

    @property
    def sig_clusters(self) -> list:
        """Clusters with ``p < alpha`` (after the size / duration filters)."""
        return [c for c, p in zip(self.clusters, self.cluster_pvals) if p < self.alpha]

    @property
    def sig_cluster_pvals(self) -> np.ndarray:
        return np.array([p for p in self.cluster_pvals if p < self.alpha])

    @property
    def stat_masked(self) -> np.ndarray:
        """``stat`` zeroed outside the significant mask (ready for source plots)."""
        if self.sig_mask is None:
            return np.asarray(self.stat, dtype=float)
        return np.where(np.asarray(self.sig_mask, dtype=bool), self.stat, 0.0)

    @property
    def time_profile(self) -> np.ndarray:
        """Number of significant units per time point (1 for 1-D results)."""
        if self.sig_mask is None:
            return np.zeros(np.shape(self.stat)[0], dtype=int)
        mask = np.asarray(self.sig_mask, dtype=bool)
        if mask.ndim <= 1:
            return mask.astype(int)
        return mask.sum(axis=tuple(range(1, mask.ndim)))

    @property
    def intervals(self) -> list:
        """Contiguous significant intervals ``[(start, end), ...]`` along the time axis.

        For 2-D (time x vertex) results the interval spans any significant vertex.
        Values are in seconds when ``times`` / ``tstep`` were provided, otherwise in
        sample indices.
        """
        idx = np.flatnonzero(self.time_profile > 0)
        if idx.size == 0:
            return []
        breaks = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        out = []
        for run in breaks:
            if self.times is not None:
                out.append((float(np.asarray(self.times)[run[0]]),
                            float(np.asarray(self.times)[run[-1]])))
            elif self.tstep:
                out.append((float(run[0] * self.tstep), float(run[-1] * self.tstep)))
            else:
                out.append((int(run[0]), int(run[-1])))
        return out


# --------------------------------------------------------------------------- #
# thresholds and small helpers
# --------------------------------------------------------------------------- #
def cluster_forming_threshold(cluster_forming: float, *, df: int | None = None,
                             tail: int = 0, statistic: str = "t") -> float:
    """Positive magnitude of the cluster-forming threshold for a given alpha.

    ==================  =========================================================
    ``tail``            threshold
    ==================  =========================================================
    0 (two-sided)       ``t.ppf(1 - cluster_forming / 2, df)``
    +1 (right tail)     ``t.ppf(1 - cluster_forming, df)``
    -1 (left tail)      ``-t.ppf(cluster_forming, df)`` (returned positive)
    ==================  =========================================================

    ``statistic='z'`` uses the normal distribution instead (``df`` then unused).
    """
    if not 0 < cluster_forming < 1:
        raise ValueError("cluster_forming must lie in (0, 1)")
    if statistic == "t":
        if df is None:
            raise ValueError("df is required for a t-based cluster-forming threshold")
        if tail == 0:
            return float(ss.t.ppf(1 - cluster_forming / 2, df))
        if tail == 1:
            return float(ss.t.ppf(1 - cluster_forming, df))
        if tail == -1:
            return float(-ss.t.ppf(cluster_forming, df))
    elif statistic == "z":
        if tail == 0:
            return float(ss.norm.ppf(1 - cluster_forming / 2))
        if tail == 1:
            return float(ss.norm.ppf(1 - cluster_forming))
        if tail == -1:
            return float(-ss.norm.ppf(cluster_forming))
    raise ValueError(f"unsupported statistic / tail: {statistic!r}, {tail!r}")


def _orient(data: np.ndarray, tail: int):
    """Orient data so the test is always run on the right tail.

    Returns ``(oriented, mne_tail, sign)`` where ``sign`` maps the oriented statistic
    back to the orientation of the input.
    """
    if tail == -1:
        return -np.asarray(data, dtype=float), 1, -1.0
    if tail in (0, 1):
        return np.asarray(data, dtype=float), tail, 1.0
    raise ValueError(f"tail must be -1, 0 or 1, got {tail!r}")


def _resolve_tstep(times, tstep):
    if tstep is not None:
        return float(tstep)
    if times is not None and len(times) > 1:
        return float(np.mean(np.diff(np.asarray(times, dtype=float))))
    return None


def _seed_kwargs(func, seed):
    """Pass the random seed in the keyword the installed MNE version expects."""
    import inspect

    return {"rng": seed} if "rng" in inspect.signature(func).parameters else {"seed": seed}


def _cluster_to_mask(cluster, shape) -> np.ndarray:
    """Normalise any MNE cluster representation into a boolean mask of ``shape``.

    MNE returns clusters in several formats depending on the dimensionality and on
    ``out_type``: a boolean array, a tuple of slices / index vectors (one per axis,
    which is what 1-D data produce) or a flat index array.  All of them are mapped
    to the same boolean mask here.
    """
    shape = tuple(int(s) for s in shape)
    size = int(np.prod(shape))

    # already a boolean mask of the right shape
    if isinstance(cluster, np.ndarray) and cluster.dtype == bool:
        if cluster.shape == shape:
            return cluster
        if cluster.size == size:
            return cluster.reshape(shape)
    # boolean mask of a different (flat) shape
    if isinstance(cluster, np.ndarray) and cluster.dtype != bool:
        flat = np.zeros(size, dtype=bool)
        flat[np.asarray(cluster).astype(int).ravel()] = True
        return flat.reshape(shape)
    # tuple with one entry per axis (slice, boolean vector or index vector)
    if isinstance(cluster, tuple) and len(cluster) == len(shape):
        masks = []
        for part, dim in zip(cluster, shape):
            if isinstance(part, slice):
                mask = np.zeros(dim, dtype=bool)
                mask[part] = True
            else:
                part = np.asarray(part)
                mask = np.zeros(dim, dtype=bool)
                if part.dtype == bool:
                    mask[: part.size] = part
                else:
                    mask[part.astype(int)] = True
            masks.append(mask)
        if masks:
            return functools.reduce(lambda a, b: a[..., None] & b, masks)
    raise ValueError(f"cannot interpret cluster of type {type(cluster).__name__}")


def _cluster_duration(mask: np.ndarray, tstep: float) -> float:
    """Duration (in seconds) covered by a cluster mask whose axis 0 is time."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 1:
        return float(mask.sum() * tstep)
    time_idx = np.flatnonzero(mask.any(axis=tuple(range(1, mask.ndim))))
    if time_idx.size == 0:
        return 0.0
    return float((time_idx[-1] - time_idx[0] + 1) * tstep)


def _cluster_stat(values: np.ndarray, mask: np.ndarray, tail: int,
                  how: str = "sum") -> float:
    """Cluster summary statistic as a positive magnitude."""
    if how == "size":
        return float(np.sum(mask))
    if tail == 0:
        return float(np.abs(values[mask]).sum())
    signed = float(values[mask].sum())
    return signed if tail == 1 else -signed


def _select_clusters(clusters, cluster_pvals, *, alpha, min_cluster_size,
                     min_cluster_duration, tstep, shape=None):
    """Apply cluster-level alpha and size / duration filters.

    Returns ``(sig_mask, kept_clusters, kept_pvals)``.
    """
    if shape is None:
        shape = np.asarray(clusters[0]).shape if len(clusters) else (0,)
    keep, keep_p = [], []
    for mask, p in zip(clusters, cluster_pvals):
        mask = np.asarray(mask, dtype=bool)
        if p >= alpha:
            continue
        if min_cluster_size and int(mask.sum()) < min_cluster_size:
            continue
        if min_cluster_duration:
            if not tstep:
                raise ValueError("min_cluster_duration requires `tstep` or `times`")
            if _cluster_duration(mask, tstep) < min_cluster_duration:
                continue
        keep.append(mask)
        keep_p.append(p)
    sig_mask = np.zeros(shape, dtype=bool)
    for mask in keep:
        sig_mask |= mask
    return sig_mask, keep, np.asarray(keep_p)


def _subset_units(data, adjacency, exclude, axis):
    """Drop excluded units from ``data`` and ``adjacency``; return the kept indices."""
    n_units = data.shape[axis]
    keep = np.arange(n_units)
    if exclude is not None:
        keep = np.setdiff1d(keep, np.asarray(exclude, dtype=int))
    if adjacency is not None:
        adjacency = sp.csr_matrix(adjacency)
        adjacency = adjacency[keep][:, keep]
    return np.take(data, keep, axis=axis), adjacency, keep, n_units


def _expand_mask(mask, keep, n_units, axis):
    """Expand a mask over a subset of units back to the full unit space."""
    shape = list(np.shape(mask))
    shape[axis] = n_units
    full = np.zeros(shape, dtype=bool)
    index = [slice(None)] * full.ndim
    index[axis] = np.asarray(keep, dtype=int)
    full[tuple(index)] = mask
    return full


# --------------------------------------------------------------------------- #
# 1-D time courses
# --------------------------------------------------------------------------- #
def cluster_permutation_1d(data, *, chance: float = 0.0, tail: int = 0,
                           cluster_forming: float = 0.05, threshold: float | None = None,
                           n_permutations: int = 1000, adjacency=None,
                           times: np.ndarray | None = None, tstep: float | None = None,
                           min_cluster_size: int = 0, min_cluster_duration: float = 0.0,
                           alpha: float = 0.05, n_jobs: int = 1, seed=None,
                           verbose=None) -> ClusterResult:
    """Cluster-based permutation one-sample test on ``(n_observations, n_points)``.

    Parameters
    ----------
    data : array_like, shape (n_observations, n_points)
        Observations along axis 0 (e.g. subjects), test units along axis 1 (time).
    chance : float
        Reference value the data are tested against (0.5 for ROC-AUC, 0 for rho/R²).
    tail : {-1, 0, 1}
        Direction of the alternative.
    cluster_forming : float
        Cluster-forming alpha, converted with :func:`cluster_forming_threshold`.
    threshold : float | None
        Explicit positive threshold; overrides ``cluster_forming``.
    n_permutations : int
    adjacency : sparse matrix | None
        Adjacency between the ``n_points`` units; ``None`` assumes a 1-D chain.
    times, tstep : axis values of the tested dimension (for ``intervals`` and for
        ``min_cluster_duration``).
    min_cluster_size, min_cluster_duration : filters on the *reported* clusters
        (units and seconds respectively).
    alpha : float
        Cluster-level significance level.
    n_jobs, seed, verbose : forwarded to MNE.

    Returns
    -------
    ClusterResult
    """
    from mne.stats import permutation_cluster_1samp_test

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be 2-D (n_observations, n_points)")

    oriented, mne_tail, sign = _orient(data - chance, tail)
    thr = threshold if threshold is not None else cluster_forming_threshold(
        cluster_forming, df=data.shape[0] - 1, tail=tail)

    t_obs, clusters, pvals, h0 = permutation_cluster_1samp_test(
        oriented, threshold=thr, n_permutations=n_permutations, tail=mne_tail,
        adjacency=adjacency, n_jobs=n_jobs, out_type="mask", verbose=verbose,
        **_seed_kwargs(permutation_cluster_1samp_test, seed))

    tstep = _resolve_tstep(times, tstep)
    shape = (data.shape[1],)
    clusters = [_cluster_to_mask(c, shape) for c in clusters]
    sig_mask, _, _ = _select_clusters(
        clusters, pvals, alpha=alpha, min_cluster_size=min_cluster_size,
        min_cluster_duration=min_cluster_duration, tstep=tstep, shape=shape)

    return ClusterResult(stat=sign * np.asarray(t_obs), clusters=clusters,
                         cluster_pvals=np.asarray(pvals), sig_mask=sig_mask,
                         alpha=alpha, null_max=h0, times=times, tstep=tstep)


# --------------------------------------------------------------------------- #
# single map over units (searchlight)
# --------------------------------------------------------------------------- #
def cluster_permutation_vertices(values, adjacency, *, exclude=None, chance: float = 0.0,
                                transform: str | None = None, tail: int = 0,
                                cluster_forming: float = 0.05,
                                threshold: float | None = None,
                                n_permutations: int = 1000, min_cluster_size: int = 0,
                                alpha: float = 0.05, n_jobs: int = 1, seed=None,
                                verbose=None) -> ClusterResult:
    """Cluster-based permutation test on one value per observation and unit.

    Suitable for searchlight scores or searchlight RSA maps: one value per subject and
    vertex, tested against ``chance`` with spatial clusters.

    Parameters
    ----------
    values : array_like, shape (n_observations, n_units)
    adjacency : sparse matrix, shape (n_units, n_units)
        Spatial adjacency (``kit.source.source_adjacency``).
    exclude : array_like | None
        Unit indices to drop (e.g. medial wall).  They are removed before testing and
        re-inserted as non-significant.
    chance : float
        Reference value (0.5 for ROC-AUC, 0 for correlation maps).
    transform : {'fisher_z', None}
        ``fisher_z`` is the standard choice for correlation maps.
    others : see :func:`cluster_permutation_1d`.

    Returns
    -------
    ClusterResult
        ``stat`` and ``sig_mask`` have shape ``(n_units,)``.
    """
    from mne.stats import permutation_cluster_1samp_test

    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be 2-D (n_observations, n_units)")
    if transform == "fisher_z":
        values = np.arctanh(np.clip(values, -1 + 1e-6, 1 - 1e-6))
    elif transform not in (None, "none"):
        raise ValueError(f"unknown transform {transform!r}")

    sub, adj, keep, n_units = _subset_units(values, adjacency, exclude, axis=1)

    oriented, mne_tail, sign = _orient(sub - chance, tail)
    thr = threshold if threshold is not None else cluster_forming_threshold(
        cluster_forming, df=sub.shape[0] - 1, tail=tail)

    t_obs, clusters, pvals, h0 = permutation_cluster_1samp_test(
        oriented, threshold=thr, n_permutations=n_permutations, tail=mne_tail,
        adjacency=adj, n_jobs=n_jobs, out_type="mask", verbose=verbose,
        **_seed_kwargs(permutation_cluster_1samp_test, seed))

    shape = (keep.size,)
    sub_clusters = [_cluster_to_mask(c, shape) for c in clusters]
    sub_sig, _, _ = _select_clusters(
        sub_clusters, pvals, alpha=alpha, min_cluster_size=min_cluster_size,
        min_cluster_duration=0.0, tstep=None, shape=shape)

    stat_full = np.zeros(n_units)
    stat_full[keep] = sign * np.asarray(t_obs).reshape(-1)

    return ClusterResult(
        stat=stat_full,
        clusters=[_expand_mask(c, keep, n_units, 0) for c in sub_clusters],
        cluster_pvals=np.asarray(pvals),
        sig_mask=_expand_mask(sub_sig, keep, n_units, 0),
        alpha=alpha, null_max=h0)


# --------------------------------------------------------------------------- #
# spatio-temporal maps
# --------------------------------------------------------------------------- #
def cluster_permutation_spatiotemporal(data, *, adjacency, exclude=None, chance: float = 0.0,
                                      tail: int = 0, cluster_forming: float = 0.05,
                                      threshold: float | None = None,
                                      n_permutations: int = 1000,
                                      times: np.ndarray | None = None,
                                      tstep: float | None = None,
                                      min_cluster_size: int = 0,
                                      min_cluster_duration: float = 0.0,
                                      alpha: float = 0.05, n_jobs: int = 1, seed=None,
                                      verbose=None) -> ClusterResult:
    """Cluster-based permutation test on ``(n_observations, n_times, n_units)``.

    Parameters
    ----------
    data : array_like
        ``(n_observations, n_times, n_units)`` (MNE convention: time before units).
    adjacency : sparse matrix
        Spatial adjacency (``n_units x n_units``); it is combined with the temporal
        adjacency internally.  A pre-combined ``(n_times * n_units)`` matrix is
        accepted as well.
    exclude : array_like | None
        Unit indices to drop (medial wall, visual cortex, ...).
    times, tstep : time axis; needed for ``min_cluster_duration`` and ``intervals``.
    others : see :func:`cluster_permutation_1d`.

    Returns
    -------
    ClusterResult
        ``stat`` / ``sig_mask`` have shape ``(n_times, n_units)``.
        ``result.intervals`` gives the cluster time spans;
        :func:`cluster_recurrence` gives the per-unit recurrence map.
    """
    from mne.stats import combine_adjacency, spatio_temporal_cluster_1samp_test

    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError("data must be 3-D (n_observations, n_times, n_units)")

    sub, adj, keep, n_units = _subset_units(data, adjacency, exclude, axis=2)
    n_times = sub.shape[1]
    if adj is not None and adj.shape[0] != n_times * adj.shape[1]:
        adj = combine_adjacency(n_times, adj)

    oriented, mne_tail, sign = _orient(sub - chance, tail)
    thr = threshold if threshold is not None else cluster_forming_threshold(
        cluster_forming, df=sub.shape[0] - 1, tail=tail)

    t_obs, clusters, pvals, h0 = spatio_temporal_cluster_1samp_test(
        oriented, threshold=thr, n_permutations=n_permutations, tail=mne_tail,
        adjacency=adj, n_jobs=n_jobs, out_type="mask", verbose=verbose,
        **_seed_kwargs(spatio_temporal_cluster_1samp_test, seed))

    tstep = _resolve_tstep(times, tstep)
    shape = (n_times, keep.size)
    sub_clusters = [_cluster_to_mask(c, shape) for c in clusters]
    sub_sig, _, _ = _select_clusters(
        sub_clusters, pvals, alpha=alpha, min_cluster_size=min_cluster_size,
        min_cluster_duration=min_cluster_duration, tstep=tstep, shape=shape)

    stat_full = np.zeros((n_times, n_units))
    stat_full[:, keep] = sign * np.asarray(t_obs)

    return ClusterResult(
        stat=stat_full,
        clusters=[_expand_mask(c, keep, n_units, 1) for c in sub_clusters],
        cluster_pvals=np.asarray(pvals),
        sig_mask=_expand_mask(sub_sig, keep, n_units, 1),
        alpha=alpha, null_max=h0, times=times, tstep=tstep)


# --------------------------------------------------------------------------- #
# empirical surrogate null
# --------------------------------------------------------------------------- #
def cluster_permutation_surrogate(true_map, surrogate_maps, *, cluster_forming: float = 0.05,
                                 tail: int = 1, times: np.ndarray | None = None,
                                 tstep: float | None = None, min_cluster_size: int = 0,
                                 alpha: float = 0.05,
                                 connectivity: int | None = None) -> ClusterResult:
    """Cluster correction against an empirical surrogate distribution.

    The observed map is z-scored against the surrogate distribution and the same is
    done for every surrogate, giving the null distribution of the maximal cluster
    statistic.  Use this when label-shuffled surrogates - not a permutation of the
    observations - define the null.

    Parameters
    ----------
    true_map : array_like
        Observed statistic map; 1-D time course or a grid of any shape.
    surrogate_maps : array_like, shape (n_surrogates, *true_map.shape)
    cluster_forming : float
        Cluster-forming alpha applied in z-space.
    tail : {-1, 0, 1}
        Direction of the effect (``0`` is two-sided, based on |z|).
    times, tstep : used for ``intervals``.
    min_cluster_size, alpha : see :func:`cluster_permutation_1d`.
    connectivity : int | None
        Connectivity order for cluster detection (``None`` -> 1 for 1-D, 2 for 2-D).

    Returns
    -------
    ClusterResult
        ``stat`` holds the observed z-map.
    """
    from scipy.ndimage import generate_binary_structure, label

    true_map = np.asarray(true_map, dtype=float)
    surrogate_maps = np.asarray(surrogate_maps, dtype=float)
    if surrogate_maps.shape[1:] != true_map.shape:
        raise ValueError("surrogate_maps must be (n_surrogates, *true_map.shape)")
    if connectivity is None:
        connectivity = 1 if true_map.ndim == 1 else int(true_map.ndim)
    structure = generate_binary_structure(true_map.ndim, connectivity)

    mean = surrogate_maps.mean(axis=0)
    sd = surrogate_maps.std(axis=0)
    sd = np.where(sd > 0, sd, np.nan)

    if tail == 0:
        z_thr = ss.norm.ppf(1 - cluster_forming / 2)
        mask_fn = lambda z: np.abs(z) > z_thr  # noqa: E731
    elif tail == 1:
        z_thr = ss.norm.ppf(1 - cluster_forming)
        mask_fn = lambda z: z > z_thr  # noqa: E731
    elif tail == -1:
        z_thr = ss.norm.ppf(1 - cluster_forming)
        mask_fn = lambda z: z < -z_thr  # noqa: E731
    else:
        raise ValueError(f"tail must be -1, 0 or 1, got {tail!r}")

    def find(z):
        labelled, n_labels = label(mask_fn(z), structure=structure)
        out = []
        for i in range(1, n_labels + 1):
            mask = labelled == i
            if min_cluster_size and mask.sum() < min_cluster_size:
                continue
            out.append((mask, _cluster_stat(z, mask, tail)))
        return out

    z_true = (true_map - mean) / sd
    observed = find(z_true)

    null_max = np.zeros(surrogate_maps.shape[0])
    for i, surrogate in enumerate(surrogate_maps):
        stats = [s for _, s in find((surrogate - mean) / sd)]
        null_max[i] = max(stats) if stats else 0.0

    clusters = [mask for mask, _ in observed]
    pvals = np.array([(np.sum(null_max >= s) + 1) / (null_max.size + 1)
                      for _, s in observed])
    sig_mask, _, _ = _select_clusters(
        clusters, pvals, alpha=alpha, min_cluster_size=0, min_cluster_duration=0.0,
        tstep=None, shape=true_map.shape)

    return ClusterResult(stat=z_true, clusters=clusters, cluster_pvals=pvals,
                         sig_mask=sig_mask, alpha=alpha, null_max=null_max,
                         times=times, tstep=_resolve_tstep(times, tstep))


# --------------------------------------------------------------------------- #
# sign flipping on a summary map
# --------------------------------------------------------------------------- #
def cluster_permutation_signflip(stat_map, *, adjacency=None, threshold: float | None = None,
                                df: int | None = None, cluster_forming: float = 0.05,
                                tail: int = 0, n_permutations: int = 1000,
                                cluster_statistic: str = "sum",
                                min_cluster_size: int = 0, alpha: float = 0.05,
                                seed=None) -> ClusterResult:
    """Cluster correction from a summary statistic map (no per-subject data).

    Sign flipping of the map entries yields the null distribution of the maximal
    cluster statistic.  Provide either an explicit positive ``threshold`` or ``df``
    together with ``cluster_forming``.

    Parameters
    ----------
    stat_map : array_like
        Observed statistic over units (1-D) or a grid.
    adjacency : sparse matrix | None
        Adjacency between units; ``None`` falls back to grid connectivity
        (:func:`scipy.ndimage.label`).
    threshold : float | None
        Cluster-forming threshold (positive magnitude).
    df : int | None
        Degrees of freedom, used with ``cluster_forming`` when ``threshold`` is None.
    cluster_statistic : {'sum', 'size'}
    tail, n_permutations, min_cluster_size, alpha, seed : see
        :func:`cluster_permutation_1d`.

    Returns
    -------
    ClusterResult
    """
    from scipy.ndimage import generate_binary_structure, label
    from scipy.sparse.csgraph import connected_components

    stat_map = np.asarray(stat_map, dtype=float)
    flat_map = stat_map.ravel()
    if threshold is None:
        if df is None:
            raise ValueError("provide either `threshold` or `df` (with cluster_forming)")
        threshold = cluster_forming_threshold(cluster_forming, df=df, tail=tail)
    threshold = abs(float(threshold))

    adj = None
    if adjacency is not None:
        adj = sp.csr_matrix(adjacency)
        if adj.shape[0] != flat_map.size:
            raise ValueError("adjacency must match the number of map entries")
    structure = generate_binary_structure(stat_map.ndim, 1 if stat_map.ndim == 1
                                          else stat_map.ndim)

    def find(values):
        values = np.asarray(values, dtype=float).ravel()
        if tail == 0:
            mask = np.abs(values) > threshold
        elif tail == 1:
            mask = values > threshold
        elif tail == -1:
            mask = values < -threshold
        else:
            raise ValueError(f"tail must be -1, 0 or 1, got {tail!r}")
        if not mask.any():
            return []
        if adj is None:
            labelled, n_labels = label(mask.reshape(stat_map.shape), structure=structure)
            labelled = labelled.ravel()
        else:
            n_labels, sub_labels = connected_components(adj[mask][:, mask], directed=False)
            labelled = np.zeros(mask.size, dtype=int)
            labelled[mask] = sub_labels + 1
        out = []
        for i in range(1, n_labels + 1):
            idx = labelled == i
            if min_cluster_size and idx.sum() < min_cluster_size:
                continue
            out.append((idx, _cluster_stat(values, idx, tail, how=cluster_statistic)))
        return out

    observed = find(stat_map)
    rng = np.random.default_rng(seed)
    null_max = np.zeros(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=flat_map.size)
        stats = [s for _, s in find(flat_map * signs)]
        null_max[i] = max(stats) if stats else 0.0

    clusters = [mask for mask, _ in observed]
    pvals = np.array([(np.sum(null_max >= s) + 1) / (n_permutations + 1)
                      for _, s in observed])
    sig_mask, _, _ = _select_clusters(
        clusters, pvals, alpha=alpha, min_cluster_size=0, min_cluster_duration=0.0,
        tstep=None, shape=stat_map.shape)

    return ClusterResult(stat=stat_map, clusters=clusters, cluster_pvals=pvals,
                         sig_mask=sig_mask, alpha=alpha, null_max=null_max)


# --------------------------------------------------------------------------- #
# post-processing helpers
# --------------------------------------------------------------------------- #
def cluster_recurrence(clusters, spatial_axis: int = -1) -> np.ndarray:
    """Per-unit recurrence across a set of clusters (for surface maps).

    Each cluster mask is reduced over all axes except ``spatial_axis`` - i.e. how
    often a unit belongs to that cluster along time - and the results are summed.

    Parameters
    ----------
    clusters : sequence of boolean masks
        Masks sharing one shape.
    spatial_axis : int
        Axis holding the units.

    Returns
    -------
    np.ndarray
        1-D map over units.
    """
    maps = []
    for mask in clusters:
        mask = np.asarray(mask, dtype=bool)
        axes = tuple(i for i in range(mask.ndim) if i != (spatial_axis % mask.ndim))
        maps.append(mask.sum(axis=axes))
    if not maps:
        return np.zeros(0)
    return np.sum(maps, axis=0)
