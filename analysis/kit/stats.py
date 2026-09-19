"""Group-level statistics: tests against a reference, effect sizes, HDI, FDR, resampling.

Data layout
-----------
``axis=0`` is always the observation axis (usually subjects), so ``data`` of shape
``(n_subjects, n_times)`` gives one test per time point and the returned statistic /
p-value arrays have shape ``(n_times,)``.

Design rule
-----------
Nothing that belongs to a study design is baked in: chance level, alternative,
test type, FDR alpha, HDI probability, resample / permutation counts and the random
seed are all explicit arguments with neutral defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats as ss

__all__ = [
    "GroupTestResult",
    "one_sample_against_chance",
    "test_against",
    "paired_test",
    "cohens_d",
    "mean_sem",
    "hdi",
    "posterior_summary",
    "bootstrap_ci",
    "fisher_z",
    "fisher_z_inv",
    "fdr_correct",
    "fdr_mask",
    "permutation_p",
    "cluster_significance_mask",
]


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class GroupTestResult:
    """Outcome of a group-level test.

    Attributes
    ----------
    statistic, pvalue : float | np.ndarray
        Shaped like ``data.shape[1:]``.
    effect_size : float | np.ndarray
        Cohen's d (``mean - reference`` over SD) for one-sample and paired tests.
    n : int
        Number of observations along the tested axis.
    test, alternative : str
        Echo of the requested options.
    """

    statistic: object
    pvalue: object
    effect_size: object
    n: int
    test: str
    alternative: str

    @property
    def significant(self) -> np.ndarray:
        """Boolean mask for ``pvalue < 0.05`` (uncorrected)."""
        return np.asarray(self.pvalue) < 0.05


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _axis_map(func: Callable, data: np.ndarray, axis: int = 0):
    """Apply a 1-D function along ``axis``; scalars or tuples are re-assembled."""
    a = np.moveaxis(np.asarray(data, dtype=float), axis, 0)
    flat = a.reshape(a.shape[0], -1)
    out = np.array([func(flat[:, i]) for i in range(flat.shape[1])], dtype=float)
    if out.ndim == 1:
        return out.reshape(a.shape[1:])
    return [out[:, k].reshape(a.shape[1:]) for k in range(out.shape[-1])]


def _wilcoxon(values, alternative: str):
    result = ss.wilcoxon(values, alternative=alternative)
    return result.statistic, result.pvalue


def _signflip_null(values: np.ndarray, n_permutations: int, seed) -> np.ndarray:
    """Null distribution of the mean under random sign flips (shape (n_perm, ...))."""
    rng = np.random.default_rng(seed)
    a = np.asarray(values, dtype=float)
    n = a.shape[0]
    flat = a.reshape(n, -1)
    null = np.empty((n_permutations, flat.shape[1]), dtype=float)
    for i in range(n_permutations):
        null[i] = (flat * rng.choice([-1.0, 1.0], size=(n, 1))).mean(axis=0)
    return null.reshape((n_permutations,) + a.shape[1:])


def _p_from_null(observed, null, alternative: str) -> np.ndarray:
    observed = np.asarray(observed, dtype=float)
    n = null.shape[0]
    if alternative == "greater":
        count = (null >= observed).sum(axis=0)
    elif alternative == "less":
        count = (null <= observed).sum(axis=0)
    elif alternative == "two-sided":
        count = (np.abs(null) >= np.abs(observed)).sum(axis=0)
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return (count + 1) / (n + 1)


# --------------------------------------------------------------------------- #
# tests against a reference value
# --------------------------------------------------------------------------- #
def cohens_d(data, reference: float = 0.0, axis: int = 0, ddof: int = 1):
    """Cohen's d of ``data`` against ``reference`` (one-sample / paired effect size)."""
    data = np.asarray(data, dtype=float)
    mean = np.nanmean(data, axis=axis)
    sd = np.nanstd(data, axis=axis, ddof=ddof)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (mean - reference) / sd


def test_against(data, chance: float = 0.0, *, test: str = "ttest",
                 alternative: str = "two-sided", axis: int = 0,
                 n_permutations: int = 10_000, seed=None) -> GroupTestResult:
    """One-sample test of ``data`` against ``chance`` along ``axis``.

    Parameters
    ----------
    data : array_like
        ``(n_observations, ...)``; observations are taken along ``axis``.
    chance : float
        Reference value (0.5 for ROC-AUC decoding, 0 for mean-difference / R² / rho).
    test : {'ttest', 'wilcoxon', 'signflip'}
        ``signflip`` is an exact-ish permutation test on the mean (no distributional
        assumption, robust for small n).
    alternative : {'two-sided', 'greater', 'less'}
    axis : int
        Observation axis.
    n_permutations, seed : only used by ``test='signflip'``.

    Returns
    -------
    GroupTestResult
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 0:
        raise ValueError("data must have at least one dimension")

    if test == "ttest":
        res = ss.ttest_1samp(data, chance, axis=axis, alternative=alternative,
                             nan_policy="omit")
        statistic, pvalue = np.asarray(res.statistic), np.asarray(res.pvalue)
    elif test == "wilcoxon":
        statistic, pvalue = _axis_map(
            lambda v: _wilcoxon(v, alternative), data - chance, axis=axis)
    elif test == "signflip":
        observed = np.nanmean(data, axis=axis) - chance
        null = _signflip_null(np.moveaxis(data - chance, axis, 0), n_permutations, seed)
        statistic = observed
        pvalue = _p_from_null(observed, null, alternative)
    else:
        raise ValueError(f"unknown test {test!r}")

    n = np.asarray(data).shape[axis]
    return GroupTestResult(statistic=statistic, pvalue=pvalue,
                           effect_size=cohens_d(data, chance, axis=axis),
                           n=int(n), test=test, alternative=alternative)


def paired_test(a, b, *, test: str = "ttest", alternative: str = "two-sided",
                axis: int = 0, n_permutations: int = 10_000,
                seed=None) -> GroupTestResult:
    """Paired test of ``a`` vs ``b`` (equivalent to one-sample test of ``a - b``)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return test_against(diff, 0.0, test=test, alternative=alternative, axis=axis,
                        n_permutations=n_permutations, seed=seed)


def one_sample_against_chance(data, chance: float = 0.5, axis: int = 0):
    """One-sample t-test against ``chance``; returns ``(t, p)`` (kept for brevity)."""
    res = ss.ttest_1samp(np.asarray(data, dtype=float), chance, axis=axis,
                         nan_policy="omit")
    return res.statistic, res.pvalue


def mean_sem(data, axis: int = 0, ddof: int = 1):
    """Mean and standard error of the mean along ``axis``."""
    data = np.asarray(data, dtype=float)
    n = np.sum(~np.isnan(data), axis=axis)
    return np.nanmean(data, axis=axis), np.nanstd(data, axis=axis, ddof=ddof) / np.sqrt(n)


# --------------------------------------------------------------------------- #
# Bayesian posterior summaries
# --------------------------------------------------------------------------- #
def _hdi_1d(samples: np.ndarray, prob: float):
    s = np.sort(np.asarray(samples, dtype=float).ravel())
    n = s.size
    if n == 0:
        return np.nan, np.nan
    k = max(int(np.floor(prob * n)), 1)
    if k >= n:
        return float(s[0]), float(s[-1])
    widths = s[k:] - s[: n - k]
    i = int(np.argmin(widths))
    return float(s[i]), float(s[i + k])


def hdi(samples, prob: float = 0.95, axis: int = -1):
    """Highest-density interval of ``samples`` containing ``prob`` of the mass.

    Returns an array with a trailing axis of length 2 (``[low, high]``); for 1-D
    input the shape is simply ``(2,)``.  Dependency-free (no arviz needed).
    """
    a = np.moveaxis(np.asarray(samples, dtype=float), axis, -1)
    flat = a.reshape(-1, a.shape[-1])
    out = np.array([_hdi_1d(row, prob) for row in flat])
    return out.reshape(a.shape[:-1] + (2,))


def posterior_summary(samples, *, prob: float = 0.95, reference: float = 0.0,
                      axis: int = -1, names: Sequence[str] | None = None):
    """Summarise posterior samples as a DataFrame.

    Parameters
    ----------
    samples : dict[str, array_like] | array_like
        Either a mapping ``name -> samples`` or a 2-D array ``(n_samples, n_params)``
        together with ``names``.
    prob : float
        HDI probability.
    reference : float
        Value the interval / sign probabilities are evaluated against (usually 0).
    axis : int
        Sample axis.

    Returns
    -------
    pandas.DataFrame with columns
        ``mean, sd, hdi_low, hdi_high, p_above, p_below, excludes_reference, n``.
        Requires pandas.
    """
    import pandas as pd

    if isinstance(samples, dict):
        names = list(samples)
        matrix = np.column_stack([np.asarray(v, dtype=float).ravel() for v in samples.values()])
    else:
        matrix = np.asarray(samples, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
            axis = 0
        if names is None:
            names = [f"param_{i}" for i in range(matrix.shape[1])]
        if axis != 0:
            matrix = np.moveaxis(matrix, axis, 0).reshape(matrix.shape[axis], -1)

    interval = hdi(matrix, prob=prob, axis=0)
    rows = []
    for i, name in enumerate(names):
        col = matrix[:, i]
        lo, hi = interval[i]
        rows.append({
            "name": name,
            "mean": float(np.nanmean(col)),
            "sd": float(np.nanstd(col, ddof=1)),
            "hdi_low": lo,
            "hdi_high": hi,
            "p_above": float(np.mean(col > reference)),
            "p_below": float(np.mean(col < reference)),
            "excludes_reference": bool(lo > reference or hi < reference),
            "n": int(col.size),
        })
    return pd.DataFrame(rows).set_index("name")


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #
def bootstrap_ci(data, statistic: Callable = np.mean, *, n_boot: int = 1000,
                 ci: float = 0.95, axis: int = 0, seed=None):
    """Percentile bootstrap confidence interval of ``statistic(data)``.

    Returns ``(low, high)`` shaped like ``data.shape[1:]``.
    """
    rng = np.random.default_rng(seed)
    data = np.moveaxis(np.asarray(data, dtype=float), axis, 0)
    n = data.shape[0]
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(statistic(data[idx], axis=0))
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], axis=0)
    return lo, hi


def permutation_p(true_score, null_distribution, alternative: str = "greater"):
    """Permutation p-value of ``true_score`` against ``null_distribution``."""
    null = np.asarray(null_distribution, dtype=float)
    n = null.size
    if alternative == "greater":
        count = np.sum(null >= true_score)
    elif alternative == "less":
        count = np.sum(null <= true_score)
    elif alternative == "two-sided":
        count = np.sum(np.abs(null - null.mean()) >= abs(true_score - null.mean()))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return (count + 1) / (n + 1)


# --------------------------------------------------------------------------- #
# multiple comparison correction
# --------------------------------------------------------------------------- #
def fdr_correct(p_values, alpha: float = 0.05, method: str = "bh"):
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : array_like
        Arbitrary shape; NaNs are propagated (they should be excluded by the caller,
        see :func:`fdr_mask`).
    alpha : float
        Significance level at which ``rejected`` is computed.
    method : {'bh'}
        Only the Benjamini-Hochberg step-up procedure is implemented (dependency-free).

    Returns
    -------
    rejected, p_corrected : np.ndarray
        Same shape as ``p_values``.
    """
    if method not in ("bh", "fdr_bh", "indep"):
        raise ValueError(f"unsupported method {method!r}; only Benjamini-Hochberg")
    p = np.asarray(p_values, dtype=float)
    shape = p.shape
    flat = p.ravel()
    n = flat.size
    p_fdr = np.full(n, np.nan)
    valid = ~np.isnan(flat)
    if valid.any():
        values = flat[valid]
        order = np.argsort(values, kind="stable")
        ranked = values[order] * values.size / np.arange(1, values.size + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        corrected = np.empty(values.size)
        corrected[order] = np.clip(ranked, 0.0, 1.0)
        p_fdr[valid] = corrected
    rejected = np.zeros(n, dtype=bool)
    rejected[valid] = p_fdr[valid] < alpha
    return rejected.reshape(shape), p_fdr.reshape(shape)


def fdr_mask(values, p_values, *, alpha: float = 0.05, method: str = "bh",
             fill: float = 0.0, nan_to: float = 0.0):
    """Return ``values`` with non-significant (and NaN) entries replaced by ``fill``.

    NaN p-values are kept as ``nan_to`` without entering the correction, so masked
    vertices (e.g. medial wall) do not inflate the number of tests.
    """
    values = np.asarray(values, dtype=float)
    p_values = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p_values)
    out = np.full_like(values, nan_to, dtype=float)
    if valid.any():
        rejected, _ = fdr_correct(p_values[valid], alpha=alpha, method=method)
        mask = np.zeros_like(valid)
        mask[valid] = rejected
        out[mask] = values[mask]
    return out


def cluster_significance_mask(tg_scores, chance: float = 0.5, alpha: float = 0.05):
    """FDR-corrected significance mask of a ``(n_subjects, n_train, n_test)`` matrix."""
    _, p = one_sample_against_chance(tg_scores, chance=chance, axis=0)
    rejected, _ = fdr_correct(p, alpha=alpha)
    return rejected


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def fisher_z(r, eps: float = 1e-6):
    """Fisher r-to-z transform (safe at ±1)."""
    r = np.asarray(r, dtype=float)
    return np.arctanh(np.clip(r, -1 + eps, 1 - eps))


def fisher_z_inv(z):
    """Inverse Fisher transform (z-to-r)."""
    return np.tanh(np.asarray(z, dtype=float))
