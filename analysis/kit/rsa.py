"""Representational Similarity Analysis (RSA).

* :func:`build_model_rdm` - model RDM from condition labels (categorical similarity).
* :func:`rsa_spearman` - one neural RDM vs one model RDM.
* :func:`rsa_series` - one model RDM vs a whole family of neural RDMs (subjects x time).
* :func:`rsa_model_difference` - z-scored difference between two competing models.
* :func:`rsa_compare_models` - per-subject comparison of two models (paired t-test).
* :func:`group_rsa_test` - group-level test of per-subject correlations against chance.

Searchlight RSA lives in :mod:`kit.searchlight`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, ttest_1samp, ttest_rel

from .rdm import vec_rdm
from .stats import fisher_z

__all__ = [
    "build_model_rdm",
    "rsa_spearman",
    "rsa_series",
    "rsa_model_difference",
    "rsa_compare_models",
    "group_rsa_test",
]


def _as_vector(rdm) -> np.ndarray:
    """Upper-triangular vector of a square RDM, or pass through an existing vector."""
    rdm = np.asarray(rdm, dtype=float)
    return vec_rdm(rdm) if rdm.ndim == 2 else rdm


def build_model_rdm(labels, *, within: float = 0.0, between: float = 1.0,
                    diagonal: float = 0.0) -> np.ndarray:
    """Model RDM from categorical condition labels.

    Parameters
    ----------
    labels : sequence
        One label per condition; entries with equal labels are ``within``.
    within, between : float
        (Dis)similarity assigned to same-label and different-label pairs.
    diagonal : float
        Value placed on the diagonal.

    Returns
    -------
    np.ndarray, shape (n_conditions, n_conditions)
    """
    labels = np.asarray(labels)
    n = labels.size
    same = labels[:, None] == labels[None, :]
    rdm = np.where(same, within, between).astype(float)
    np.fill_diagonal(rdm, diagonal)
    return rdm


def rsa_spearman(neural_rdm, model_rdm):
    """Spearman correlation between a neural and a model RDM (square or vector).

    Returns ``(rho, p)``.
    """
    result = spearmanr(_as_vector(neural_rdm), _as_vector(model_rdm))
    return float(result.correlation), float(result.pvalue)


def rsa_series(neural_rdms, model_rdm, *, fisher: bool = True):
    """Correlate one model RDM with a family of neural RDMs.

    Parameters
    ----------
    neural_rdms : array_like
        ``(..., n_conditions, n_conditions)``; e.g. ``(n_subjects, n_times, n_cond,
        n_cond)`` or ``(n_subjects, n_cond, n_cond)``.
    model_rdm : array_like
        ``(n_conditions, n_conditions)`` model RDM.
    fisher : bool
        Apply the Fisher r-to-z transform (recommended before averaging / testing).

    Returns
    -------
    np.ndarray
        Shape ``neural_rdms.shape[:-2]``.
    """
    rdms = np.asarray(neural_rdms, dtype=float)
    if rdms.ndim < 2 or rdms.shape[-1] != rdms.shape[-2]:
        raise ValueError("neural_rdms must end in two square dimensions")
    vectors = vec_rdm(rdms).reshape(-1, rdms.shape[-1] * (rdms.shape[-1] - 1) // 2)
    model_vec = _as_vector(model_rdm)

    model_rank = _rank(model_vec)
    model_centered = model_rank - model_rank.mean()
    model_norm = np.linalg.norm(model_centered)

    rho = np.empty(vectors.shape[0])
    for i, row in enumerate(vectors):
        rank = _rank(row)
        centered = rank - rank.mean()
        denominator = np.linalg.norm(centered) * model_norm
        rho[i] = np.nan if denominator == 0 else float(centered @ model_centered / denominator)
    rho = rho.reshape(rdms.shape[:-2])
    return fisher_z(rho) if fisher else rho


def _rank(values) -> np.ndarray:
    """Average ranks of a 1-D array (Spearman without scipy's axis constraints)."""
    from scipy.stats import rankdata

    return np.asarray(rankdata(values), dtype=float)


def rsa_model_difference(neural_rdms, model_a, model_b, *, fisher: bool = True):
    """Difference of two model correlations for each neural RDM (A minus B).

    When ``fisher=True`` the difference is taken in z-space, which is the appropriate
    input for a paired group test.
    """
    return (rsa_series(neural_rdms, model_a, fisher=fisher)
            - rsa_series(neural_rdms, model_b, fisher=fisher))


def rsa_compare_models(subject_rdms, model_rdm_a, model_rdm_b, *, fisher: bool = True,
                       alternative: str = "two-sided"):
    """Per-subject RSA against two competing model RDMs, then a paired t-test.

    Returns
    -------
    rA, rB : np.ndarray
        Per-subject correlations (z-scored when ``fisher=True``).
    t, p : float
    """
    rA = rsa_series(np.asarray(list(subject_rdms)), model_rdm_a, fisher=fisher)
    rB = rsa_series(np.asarray(list(subject_rdms)), model_rdm_b, fisher=fisher)
    test = ttest_rel(np.ravel(rA), np.ravel(rB), alternative=alternative)
    return rA, rB, float(test.statistic), float(test.pvalue)


def group_rsa_test(subject_rhos, chance: float = 0.0, *, fisher: bool = True,
                   alternative: str = "two-sided"):
    """One-sample test of per-subject RSA values against ``chance``."""
    values = np.asarray(subject_rhos, dtype=float)
    if fisher:
        values = fisher_z(values)
    result = ttest_1samp(values, chance, alternative=alternative, nan_policy="omit")
    return float(result.statistic), float(result.pvalue)
