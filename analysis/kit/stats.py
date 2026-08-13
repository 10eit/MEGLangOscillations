"""Group-level statistics: one-sample tests against chance, FDR correction, permutations.

Consolidates Figure6/statsTG.py and Figure3Cont/Stats.py.
"""

import numpy as np
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection


def one_sample_against_chance(data, chance=0.5, axis=0):
    """One-sample t-test of ``data`` against a chance level, along ``axis``.

    Parameters
    ----------
    data : np.ndarray  e.g. (n_subjects, n_times, n_times) or (n_subjects, n_times).
    chance : float  chance level (0.5 for AUC).
    axis : int  subjects axis.

    Returns
    -------
    t : np.ndarray
    p : np.ndarray (uncorrected)
    """
    return ttest_1samp(data, chance, axis=axis)


def fdr_correct(p_values, alpha=0.001, method="indep"):
    """Benjamini-Hochberg FDR correction.

    Returns (rejected, p_corrected), both shaped like ``p_values``.
    """
    p_flat = np.asarray(p_values).flatten()
    rejected, p_fdr = fdrcorrection(p_flat, alpha=alpha, method=method)
    shape = np.asarray(p_values).shape
    return rejected.reshape(shape), p_fdr.reshape(shape)


def cluster_significance_mask(tg_scores, chance=0.5, alpha=0.001):
    """FDR-corrected significance mask for a (n_subjects, n_train, n_test) matrix.

    Parameters
    ----------
    tg_scores : np.ndarray, shape (n_subjects, n_train, n_test)
    chance : float

    Returns
    -------
    sig_mask : bool array (n_train, n_test).
    """
    _, p = one_sample_against_chance(tg_scores, chance=chance, axis=0)
    rejected, _ = fdr_correct(p, alpha=alpha)
    return rejected


def permutation_p(true_score, null_distribution, alternative="greater"):
    """Permutation p-value of a true score against a null distribution.

    Parameters
    ----------
    true_score : float
    null_distribution : np.ndarray
    alternative : 'greater' | 'less' | 'two-sided'
    """
    null = np.asarray(null_distribution)
    n = len(null)
    if alternative == "greater":
        count = np.sum(null >= true_score)
    elif alternative == "less":
        count = np.sum(null <= true_score)
    else:
        count = np.sum(np.abs(null - null.mean()) >= abs(true_score - null.mean()))
    return (count + 1) / (n + 1)