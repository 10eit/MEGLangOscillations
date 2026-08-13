"""Representational Similarity Analysis (RSA): correlations between RDMs.

Consolidates Figure3Cont/ModelEvaluation.py and Figure4Subj/CompareRDMs*.py,
Figure4Subj/SearchlightRSA.py (searchlight RSA lives in ``kit.searchlight``).
"""

import numpy as np
from scipy.stats import spearmanr, ttest_rel, ttest_1samp


def rsa_spearman(neural_rdm, model_rdm):
    """Spearman correlation between a neural RDM and a model RDM (both square or vec).

    Both inputs may be square (n,n) or upper-triangular vectors. Returns rho, p.
    """
    from .rdm import vec_rdm
    nv = vec_rdm(neural_rdm) if np.ndim(neural_rdm) == 2 else np.asarray(neural_rdm)
    mv = vec_rdm(model_rdm) if np.ndim(model_rdm) == 2 else np.asarray(model_rdm)
    r = spearmanr(nv, mv)
    return r.correlation, r.pvalue


def rsa_compare_models(subject_rdms, model_rdm_a, model_rdm_b):
    """Per-subject RSA against two competing model RDMs, then paired t-test.

    Parameters
    ----------
    subject_rdms : np.ndarray, shape (n_subjects,) of square RDMs (pass as list/array).
    model_rdm_a, model_rdm_b : np.ndarray  square model RDMs.

    Returns
    -------
    rA, rB : np.ndarray (n_subjects,) Spearman rho per subject for each model.
    t, p : paired t-test (model A vs model B).
    """
    rA, rB = [], []
    for rdm in subject_rdms:
        ca, _ = rsa_spearman(rdm, model_rdm_a)
        cb, _ = rsa_spearman(rdm, model_rdm_b)
        rA.append(ca)
        rB.append(cb)
    rA, rB = np.array(rA), np.array(rB)
    t, p = ttest_rel(rA, rB)
    return rA, rB, t, p


def group_rsa_test(subject_rhodos, chance=0.0):
    """One-sample t-test of per-subject RSA rho values against chance (default 0)."""
    rhos = np.asarray(subject_rhodos)
    return ttest_1samp(rhos, chance, axis=0)