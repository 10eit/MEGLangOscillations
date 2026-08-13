"""Decoding: sliding (diagonal) and generalizing (cross-temporal) classifiers.

Consolidates Figure6/crossword_decoding.py, Fig2DevPowerMVPA/RunTG.py,
Fig2RegPowerMVPA/diag_decoding.py, Figure4Task/decoding/*.

Convention
----------
Epoch data -> X of shape (n_trials, n_channels_or_features, n_timepoints),
labels y of shape (n_trials,).
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore


def make_logistic_clf(C=1.0, solver="liblinear", class_weight="balanced",
                      max_iter=2000, random_state=42, linear_model=False):
    """A StandardScaler -> LogisticRegression pipeline.

    Parameters
    ----------
    linear_model : bool
        If True, wrap LogisticRegression in ``mne.decoding.LinearModel`` to expose
        patterns/weights (useful for plotting sensor patterns).

    Returns
    -------
    sklearn pipeline.
    """
    lr = LogisticRegression(
        solver=solver, class_weight=class_weight, C=C,
        max_iter=max_iter, random_state=random_state,
    )
    if linear_model:
        from mne.decoding import LinearModel
        lr = LinearModel(lr)
    return make_pipeline(StandardScaler(), lr)


def smooth_time(X, sigma=3, axis=-1):
    """1-D Gaussian smoothing along the time axis."""
    if sigma is not None and sigma > 0:
        X = gaussian_filter1d(X, sigma=sigma, axis=axis)
    return X


def _prep_xy(X, y, sigma=None):
    X = np.asarray(X)
    y = np.asarray(y)
    if sigma is not None:
        X = smooth_time(X, sigma=sigma)
    return X, y


def diagonal_decoding(X, y, clf=None, cv=None, scoring="roc_auc", n_jobs=1,
                     sigma=None):
    """Sliding (per-timepoint) cross-validated decoding -> diagonal scores.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_features, n_times)
    y : np.ndarray, shape (n_trials,)
    clf : sklearn pipeline (default ``make_logistic_clf``)
    cv : cross-validator (default StratifiedKFold(5))
    scoring : str
    n_jobs : int
    sigma : float | None  Gaussian smoothing of X along time before decoding.

    Returns
    -------
    scores : np.ndarray, shape (n_folds, n_times)
    """
    X, y = _prep_xy(X, y, sigma)
    if clf is None:
        clf = make_logistic_clf()
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    slider = SlidingEstimator(clf, scoring=scoring, n_jobs=n_jobs, verbose="warning")
    return cross_val_multiscore(slider, X, y, cv=cv, n_jobs=n_jobs, verbose="warning")


def temporal_generalization(X, y, clf=None, cv=None, scoring="roc_auc",
                            n_jobs=1, sigma=None, fit=True, score_data=None,
                            score_labels=None):
    """Cross-temporal generalization decoding.

    If ``score_data`` is None: full within-set GeneralizingEstimator cross-validation
    (returns a (n_folds, n_train_times, n_test_times) matrix via cross_val_multiscore).
    If ``score_data`` is given: fit on (X, y) then score on (score_data, score_labels)
    (used for cross-condition / cross-task generalization, as in Figure6).

    Returns
    -------
    scores : np.ndarray
        (n_folds, n_train, n_test) when CV; or (n_test_times, n_train_times) when
        fitting then scoring on new data.
    """
    X, y = _prep_xy(X, y, sigma)
    if clf is None:
        clf = make_logistic_clf()
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gen = GeneralizingEstimator(clf, scoring=scoring, n_jobs=n_jobs, verbose="warning")

    if score_data is None:
        return cross_val_multiscore(gen, X, y, cv=cv, n_jobs=n_jobs, verbose="warning")

    sX, sy = _prep_xy(score_data, score_labels, sigma)
    gen.fit(X, y)
    return gen.score(sX, sy)


def temporal_generalization_permutation(X, y, clf=None, cv=None, scoring="roc_auc",
                                        n_jobs=1, sigma=None, n_permutations=100,
                                        random_state=42):
    """Null distribution for temporal generalization by permuting labels.

    Returns the permuted score distributions (n_permutations, n_train, n_test).
    """
    from sklearn.utils import shuffle
    gen_scores = temporal_generalization(X, y, clf=clf, cv=cv, scoring=scoring,
                                         n_jobs=n_jobs, sigma=sigma)
    # permutation: shuffle labels per repetition and re-run full CV
    perm_dist = []
    for seed in range(n_permutations):
        y_perm = shuffle(y, random_state=seed + random_state)
        s = temporal_generalization(X, y_perm, clf=clf, cv=cv, scoring=scoring,
                                    n_jobs=n_jobs, sigma=sigma)
        perm_dist.append(s.mean(axis=0))
    return gen_scores, np.array(perm_dist)


def diagonal_scores(tg_scores):
    """Extract the diagonal (train==test time) from a generalization matrix.

    Parameters
    ----------
    tg_scores : np.ndarray
        Either (n_subjects, n_train, n_test) or (n_train, n_test). If averaged over
        folds first, pass ``.mean(axis=0)``.
    """
    tg = np.asarray(tg_scores)
    if tg.ndim == 3:
        tg = tg.mean(axis=0)
    return np.diag(tg)