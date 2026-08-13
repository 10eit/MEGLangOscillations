"""Representational Dissimilarity Matrices (RDMs): Euclidean + Crossnobis.

Consolidates:
- Figure4Subj/NeuralRDM.py        -> euclidean RDM
- Figure3RSA/SrcRDM.py            -> crossnobis RDM (LedoitWolf + k-fold cross)
- Figure3RSA/TemporalRDM.py       -> crossnobis RDM *movie* via rsatoolbox
- Figure3Cont/ModelEvaluation.py  -> crossnobis windowed RDM via rsatoolbox
"""

import numpy as np
import scipy.spatial.distance as spd
from scipy.linalg import solve
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedKFold


def vec_rdm(rdm):
    """Upper-triangular vector of a square RDM (pdist form)."""
    rdm = np.asarray(rdm)
    iu = np.triu_indices(rdm.shape[-1], 1)
    return rdm[..., iu[0], iu[1]] if rdm.ndim > 2 else rdm[iu]


def rdm_to_vector(rdm):
    """Alias of vec_rdm accepting either a square matrix or list of matrices."""
    rdm = np.asarray(rdm)
    if rdm.ndim == 2:
        return vec_rdm(rdm)
    return np.array([vec_rdm(m) for m in rdm])


def add_to_rdm(rdm):
    """Fold a vector back into a symmetric matrix with zero diagonal."""
    return spd.squareform(rdm)


# --------------------------------------------------------------------------- #
# Euclidean
# --------------------------------------------------------------------------- #
def euclidean_rdm(X, zscore=False):
    """Pairwise Euclidean distance RDM.

    Parameters
    ----------
    X : np.ndarray, shape (n_items, n_features) or (n_items, n_features, n_times)
        One row per condition/subject. When 3D, an RDM is returned per timepoint
        along the last axis (shape (n_items, n_items, n_times)).
    zscore : bool
        Z-score features before computing distances.

    Returns
    -------
    rdm : np.ndarray
        (n_items, n_items) for 2D input, or (n_items, n_items, n_times) for 3D.
    """
    X = np.asarray(X, dtype=float)
    if zscore:
        X = scipy_zscore(X, axis=-1)
    if X.ndim == 2:
        return spd.squareform(spd.pdist(X, "euclidean"))
    return np.stack([spd.squareform(spd.pdist(X[..., t], "euclidean"))
                     for t in range(X.shape[-1])], axis=-1)


def scipy_zscore(X, axis=-1):
    import scipy.stats as ss
    return ss.zscore(X, axis=axis)


# --------------------------------------------------------------------------- #
# Crossnobis (manual LedoitWolf + k-fold cross, no rsatoolbox needed)
# --------------------------------------------------------------------------- #
def crossnobis_rdm(data, labels, n_folds=5, random_state=42):
    """Cross-validated crossnobis RDM using LedoitWolf shrinkage covariance.

    Parameters
    ----------
    data : np.ndarray, shape (n_trials, [n_rois], n_features)
        Per-trial activity vectors (e.g. ROI time-series). If 3D, an independent RDM
        is computed per ROI (axis 1) — matches Figure3RSA/SrcRDM.py.
    labels : np.ndarray, shape (n_trials,)
        Condition label per trial.
    n_folds : int
    random_state : int

    Returns
    -------
    rdm : np.ndarray
        (n_conditions, n_conditions) if data is 2D, else (n_rois, n_conditions, n_conditions).
    """
    data = np.asarray(data, dtype=float)
    labels = np.asarray(labels)
    conditions = np.unique(labels)
    n_cond = len(conditions)

    if data.ndim == 2:
        return _crossnobis_2d(data, labels, conditions, n_folds, random_state)

    n_rois = data.shape[1]
    out = np.zeros((n_rois, n_cond, n_cond))
    for r in range(n_rois):
        out[r] = _crossnobis_2d(data[:, r, :], labels, conditions, n_folds, random_state)
    return out


def _crossnobis_2d(roi_data, labels, conditions, n_folds, random_state):
    n_cond = len(conditions)
    _, n_features = roi_data.shape
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_means = []
    for train_idx, test_idx in skf.split(roi_data, labels):
        train, test = roi_data[train_idx], roi_data[test_idx]
        test_labels = labels[test_idx]
        cov = LedoitWolf().fit(train).covariance_
        means = []
        for cond in conditions:
            cd = test[test_labels == cond]
            means.append(np.mean(cd, axis=0) if cd.shape[0] else np.zeros(n_features))
        fold_means.append((np.array(means), cov))

    rdm = np.zeros((n_cond, n_cond))
    n_pairs = 0
    for f1 in range(n_folds):
        for f2 in range(f1 + 1, n_folds):
            means1, cov1 = fold_means[f1]
            means2, cov2 = fold_means[f2]
            cov = (cov1 + cov2) / 2.0
            for i in range(n_cond):
                for j in range(i + 1, n_cond):
                    diff1 = means1[i] - means1[j]
                    diff2 = means2[i] - means2[j]
                    dist = diff1 @ solve(cov, diff2, assume_a="pos")
                    rdm[i, j] += dist
                    rdm[j, i] += dist
                    n_pairs += 1
    rdm /= max(n_pairs, 1)
    return rdm


# --------------------------------------------------------------------------- #
# Balanced k-fold assignment (one value per trial, every condition in every fold)
# --------------------------------------------------------------------------- #
def make_balanced_kfold(labels, desired_folds=8, seed=0):
    """Assign trials to balanced folds (each condition present in each fold).

    The number of folds is automatically reduced to the minimum trial count across
    conditions if needed (mirrors Figure3RSA/TemporalRDM.py).

    Returns
    -------
    folds : np.ndarray (n_trials,) int in [0, n_folds_eff)
    n_folds_eff : int
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    n_folds_eff = int(min(desired_folds, counts.min()))
    if n_folds_eff < 2:
        raise RuntimeError(
            f"Not enough trials per condition (min={counts.min()}) for crossnobis CV.")
    folds = np.full(labels.shape, -1, dtype=int)
    for u in np.unique(labels):
        idx = np.where(labels == u)[0]
        rng.shuffle(idx)
        for fid, ids in enumerate(np.array_split(idx, n_folds_eff)):
            folds[ids] = fid
    assert np.all(folds >= 0)
    return folds, n_folds_eff


# --------------------------------------------------------------------------- #
# Crossnobis RDM *movie* via rsatoolbox (temporal RDM, noise precision from residuals)
# --------------------------------------------------------------------------- #
def crossnobis_rdm_movie(data, labels, times, ch_names=None,
                        n_folds=8, seed=42):
    """Crossnobis temporal RDM movie using rsatoolbox.

    Parameters
    ----------
    data : np.ndarray, shape (n_trials, n_channels, n_times)
    labels : np.ndarray (n_trials,)  condition per trial (will be used as descriptor).
    times : np.ndarray (n_times,)
    ch_names : list[str] | None
    n_folds : int
    seed : int

    Returns
    -------
    rdm_movie : np.ndarray, shape (n_times, n_conditions, n_conditions)
    """
    import rsatoolbox
    from rsatoolbox.data import TemporalDataset
    from rsatoolbox.data.noise import prec_from_residuals
    from rsatoolbox.rdm import calc_rdm_movie

    data = np.asarray(data, dtype=float)
    labels = np.asarray(labels)

    folds, _ = make_balanced_kfold(labels, desired_folds=n_folds, seed=seed)

    # noise precision from per-condition residuals (trial - condition mean)
    residuals = np.zeros_like(data)
    for c in np.unique(labels):
        mask = labels == c
        residuals[mask, :, :] = data[mask, :, :] - data[mask, :, :].mean(axis=0, keepdims=True)
    n_channels = data.shape[1]
    reshaped = np.swapaxes(residuals, 1, 2).reshape(-1, n_channels)
    prec = prec_from_residuals(reshaped, method="shrinkage_diag")

    obs_des = {
        "TrialCond": labels,
        "cv_fold": folds,
    }
    ch_des = {"channels": ch_names} if ch_names is not None else {"channels": np.arange(n_channels)}
    tim_des = {"time": times}

    ds = TemporalDataset(data, descriptors={"subject": 0},
                         obs_descriptors=obs_des, channel_descriptors=ch_des,
                         time_descriptors=tim_des)
    rdms = calc_rdm_movie(ds, method="crossnobis", descriptor="TrialCond",
                         cv_descriptor="cv_fold", noise=prec)
    return rdms.get_matrices()  # (n_times, n_cond, n_cond)


def crossnobis_rdm_window(data, labels, t_window, sfreq=None, ts=None, **kwargs):
    """Crossnobis RDM averaged over a time window (rsatoolbox)."""
    import rsatoolbox
    from rsatoolbox.data import Dataset
    from rsatoolbox.rdm import calc_rdm

    # window -> reduce time dimension (mean)
    raise NotImplementedError(
        "Use crossnobis_rdm_movie + index the time of interest, or implement "
        "the rsatoolbox window path directly (see Figure3Cont/ModelEvaluation.py).")