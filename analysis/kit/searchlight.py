"""Vertex searchlight decoding and searchlight RSA.

Consolidates the (duplicated) searchlight toolkit that lives in tools.py across the
Fig2DevPowerMVPA, Figure3SEM, Figure3STR, Figure4Task, Figure6 folders.
"""

import warnings
import numpy as np
from scipy.sparse import csr_matrix, issparse
import mne
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from joblib import Parallel, delayed


def source_patches(src, k, d=None, exclude_medial=None):
    """Return vertices within graph-distance ``d`` of vertex ``k`` (a searchlight ball).

    Parameters
    ----------
    src : mne.SourceSpaces or scipy.sparse.csr_matrix
        Source space or precomputed adjacency.
    k : int
        Seed vertex index.
    d : int | None
        Graph distance (number of edges). If None, a sensible default is chosen from
        the source-space resolution:
        ico=7 -> d=9, ico=6 -> d=4, ico=5 -> d=2, ico=4 -> d=1 (~6mm radius).
    exclude_medial : array-like | None
        Vertex indices to drop from the ball (e.g. medial wall).

    Returns
    -------
    np.ndarray of vertex indices in the ball.
    """
    if isinstance(src, mne.SourceSpaces):
        adjacency, _ = mne.channels.find_ch_adjacency(src, ch_type="meg")
    elif issparse(src):
        adjacency = src
    else:
        raise ValueError("src must be mne.SourceSpaces or a scipy.sparse matrix")

    if not (0 <= k < adjacency.shape[0]):
        raise ValueError(f"k={k} out of bounds for adjacency of size {adjacency.shape[0]}")

    if d is None:
        if isinstance(src, mne.SourceSpaces):
            n_vertices = sum(len(h["vertno"]) for h in src)
            if n_vertices > 100000:
                d = 9
            elif n_vertices > 40000:
                d = 4
            elif n_vertices > 20000:
                d = 2
            else:
                d = 1
                warnings.warn("Low ico-sampling (ico<4): consider univariate analysis.")
        else:
            d = 1
            warnings.warn("No SourceSpaces provided; using default d=1.")

    if not isinstance(d, int) or d < 1:
        raise ValueError("d must be a positive integer")

    current = csr_matrix(([1], ([0], [k])), shape=(1, adjacency.shape[0]))
    visited = set(current.indices.tolist())
    for _ in range(d):
        current = current @ adjacency
        visited.update(current.indices.tolist())
    ball = np.array(list(visited))

    if exclude_medial is not None:
        exclude_medial = np.asarray(exclude_medial)
        if not np.all(np.isin(exclude_medial, np.arange(adjacency.shape[0]))):
            raise ValueError("exclude_medial contains invalid vertex indices")
        ball = np.setdiff1d(ball, exclude_medial)
    return ball


def _decode_vertex(idx, src, data, label, clf, cv, scoring, d, exclude_medial):
    if exclude_medial is not None and idx in exclude_medial:
        return np.nan
    neighbours = source_patches(src, idx, d=d, exclude_medial=exclude_medial)
    neighbours = neighbours[np.isin(neighbours, np.arange(data.shape[1]))]
    if len(neighbours) == 0:
        return np.nan
    X = data[:, neighbours, :].reshape(data.shape[0], -1)
    try:
        score = cross_val_score(clf, X, label, cv=cv, scoring=scoring, n_jobs=1)
        return float(np.mean(score))
    except ValueError as e:
        print(f"Warning: decoding failed at vertex {idx}: {e}")
        return np.nan


def searchlight_decoding(src, data, label, clf, scoring="roc_auc", d=None,
                         exclude_medial=None, cv=None, n_jobs=1):
    """Decoding at every vertex using its searchlight ball as features.

    Parameters
    ----------
    src : mne.SourceSpaces or sparse adjacency
    data : np.ndarray, shape (n_trials, n_vertices, n_timepoints)
    label : np.ndarray, shape (n_trials,)
    clf : sklearn classifier
    scoring : str | callable
    d : int | None  searchlight radius (edges).
    exclude_medial : array-like | None
    cv : cross-validator (default StratifiedKFold(5))
    n_jobs : int

    Returns
    -------
    scores : np.ndarray, shape (n_vertices,)
    """
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError("data must be (n_trials, n_vertices, n_timepoints)")
    if label.shape[0] != data.shape[0]:
        raise ValueError("label length must match n_trials")

    if cv is None:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n_vertices = data.shape[1]
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_decode_vertex)(i, src, data, label, clf, cv, scoring, d, exclude_medial)
        for i in range(n_vertices)
    )
    return np.array(scores, dtype=float)


# --------------------------------------------------------------------------- #
# Searchlight RSA
# --------------------------------------------------------------------------- #
def _vertex_rsa_rdm(idx, src, source_data, d, exclude_medial):
    """Compute the neural RDM vector for the searchlight around ``idx``."""
    if exclude_medial is not None and idx in exclude_medial:
        return None, None
    neighbours = source_patches(src, idx, d=d, exclude_medial=exclude_medial)
    if len(neighbours) == 0:
        return None, None
    # source_data: (n_subjects, n_vertices, n_timepoints) OR (n_trials, n_vertices, ...)
    smooth = source_data[:, neighbours, :].mean(axis=1)
    return smooth


def searchlight_rsa(bhv_rdm_vec, source_power, src, d=1, exclude_medial=None,
                   n_jobs=1, metric="euclidean", zscore=True):
    """Searchlight RSA: correlate each vertex-ball's neural RDM with a model RDM vector.

    Parameters
    ----------
    bhv_rdm_vec : np.ndarray, shape (n_pairs,)
        Upper-triangular vector of the model / behavioural RDM (pdist form).
    source_power : np.ndarray, shape (n_subjects, n_vertices, n_timepoints)
        Source-level (smoothed) power per subject × vertex × time.
    src : mne.SourceSpaces or sparse adjacency
    d : int  searchlight radius.
    exclude_medial : array-like | None
    metric : str  distance metric for pdist (default 'euclidean').
    zscore : bool  z-score features before computing distances.

    Returns
    -------
    rho_values, p_values : np.ndarray (n_vertices,)  (nan on excluded vertices)
    """
    import scipy.stats as ss
    import scipy.spatial.distance as spd

    n_vertices = source_power.shape[1]
    rho = np.full(n_vertices, np.nan)
    pv = np.full(n_vertices, np.nan)

    def _one(idx):
        smooth = _vertex_rsa_rdm(idx, src, source_power, d, exclude_medial)
        if smooth is None:
            return np.nan, np.nan
        if zscore:
            smooth = ss.zscore(smooth, axis=-1)
        neural = spd.pdist(smooth, metric)
        r = ss.spearmanr(bhv_rdm_vec, neural)
        return r.correlation, r.pvalue

    out = Parallel(n_jobs=n_jobs)(delayed(_one)(i) for i in range(n_vertices))
    for i, (r, p) in enumerate(out):
        rho[i], pv[i] = r, p
    return rho, pv