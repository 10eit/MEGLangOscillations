"""Model fitting on RDMs and on time courses.

Two independent tools live here:

* :func:`fit_latent_rdm` - "cognitive-effort style" representational modelling.
  A latent per-condition variable ``y`` defines a model RDM (pairwise squared
  differences); the vectorised model RDM is predicted from one or several neural
  RDMs by ridge regression and ``y`` is optimised (multi-start L-BFGS-B) so that the
  prediction error is minimal.  ``R²`` quantifies how well the neural geometry is
  explained.
* :func:`fit_gaussian_peaks` - Gaussian (1..n peaks) fitting of a time course, e.g.
  to split a decoding time series into stages via the FWHM of each peak.

Both take plain arrays and return small dataclasses; no analysis-specific constants
are baked in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit, minimize

__all__ = [
    "vec_upper",
    "square_from_vec",
    "latent_rdm",
    "ridge_regression",
    "LatentRDMFit",
    "fit_latent_rdm",
    "fit_latent_rdm_subjects",
    "gaussian",
    "gaussian_sum",
    "fwhm",
    "Peak",
    "PeakFit",
    "fit_gaussian_peaks",
]

_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ~2.3548


# --------------------------------------------------------------------------- #
# RDM helpers
# --------------------------------------------------------------------------- #
def vec_upper(matrix, k: int = 1) -> np.ndarray:
    """Upper-triangular vector (above the ``k``-th diagonal) of a square matrix."""
    matrix = np.asarray(matrix)
    idx = np.triu_indices(matrix.shape[-1], k)
    return matrix[..., idx[0], idx[1]]


def square_from_vec(vector, n: int | None = None) -> np.ndarray:
    """Inverse of :func:`vec_upper`: symmetric matrix with a zero diagonal."""
    vector = np.asarray(vector, dtype=float)
    if n is None:
        n = int(round((1 + np.sqrt(1 + 8 * vector.size)) / 2))
    out = np.zeros((n, n))
    idx = np.triu_indices(n, 1)
    out[idx] = vector
    return out + out.T


def latent_rdm(y) -> np.ndarray:
    """Model RDM from a latent per-condition variable: pairwise squared differences."""
    y = np.asarray(y, dtype=float)
    diff = np.subtract.outer(y, y)
    return diff ** 2


def ridge_regression(X, y, alpha: float = 1e-3, fit_intercept: bool = False):
    """Closed-form ridge regression.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
    y : array_like, shape (n_samples,)
    alpha : float
        L2 penalty (non-negative).  ``0`` gives ordinary least squares.
    fit_intercept : bool
        Centring the data is preferred over estimating an intercept.

    Returns
    -------
    weights : np.ndarray, shape (n_features,)
    predicted : np.ndarray, shape (n_samples,)
    r2 : float
        Coefficient of determination of the fitted values.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_features)")
    if fit_intercept:
        X = X - X.mean(axis=0, keepdims=True)
        y = y - y.mean()
    n_features = X.shape[1]
    weights = np.linalg.solve(X.T @ X + alpha * np.eye(n_features), X.T @ y)
    predicted = X @ weights
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return weights, predicted, r2


# --------------------------------------------------------------------------- #
# latent representational model
# --------------------------------------------------------------------------- #
@dataclass
class LatentRDMFit:
    """Result of :func:`fit_latent_rdm`.

    Attributes
    ----------
    latent : np.ndarray
        Fitted per-condition values (centred when ``center=True``).
    weights : np.ndarray
        Ridge weights on the neural RDMs.
    target : np.ndarray
        Vectorised model RDM implied by ``latent``.
    predicted : np.ndarray
        Ridge prediction of ``target`` from the neural RDMs.
    r2, mse : float
        Explained variance and mean squared error of that prediction.
    neural_rdms : list of np.ndarray
        The neural RDMs that were used as predictors.
    """

    latent: np.ndarray
    weights: np.ndarray
    target: np.ndarray
    predicted: np.ndarray
    r2: float
    mse: float
    neural_rdms: list = field(default_factory=list)

    @property
    def model_rdm(self) -> np.ndarray:
        """Model RDM as a square matrix."""
        return latent_rdm(self.latent)

    def contrast(self, reference: int = 0) -> np.ndarray:
        """``latent - latent[reference]``, a scale-free condition profile."""
        return self.latent - self.latent[reference]


def fit_latent_rdm(neural_rdms, *, n_restarts: int = 64, ridge: float = 1e-3,
                   maxiter: int = 300, seed=None, x0=None, normalize: bool = True,
                   return_all: bool = False) -> LatentRDMFit:
    """Fit a latent condition variable explaining one or more neural RDMs.

    Parameters
    ----------
    neural_rdms : sequence of array_like
        One square RDM per predictor (all ``(n_conditions, n_conditions)``), e.g.
        ``[alpha_rdm, beta_rdm]``.
    n_restarts : int
        Number of random starting points for the latent vector (multi-start L-BFGS-B).
    ridge : float
        L2 penalty of the closed-form ridge regression that maps the neural RDMs onto
        the model RDM.
    maxiter : int
        Iteration cap per L-BFGS-B run.
    seed : int | None
        Seed of the random restarts (reproducibility).
    x0 : array_like | None
        Additional deterministic starting point.
    normalize : bool
        Centre the latent and fix its L2 norm (recommended).  The model RDM is
        invariant to a constant shift of ``y`` and scales as ``y**2``; without a
        constraint the optimisation can shrink ``y`` towards zero and report a
        meaningless fit.  With ``normalize=True`` the latent has a fixed scale, which
        also makes subjects comparable.  Set to ``False`` for the unconstrained
        (scale-degenerate) objective.
    return_all : bool
        Also return the objective value of every restart.

    Returns
    -------
    LatentRDMFit
    """
    neural_rdms = [np.asarray(r, dtype=float) for r in neural_rdms]
    if not neural_rdms:
        raise ValueError("at least one neural RDM is required")
    n_cond = neural_rdms[0].shape[0]
    if any(r.shape != (n_cond, n_cond) for r in neural_rdms):
        raise ValueError("all neural RDMs must share the same square shape")

    X = np.stack([vec_upper(r) for r in neural_rdms], axis=1)  # (n_pairs, n_predictors)
    scale = np.sqrt(n_cond)

    def to_latent(parameters):
        parameters = np.asarray(parameters, dtype=float)
        if not normalize:
            return parameters
        centered = parameters - parameters.mean()
        norm = np.linalg.norm(centered)
        if norm < 1e-12:
            return centered
        return centered / norm * scale

    def objective(parameters):
        target = vec_upper(latent_rdm(to_latent(parameters)))
        _, predicted, _ = ridge_regression(X, target, alpha=ridge)
        return float(((target - predicted) ** 2).mean())

    rng = np.random.default_rng(seed)
    best_x, best_fun, history = None, np.inf, []
    starts = [np.asarray(x0, dtype=float)] if x0 is not None else []
    starts += [rng.standard_normal(n_cond) for _ in range(n_restarts)]
    for parameters in starts:
        res = minimize(objective, parameters, method="L-BFGS-B",
                       options={"maxiter": maxiter})
        history.append(float(res.fun))
        if res.fun < best_fun:
            best_x, best_fun = res.x, float(res.fun)

    latent = to_latent(best_x)
    target = vec_upper(latent_rdm(latent))
    weights, predicted, r2 = ridge_regression(X, target, alpha=ridge)

    fit = LatentRDMFit(latent=latent, weights=weights, target=target,
                       predicted=predicted, r2=float(r2), mse=best_fun,
                       neural_rdms=neural_rdms)
    return (fit, np.asarray(history)) if return_all else fit


def fit_latent_rdm_subjects(subject_rdms, *, n_restarts: int = 64, ridge: float = 1e-3,
                            maxiter: int = 300, seed=None, normalize: bool = True,
                            n_jobs: int = 1):
    """Apply :func:`fit_latent_rdm` to a group of subjects.

    Parameters
    ----------
    subject_rdms : array_like
        ``(n_subjects, n_predictors, n_conditions, n_conditions)`` or a sequence of
        ``n_subjects`` sequences of square RDMs.
    other parameters : see :func:`fit_latent_rdm`.
    n_jobs : int
        Parallel subjects via joblib.

    Returns
    -------
    pandas.DataFrame
        One row per subject with ``r2``, ``mse``, ``latent_*`` and ``weight_*``.
        Requires pandas.
    """
    import pandas as pd

    subject_rdms = list(subject_rdms)
    n_subjects = len(subject_rdms)

    def _one(index):
        rdms = subject_rdms[index]
        return fit_latent_rdm(rdms, n_restarts=n_restarts, ridge=ridge, maxiter=maxiter,
                              seed=None if seed is None else seed + index,
                              normalize=normalize)

    if n_jobs and n_jobs > 1:
        from joblib import Parallel, delayed
        fits = Parallel(n_jobs=n_jobs)(delayed(_one)(i) for i in range(n_subjects))
    else:
        fits = [_one(i) for i in range(n_subjects)]

    rows = []
    for index, fit in enumerate(fits):
        row = {"subject": index, "r2": fit.r2, "mse": fit.mse}
        row.update({f"latent_{i}": v for i, v in enumerate(fit.latent)})
        row.update({f"weight_{i}": v for i, v in enumerate(np.atleast_1d(fit.weights))})
        rows.append(row)
    return pd.DataFrame(rows).set_index("subject")


# --------------------------------------------------------------------------- #
# Gaussian peak fitting
# --------------------------------------------------------------------------- #
def gaussian(t, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    """Unnormalised Gaussian."""
    return amplitude * np.exp(-((np.asarray(t, dtype=float) - mu) ** 2) / (2 * sigma ** 2))


def gaussian_sum(t, *params) -> np.ndarray:
    """Sum of ``n`` Gaussians plus an offset.

    ``params = (offset, amplitude_1, mu_1, sigma_1, ..., amplitude_n, mu_n, sigma_n)``.
    """
    params = np.asarray(params, dtype=float)
    offset = params[0]
    out = np.full_like(np.asarray(t, dtype=float), offset, dtype=float)
    for i in range(1, params.size, 3):
        out += gaussian(t, params[i], params[i + 1], params[i + 2])
    return out


def fwhm(sigma: float) -> float:
    """Full width at half maximum of a Gaussian with standard deviation ``sigma``."""
    return _FWHM_FACTOR * float(sigma)


@dataclass
class Peak:
    """A single fitted Gaussian peak."""

    amplitude: float
    mu: float
    sigma: float

    @property
    def fwhm(self) -> float:
        return fwhm(self.sigma)

    @property
    def interval(self) -> tuple:
        """``(start, end)`` of the FWHM interval."""
        half = self.fwhm / 2
        return (self.mu - half, self.mu + half)


@dataclass
class PeakFit:
    """Result of :func:`fit_gaussian_peaks`."""

    peaks: list
    offset: float
    curve: np.ndarray
    r2: float
    params: np.ndarray

    @property
    def peak_times(self) -> np.ndarray:
        return np.array([p.mu for p in self.peaks])

    @property
    def intervals(self) -> list:
        return [p.interval for p in self.peaks]

    @property
    def fwhms(self) -> np.ndarray:
        return np.array([p.fwhm for p in self.peaks])


def fit_gaussian_peaks(t, y, *, n_peaks: int = 1, p0=None,
                       bounds=None, maxfev: int = 20000) -> PeakFit:
    """Fit a sum of ``n_peaks`` Gaussians (plus offset) to a time course.

    Parameters
    ----------
    t, y : array_like
        Sampling axis and values (e.g. time and a decoding time series).
    n_peaks : int
        Number of Gaussians.  ``2`` splits a sustained response into two stages.
    p0, bounds : optional
        Explicit initial parameters / bounds in the ``gaussian_sum`` layout.
    maxfev : int
        Iteration cap.

    Returns
    -------
    PeakFit
        ``intervals`` holds the FWHM intervals, i.e. the stage periods.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.shape != y.shape:
        raise ValueError("t and y must have the same shape")
    span = float(t.max() - t.min())
    offset0 = float(np.min(y))
    amp0 = float(np.max(y) - np.min(y)) / max(n_peaks, 1)
    centers = np.linspace(t.min() + span / (2 * n_peaks),
                          t.max() - span / (2 * n_peaks), n_peaks)
    sigma0 = span / (4 * max(n_peaks, 1))

    if p0 is None:
        p0 = [offset0]
        for c in centers:
            p0 += [amp0, float(c), sigma0]
    if bounds is None:
        lo = [-np.inf]
        hi = [np.inf]
        for _ in range(n_peaks):
            lo += [0.0, t.min(), np.finfo(float).eps]
            hi += [np.inf, t.max(), span]
        bounds = (lo, hi)

    params, _ = curve_fit(gaussian_sum, t, y, p0=p0, bounds=bounds, maxfev=maxfev)
    curve = gaussian_sum(t, *params)
    ss_res = float(((y - curve) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    peaks = [Peak(amplitude=params[i], mu=params[i + 1], sigma=params[i + 2])
             for i in range(1, params.size, 3)]
    peaks.sort(key=lambda p: p.mu)
    return PeakFit(peaks=peaks, offset=float(params[0]), curve=curve, r2=r2,
                   params=np.asarray(params, dtype=float))
