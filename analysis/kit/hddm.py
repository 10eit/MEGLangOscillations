"""Hierarchical drift-diffusion modelling (HDDM) for two-choice behaviour.

The module wraps four steps that are needed for any HDDM analysis, and nothing more:

1. :func:`history_regressors` - trial-level dummies describing *which* option was
   reported on the previous deviant trial.
2. :func:`hddm_frame` - selection / recoding / subject factorisation of the trial
   table, returning the frame HDDM needs plus the subject mapping.
3. :func:`fit_hddm` and :func:`sample` - model construction and posterior sampling.
4. :func:`compare`, :func:`pointwise_ic`, :func:`delta_ic`, :func:`coefficients`,
   :func:`coefficient_summary`, :func:`subject_parameters`, :func:`drift_rate` -
   model comparison and posterior summaries.

``hddm`` / ``pymc`` are imported lazily, so this module is importable without them.
Sampling parameters (draws, burn-in, chains) and the outlier mixture probability are
arguments; the module itself assumes no particular study design.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import stats as kit_stats

__all__ = [
    "history_regressors",
    "hddm_frame",
    "fit_hddm",
    "sample",
    "sample_models",
    "save_model",
    "load_model",
    "log_likelihood",
    "waic",
    "compare",
    "pointwise_ic",
    "delta_ic",
    "coefficients",
    "coefficient_summary",
    "subject_parameters",
    "drift_rate",
]


# --------------------------------------------------------------------------- #
# data preparation
# --------------------------------------------------------------------------- #
def history_regressors(trials: pd.DataFrame, *, deviant_col: str = "deviant",
                       positive="str", negative="sem", both="both",
                       names: tuple = ("x", "y")) -> pd.DataFrame:
    """Add dummies for the option reported on the previous deviant trial.

    For every trial the most recent *deviant* trial (any of ``positive``, ``negative``
    or ``both``) is located; the two returned columns are 1 when that trial was of
    type ``positive`` / ``negative`` respectively, and both are 1 when it was
    ``both``.  All other trials get 0.

    Parameters
    ----------
    trials : pandas.DataFrame
        Trials in presentation order.
    deviant_col : str
        Column holding the deviant type of each trial.
    positive, negative, both : hashable
        Labels of the two deviant categories and of the ambiguous ("both") category.
    names : (str, str)
        Names of the two dummy columns.

    Returns
    -------
    pandas.DataFrame
        Copy of ``trials`` with the two extra columns.
    """
    out = trials.copy()
    labels = out[deviant_col].to_numpy()
    is_deviant = np.isin(labels, [positive, negative, both])
    deviant_idx = np.flatnonzero(is_deviant)

    x = np.zeros(len(out), dtype=float)
    y = np.zeros(len(out), dtype=float)
    if deviant_idx.size:
        # position of the last deviant trial strictly before each trial
        previous = deviant_idx[np.searchsorted(deviant_idx, np.arange(len(out)),
                                               side="left") - 1]
        valid = np.searchsorted(deviant_idx, np.arange(len(out)), side="left") > 0
        previous_labels = np.where(valid, labels[np.clip(previous, 0, None)], None)
        x = np.isin(previous_labels, [positive, both]).astype(float)
        y = np.isin(previous_labels, [negative, both]).astype(float)
    out[names[0]] = x
    out[names[1]] = y
    return out


def hddm_frame(trials: pd.DataFrame, *, subject_col: str = "subject",
               response_col: str = "response", rt_col: str = "rt",
               response_map=None, dropna: bool = True, reset_index: bool = True):
    """Prepare a trial table for HDDM.

    Parameters
    ----------
    trials : pandas.DataFrame
        Already selected trials (see :func:`kit.behavior.filter_trials`); this function
        only handles response coding and subject factorisation.
    response_col, rt_col, subject_col : str
    response_map : dict | None
        Optional recoding of the response column, e.g. ``{1: 1, -1: 0}`` when the
        solver requires ``{0, 1}``.  Keep the mapping consistent with the sign of the
        drift rate you want to interpret.
    dropna : bool
        Drop trials without response / RT.
    reset_index : bool
        Reset the index, which keeps the row order aligned with
        :func:`pointwise_ic` output.

    Returns
    -------
    (pandas.DataFrame, dict)
        The frame with an integer ``subj_idx`` column and the mapping
        ``original subject name -> integer``.
    """
    frame = trials.copy()
    if dropna:
        frame = frame.dropna(subset=[c for c in (response_col, rt_col, subject_col)
                                     if c in frame.columns])
    if response_map is not None:
        frame[response_col] = frame[response_col].map(response_map)
        frame = frame.dropna(subset=[response_col])
    codes, uniques = pd.factorize(frame[subject_col])
    frame["subj_idx"] = codes
    mapping = {name: int(code) for code, name in enumerate(uniques)}
    if reset_index:
        frame = frame.reset_index(drop=True)
    return frame, mapping


# --------------------------------------------------------------------------- #
# fitting / sampling
# --------------------------------------------------------------------------- #
def fit_hddm(frame: pd.DataFrame, *, regressor: str | None = None, include=None,
             p_outlier: float | None = None, **kwargs):
    """Build an HDDM model.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`hddm_frame`.
    regressor : str | None
        Regression formula (``'z ~ 1 + x + y'``).  ``None`` fits the plain
        :class:`hddm.HDDM` model, otherwise :class:`hddm.HDDMRegressor`.
    include : sequence of str | None
        Parameters to estimate (e.g. ``['a', 'v', 't', 'z']`` plus variabilities).
        ``None`` uses the HDDM default parameter set.
    p_outlier : float | None
        Probability of the outlier mixture component; ``None`` leaves the HDDM
        default untouched.
    kwargs : forwarded to HDDM (e.g. ``group_only_regressors``,
        ``keep_regressor_trace``).

    Returns
    -------
    hddm.HDDM | hddm.HDDMRegressor
    """
    import hddm

    options = dict(kwargs)
    if include is not None:
        options["include"] = list(include)
    if p_outlier is not None:
        options["p_outlier"] = p_outlier
    if regressor is None:
        return hddm.HDDM(frame, **options)
    return hddm.HDDMRegressor(frame, regressor, **options)


def sample(model, *, draws: int = 5000, burn: int = 1000, chains: int = 2,
           save_name: str | None = None, progress_bar: bool = False,
           return_infdata: bool = True):
    """Draw posterior samples from a fitted model.

    Returns ``(model, inference_data)``; ``inference_data`` is ``None`` when
    ``return_infdata=False``.
    """
    if save_name is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_name)), exist_ok=True)
    return model.sample(draws, burn=burn, chains=chains, return_infdata=return_infdata,
                        save_name=save_name, progress_bar=progress_bar)


def sample_models(models: dict, *, draws: int = 5000, burn: int = 1000, chains: int = 2,
                  save_dir: str | None = None, progress_bar: bool = False) -> dict:
    """Sample several models; returns ``{name: (model, inference_data)}``."""
    out = {}
    for name, model in models.items():
        save_name = os.path.join(save_dir, name) if save_dir else None
        out[name] = sample(model, draws=draws, burn=burn, chains=chains,
                           save_name=save_name, progress_bar=progress_bar)
    return out


def save_model(model, path: str):
    """Persist a fitted model (``hddm.save`` under the hood)."""
    import hddm

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    hddm.save(model, path)
    return path


def load_model(path: str):
    """Load a fitted model saved with :func:`save_model`."""
    import hddm

    return hddm.load(path)


# --------------------------------------------------------------------------- #
# model comparison (works on log-likelihood arrays; arviz only for LOO)
# --------------------------------------------------------------------------- #
def _to_infdata(model_or_infdata, *, loglike: bool = True):
    """Accept a fitted HDDM model, an inference-data object or a raw array."""
    if hasattr(model_or_infdata, "to_infdata"):
        return model_or_infdata.to_infdata(loglike=loglike)
    return model_or_infdata


def log_likelihood(value) -> np.ndarray:
    """Return the log-likelihood array of a model / inference object / array.

    Accepts
    ------
    * ``np.ndarray`` - used as is (sample dimensions first, observations last);
    * a fitted HDDM model - converted with ``model.to_infdata(loglike=True)``;
    * an inference-data object exposing a ``log_likelihood`` group.

    Returns
    -------
    np.ndarray
        Shape ``(..., n_observations)``.
    """
    if isinstance(value, np.ndarray):
        return value
    infdata = _to_infdata(value, loglike=True)
    group = getattr(infdata, "log_likelihood", None)
    if group is None:
        raise TypeError("cannot locate a log-likelihood group; pass an array instead")
    variables = list(getattr(group, "data_vars", [])) or [k for k in group]
    if len(variables) != 1:
        raise ValueError(f"expected one log-likelihood variable, found {variables}")
    node = group[variables[0]]
    values = getattr(node, "values", None)
    if values is None:  # DataTree node
        node = node[list(node.data_vars)[0]]
        values = node.values
    return np.asarray(values, dtype=float)


def waic(log_likelihood_array, *, scale: str = "deviance") -> dict:
    """Watanabe-Akaike information criterion from a log-likelihood array.

    ``elpd_i = log mean_s exp(ll_si) - var_s ll_si`` (log pointwise predictive
    density minus the effective number of parameters), summed over observations.
    Implemented directly with numpy so it does not depend on an ArviZ version that
    still provides ``waic``.

    Parameters
    ----------
    log_likelihood_array : array_like
        ``(..., n_observations)``; all leading axes are treated as posterior samples.
    scale : {'deviance', 'log'}
        ``deviance`` returns ``-2 * elpd`` (lower is better).

    Returns
    -------
    dict with keys ``elpd``, ``elpd_i``, ``p``, ``ic``, ``ic_i``, ``scale``.
    """
    ll = np.asarray(log_likelihood_array, dtype=float)
    if ll.ndim < 2:
        raise ValueError("log-likelihood must have at least (samples, observations)")
    flat = ll.reshape(-1, ll.shape[-1])
    shift = flat.max(axis=0)
    lppd_i = shift + np.log(np.mean(np.exp(flat - shift), axis=0))
    p_i = flat.var(axis=0, ddof=1)
    elpd_i = lppd_i - p_i
    sign = -2.0 if scale == "deviance" else 1.0
    return {"elpd": float(elpd_i.sum()), "elpd_i": elpd_i, "p": float(p_i.sum()),
            "p_i": p_i, "ic": float(sign * elpd_i.sum()),
            "ic_i": sign * elpd_i, "scale": scale}


def _loo(log_likelihood_array, *, scale: str = "deviance") -> dict:
    """LOO-CV via ArviZ (requires an ArviZ version with PSIS-LOO).

    ArviZ 1.x removed ``waic`` and changed ``from_dict``; when the installed version
    cannot build the inference object the caller is pointed to ``ic='waic'``, which
    is implemented here without ArviZ.
    """
    try:
        import arviz as az
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LOO requires arviz; use ic='waic' instead") from exc
    if not hasattr(az, "loo"):
        raise RuntimeError("this ArviZ version provides no `loo`; use ic='waic'")

    ll = np.asarray(log_likelihood_array, dtype=float)
    infdata = None
    for build in (lambda: az.from_dict(log_likelihood={"obs": ll}),
                  lambda: az.from_dict({"log_likelihood": {"obs": ll}})):
        try:
            infdata = build()
            break
        except TypeError:
            continue
    if infdata is None:
        raise RuntimeError(
            "this ArviZ version changed `from_dict`; pass a log-likelihood array and "
            "use ic='waic' instead of 'loo'")

    result = az.loo(infdata, pointwise=True, scale="deviance")
    elpd_i = np.asarray(result.elpd_i).ravel()
    sign = -2.0 if scale == "deviance" else 1.0
    return {"elpd": float(elpd_i.sum()), "elpd_i": elpd_i,
            "p": float(np.asarray(result.p).sum()),
            "ic": float(sign * elpd_i.sum()), "ic_i": sign * elpd_i, "scale": scale}


def compare(models: dict, *, ic: str = "waic", scale: str = "deviance", **kwargs) -> pd.DataFrame:
    """Rank models by an information criterion.

    Parameters
    ----------
    models : dict[str, array_like | model | inference-data]
        One entry per model; see :func:`log_likelihood` for accepted values.
    ic : {'waic', 'loo'}
    scale : {'deviance', 'log'}
        ``deviance`` makes lower values better.

    Returns
    -------
    pandas.DataFrame
        Columns ``model, rank, elpd, se, p, ic, ic_se, elpd_diff, dse, weight``
        sorted best first.
    """
    if ic not in ("waic", "loo"):
        raise ValueError("ic must be 'waic' or 'loo'")
    rows = []
    for name, value in models.items():
        ll = log_likelihood(value)
        result = waic(ll, scale=scale) if ic == "waic" else _loo(ll, scale=scale)
        n_obs = result["elpd_i"].size
        rows.append({
            "model": name,
            "elpd": result["elpd"],
            "se": float(np.sqrt(n_obs * np.var(result["elpd_i"], ddof=1))),
            "p": result["p"],
            "ic": result["ic"],
            "ic_i": result["ic_i"],
            "elpd_i": result["elpd_i"],
        })
    table = pd.DataFrame(rows)
    table = table.sort_values("elpd", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    best = table.loc[0, "elpd_i"]
    table["elpd_diff"] = table["elpd"].apply(lambda e: table.loc[0, "elpd"] - e)
    table["dse"] = [float(np.sqrt(table.loc[0, "elpd_i"].size
                                  * np.var(best - row, ddof=1)))
                     for row in table["elpd_i"]]
    weights = np.exp(table["elpd"] - table["elpd"].max())
    table["weight"] = weights / weights.sum()
    table["ic_se"] = table["se"] * (2.0 if scale == "deviance" else 1.0)
    return table.drop(columns=["elpd_i"])


def pointwise_ic(models: dict, *, ic: str = "waic", scale: str = "deviance") -> pd.DataFrame:
    """Pointwise information criterion per model and observation.

    Returns
    -------
    pandas.DataFrame with columns ``model, observation, value`` (lower deviance = better).
    """
    frames = []
    for name, value in models.items():
        ll = log_likelihood(value)
        result = waic(ll, scale=scale) if ic == "waic" else _loo(ll, scale=scale)
        values = np.asarray(result["ic_i"], dtype=float).ravel()
        frames.append(pd.DataFrame({"model": name,
                                    "observation": np.arange(values.size),
                                    "value": values}))
    return pd.concat(frames, ignore_index=True)


def delta_ic(pointwise: pd.DataFrame, *, reference: str | None = None,
             value_col: str = "value", model_col: str = "model",
             observation_col: str = "observation") -> pd.DataFrame:
    """Difference of the pointwise criterion against the best (or a given) model.

    Positive values mean worse pointwise performance than the reference.
    """
    if reference is None:
        totals = pointwise.groupby(model_col)[value_col].sum()
        reference = totals.idxmin()
    base = (pointwise[pointwise[model_col] == reference]
            .set_index(observation_col)[value_col])
    others = pointwise[pointwise[model_col] != reference].copy()
    others["delta"] = (others[value_col]
                       - others[observation_col].map(base).to_numpy())
    others["reference"] = reference
    return others


# --------------------------------------------------------------------------- #
# posteriors
# --------------------------------------------------------------------------- #
def _posterior(idata, var_names=None) -> dict:
    """Flattened posterior samples per variable from a Dataset or DataTree."""
    posterior = idata.posterior if hasattr(idata, "posterior") else idata
    if var_names is None:
        names = list(getattr(posterior, "data_vars", [])) or [k for k in posterior]
    else:
        names = list(var_names)
    out = {}
    for name in names:
        node = posterior[name]
        values = getattr(node, "values", None)
        if values is None:  # DataTree node
            variables = list(getattr(node, "data_vars", [])) or [k for k in node]
            if not variables:
                raise KeyError(f"no posterior variable named {name!r}")
            values = node[variables[0]].values
        out[name] = np.asarray(values, dtype=float).ravel()
    return out


def coefficients(model_or_infdata, var_names=None) -> pd.DataFrame:
    """Flattened posterior samples of the requested variables (one column per variable)."""
    return pd.DataFrame(_posterior(_to_infdata(model_or_infdata, loglike=False), var_names))


def coefficient_summary(model_or_infdata, var_names=None, *, prob: float = 0.95,
                        reference: float = 0.0) -> pd.DataFrame:
    """Posterior mean / SD / HDI and the decision ``excludes_reference`` per coefficient.

    A regression effect is conventionally declared present when the HDI excludes the
    reference value, which is exactly the ``excludes_reference`` column.
    """
    samples = coefficients(model_or_infdata, var_names)
    return kit_stats.posterior_summary(samples, prob=prob, reference=reference, axis=0)


def subject_parameters(model, *, mapping: dict | None = None) -> pd.DataFrame:
    """Per-subject posterior means (``model.params``), re-indexed to subject names.

    Parameters
    ----------
    model : fitted hddm model (must have been sampled)
    mapping : dict | None
        Subject name -> integer code as returned by :func:`hddm_frame`.
    """
    params = getattr(model, "params", None)
    if params is None or not len(params):
        raise RuntimeError("the model has no `params`; sample it before reading them")
    params = pd.DataFrame(params).copy()
    if mapping is not None:
        inverse = {code: name for name, code in mapping.items()}
        params.index = [inverse.get(i, i) for i in range(len(params))]
        params.index.name = "subject"
    return params


def drift_rate(model, *, param: str = "v", mapping: dict | None = None) -> np.ndarray:
    """Per-subject drift rate from a fitted model, ordered by subject code.

    The sign follows the response coding: with ``STRUCTURAL = +1`` a positive drift
    rate means evidence accumulation towards the structural boundary.
    """
    params = subject_parameters(model, mapping=mapping)
    if param not in params.columns:
        raise KeyError(f"{param!r} not in the fitted parameters: {list(params.columns)}")
    return params[param].to_numpy(dtype=float)
