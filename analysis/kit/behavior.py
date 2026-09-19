"""Behavioural parsing and derived indices.

Pipeline
--------
``read_run`` -> ``trial_table`` -> (``filter_trials``) -> ``accuracy_rt`` /
``choice_tendency`` -> ``analysis.kit.hddm``.

Response coding
---------------
``STRUCTURAL = +1``, ``SEMANTIC = -1``, ``0`` = no/other response.  The raw PsychoPy
log stores the key presses as a string representation of the response trace; the
parsers below are deliberately tolerant and only rely on the key codes, not on the
exact string layout.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "STRUCTURAL",
    "SEMANTIC",
    "list_subjects",
    "find_run_file",
    "read_run",
    "parse_response",
    "parse_rt",
    "word_length",
    "trial_table",
    "filter_trials",
    "accuracy_rt",
    "choice_tendency",
]

STRUCTURAL = 1
SEMANTIC = -1

DEFAULT_COLUMNS = {
    "block": "encoding_list",
    "word": "word_stim",
    "deviant": "word_deviant",
    "response": "subject_resp.keys",
    "rt": "subject_resp.rt",
}


# --------------------------------------------------------------------------- #
# raw files
# --------------------------------------------------------------------------- #
def list_subjects(raw_root, *, prefix: str | None = None, drop_first: int = 0,
                  pattern: str | None = None) -> list:
    """Subject folder names under ``raw_root``.

    Parameters
    ----------
    raw_root : str | Path
    prefix : str | None
        Only keep names starting with this string.
    drop_first : int
        Drop the first ``drop_first`` names (e.g. pilot participants).
    pattern : str | None
        Regular expression the name must match.
    """
    names = sorted(entry.name for entry in Path(raw_root).iterdir() if entry.is_dir())
    if prefix is not None:
        names = [n for n in names if n.startswith(prefix)]
    if pattern is not None:
        names = [n for n in names if re.search(pattern, n)]
    return names[drop_first:]


def find_run_file(raw_root, subject, run: int, *, extension: str = ".csv",
                  pattern: str | None = None) -> Path:
    """Path of the ``run``-th recording of a subject.

    Files are matched on a leading run number (``'1_...csv'``) unless ``pattern``
    (a regular expression) is given.
    """
    folder = Path(raw_root) / subject
    if not folder.is_dir():
        raise FileNotFoundError(f"subject folder not found: {folder}")
    regex = re.compile(pattern) if pattern else re.compile(rf"^{run}\D")
    for file in sorted(folder.iterdir()):
        if file.suffix == extension and regex.search(file.name):
            return file
    raise FileNotFoundError(f"no {extension} file for run {run} in {folder}")


def read_run(raw_root, subject, run: int, *, columns: dict | None = None,
             response_kwargs: dict | None = None, rt_kwargs: dict | None = None,
             **kwargs) -> pd.DataFrame:
    """Read one run and return a tidy trial table.

    Parameters
    ----------
    columns : dict | None
        Mapping with the keys ``block, word, deviant, response, rt`` pointing at the
        raw column names (see :data:`DEFAULT_COLUMNS`).  ``word`` may be ``None``.
    response_kwargs : dict | None
        Forwarded to :func:`parse_response` (key codes, first/last response).
    rt_kwargs : dict | None
        Forwarded to :func:`parse_rt`.
    kwargs : forwarded to :func:`pandas.read_csv`.
    """
    columns = {**DEFAULT_COLUMNS, **(columns or {})}
    path = find_run_file(raw_root, subject, run)
    raw = pd.read_csv(path, **kwargs)
    raw.columns = [c.lstrip("\ufeff") for c in raw.columns]

    table = pd.DataFrame({
        "subject": subject,
        "run": run,
        "block": raw[columns["block"]],
        "word": raw[columns["word"]] if columns.get("word") else pd.NA,
        "deviant": raw[columns["deviant"]],
    })
    table["response"] = parse_response(raw[columns["response"]], **(response_kwargs or {}))
    table["rt"] = parse_rt(raw[columns["rt"]], **(rt_kwargs or {}))
    return table


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def _numeric_sequence(value) -> list:
    """Extract the numbers of a scalar, a list-literal string or a trace string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    if isinstance(value, (list, tuple, np.ndarray)):
        out = []
        for item in value:
            out.extend(_numeric_sequence(item))
        return out
    text = str(value).strip()
    if text in ("", "None", "nan"):
        return []
    try:
        return _numeric_sequence(ast.literal_eval(text))
    except (ValueError, SyntaxError):
        return [float(m) for m in re.findall(r"-?\d+\.?\d*", text)]


def parse_response(keys, *, structural_codes=(3, 4), semantic_codes=(1, 2),
                   last: bool = True) -> np.ndarray:
    """Map raw key traces to ``{STRUCTURAL, SEMANTIC, 0}``.

    Parameters
    ----------
    keys : sequence
        Raw response column (strings, list literals, scalars).
    structural_codes, semantic_codes : sequence of int
        Key codes assigned to each option.
    last : bool
        Use the last (``True``) or first (``False``) recorded key of a trial.

    Returns
    -------
    np.ndarray of int
    """
    structural_codes = set(int(c) for c in structural_codes)
    semantic_codes = set(int(c) for c in semantic_codes)
    out = np.zeros(len(keys), dtype=int)
    for i, value in enumerate(keys):
        codes = [int(c) for c in _numeric_sequence(value)]
        for code in (reversed(codes) if last else codes):
            if code in structural_codes:
                out[i] = STRUCTURAL
                break
            if code in semantic_codes:
                out[i] = SEMANTIC
                break
    return out


def parse_rt(traces, *, last: bool = True, nonnegative: bool = True) -> np.ndarray:
    """Extract the reaction time of each trial from a (possibly multi-key) trace."""
    out = np.full(len(traces), np.nan)
    for i, value in enumerate(traces):
        values = _numeric_sequence(value)
        if not values:
            continue
        rt = values[-1] if last else values[0]
        if nonnegative and rt < 0:
            rt = np.nan
        out[i] = rt
    return out


def word_length(labels, *, short_pattern: str = r"[12]$", short: int = 3,
                long: int = 4) -> np.ndarray:
    """Syllable count implied by a block label (e.g. ``'AAB_encoding1'`` -> ``short``)."""
    return np.array([short if re.search(short_pattern, str(l)) else long for l in labels],
                    dtype=int)


def trial_table(raw_root, subjects, *, runs=(1, 2, 3), columns: dict | None = None,
                response_kwargs: dict | None = None, rt_kwargs: dict | None = None,
                length: bool = True, short_pattern: str = r"[12]$",
                short: int = 3, long: int = 4,
                read_kwargs: dict | None = None) -> pd.DataFrame:
    """Concatenate the tidy trial tables of several subjects / runs.

    Adds a ``length`` column (:func:`word_length` on the block label) when
    ``length=True``.
    """
    frames = []
    for subject in subjects:
        for run in runs:
            try:
                frames.append(read_run(raw_root, subject, run, columns=columns,
                                       response_kwargs=response_kwargs,
                                       rt_kwargs=rt_kwargs, **(read_kwargs or {})))
            except FileNotFoundError:
                continue
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    if length:
        table["length"] = word_length(table["block"], short_pattern=short_pattern,
                                      short=short, long=long)
    return table


# --------------------------------------------------------------------------- #
# trial selection
# --------------------------------------------------------------------------- #
def filter_trials(trials: pd.DataFrame, *, rt_col: str = "rt", subject_col: str = "subject",
                  response_col: str = "response", deviant_col: str = "deviant",
                  deviants=None, responses=None, rt_bounds=None, z_bounds=None,
                  dropna: bool = True) -> pd.DataFrame:
    """Select and clean trials.

    Parameters
    ----------
    deviants, responses : sequence | None
        Keep only these values (``None`` keeps everything).
    rt_bounds : (float, float) | None
        Keep reaction times inside the interval.
    z_bounds : (float, float) | None
        Keep trials whose within-subject z-scored RT lies inside the interval.
    dropna : bool
        Drop rows with missing RT / response.

    Returns
    -------
    pandas.DataFrame (a copy, index reset)
    """
    out = trials.copy()
    if deviants is not None:
        out = out[out[deviant_col].isin(list(deviants))]
    if responses is not None:
        out = out[out[response_col].isin(list(responses))]
    if dropna:
        out = out.dropna(subset=[c for c in (rt_col, response_col, subject_col)
                                 if c in out.columns])
    if rt_bounds is not None:
        out = out[(out[rt_col] >= rt_bounds[0]) & (out[rt_col] <= rt_bounds[1])]
    if z_bounds is not None:
        z = out.groupby(subject_col)[rt_col].transform(
            lambda x: (x - x.mean()) / x.std(ddof=1))
        out = out[(z >= z_bounds[0]) & (z <= z_bounds[1])]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# derived indices
# --------------------------------------------------------------------------- #
def accuracy_rt(trials: pd.DataFrame, *, correct, subject_col: str = "subject",
                condition_col: str = "deviant", rt_col: str = "rt",
                rt_correct_only: bool = False, condition_values=None) -> pd.DataFrame:
    """Accuracy and mean RT per subject and condition (long format).

    Parameters
    ----------
    correct : array_like | callable | str
        Boolean correctness per trial: an array, the name of a boolean column, or a
        callable evaluated on the frame.  Trial counts are taken from the data, so no
        fixed number of trials is assumed.
    rt_correct_only : bool
        Average RT over correct trials only.

    Returns
    -------
    pandas.DataFrame with columns ``subject, condition, n, accuracy, rt``.
    """
    frame = trials.copy()
    if callable(correct):
        frame["_correct"] = frame.apply(correct, axis=1)
    elif isinstance(correct, str):
        frame["_correct"] = frame[correct]
    else:
        frame["_correct"] = np.asarray(correct)
    frame["_correct"] = frame["_correct"].astype(bool)

    conditions = (sorted(frame[condition_col].dropna().unique())
                  if condition_values is None else list(condition_values))
    rows = []
    for subject, sub in frame.groupby(subject_col):
        for condition in conditions:
            cell = sub[sub[condition_col] == condition]
            if cell.empty:
                continue
            rt_data = cell.loc[cell["_correct"], rt_col] if rt_correct_only else cell[rt_col]
            rows.append({
                "subject": subject,
                "condition": condition,
                "n": int(len(cell)),
                "accuracy": float(cell["_correct"].mean()),
                "rt": float(rt_data.mean(skipna=True)),
            })
    return pd.DataFrame(rows)


def choice_tendency(trials: pd.DataFrame, *, subject_col: str = "subject",
                    condition_col: str = "deviant", condition_value=None,
                    response_col: str = "response", rt_col: str = "rt",
                    level_col: str | None = None, sign: dict | None = None) -> pd.DataFrame:
    """Signed-RT choice-tendency index.

    The per-trial score is ``sign(response) * rt``; the per-subject index is its mean.
    With ``STRUCTURAL = +1`` and ``SEMANTIC = -1`` a positive index means a structural
    bias, a negative index a semantic bias.

    Parameters
    ----------
    condition_value : hashable | None
        Restrict to one condition (e.g. the ambiguous trials).
    level_col : str | None
        Optional extra grouping column (e.g. syllable count).
    sign : dict | None
        Mapping ``response value -> sign``; defaults to ``{STRUCTURAL: +1,
        SEMANTIC: -1}``.

    Returns
    -------
    pandas.DataFrame with columns ``subject``, ``[level]``, ``tendency``, ``n``.
    """
    sign = {STRUCTURAL: 1.0, SEMANTIC: -1.0} if sign is None else dict(sign)
    frame = trials
    if condition_value is not None:
        frame = frame[frame[condition_col] == condition_value]
    frame = frame.copy()
    frame["_score"] = frame[rt_col] * frame[response_col].map(sign)
    frame = frame.dropna(subset=["_score"])

    group = [subject_col] + ([level_col] if level_col else [])
    out = (frame.groupby(group)["_score"]
           .agg(tendency="mean", n="size")
           .reset_index()
           .rename(columns={subject_col: "subject"}))
    return out
