"""Spectral power computation: sensor-level (Morlet) and source-level (induced power).

Consolidates Fig2DevPowerMVPA/SensorPower.py and *CalSensorPower.py,
Fig2PowerdB/CalSourcePower.py, Figure4Subj/Band{,Source}Power.py.
"""

import numpy as np
import mne


# Canonical frequency grid used across the project (4-45 Hz, 25 log-spaced freqs).---- #
DEFAULT_FREQS = np.logspace(np.log10(4), np.log10(45), 25)


def default_cycles(freqs=DEFAULT_FREQS):
    """Number of Morlet cycles per frequency (linear up to 7, then half freq-index)."""
    return np.hstack([np.linspace(2, 6, 7), freqs[7:] // 2])


def band_indices(freqs, band):
    """Indices of ``freqs`` falling in a named band.

    Bands: theta [4,8), alpha [8,13), beta [13,30), gamma [30,45].
    """
    bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }
    lo, hi = bands[band.lower()]
    return np.where((freqs >= lo) & (freqs < hi))[0]


BANDS = ["theta", "alpha", "beta", "gamma"]


# --------------------------------------------------------------------------- #

def compute_sensor_power(epochs, freqs=DEFAULT_FREQS, n_cycles=None,
                         baseline=(2.4, 2.7), baseline_mode="logratio",
                         crop=None, decim=5, n_jobs=1, output="power"):
    """Sensor-level single-trial Morlet TFR power, baseline-corrected & averaged per band.

    Parameters
    ----------
    epochs : mne.Epochs
    freqs : np.ndarray
    n_cycles : np.ndarray | None  defaults to ``default_cycles(freqs)``.
    baseline : tuple | None  baseline window (s).
    baseline_mode : str
    crop : tuple | None  crop window after baseline (e.g. (-0.1, 1.8)).
    decim : int
    n_jobs : int
    output : str  'power' (skips ITC).

    Returns
    -------
    band_power : np.ndarray, shape (n_trials, n_channels, n_bands, n_times)
        Band-averaged power for [theta, alpha, beta, gamma].
    """
    if n_cycles is None:
        n_cycles = default_cycles(freqs)
    tfr = mne.time_frequency.tfr_morlet(
        epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
        return_itc=False, average=False, decim=decim,
        output=output, n_jobs=n_jobs,
    )
    if baseline is not None:
        tfr.apply_baseline(baseline=baseline, mode=baseline_mode)
    if crop is not None:
        tfr.crop(*crop)
    return band_average(tfr.data, freqs)


def band_average(tfr_data, freqs, bands=BANDS):
    """Average a TFR array (..., n_freqs, n_times) within canonical bands.

    Returns shape (..., n_bands, n_times).
    """
    out = []
    for b in bands:
        idx = band_indices(freqs, b)
        if len(idx) == 0:
            continue
        out.append(tfr_data[..., idx, :].mean(axis=-2))
    return np.stack(out, axis=-2)


# --------------------------------------------------------------------------- #
def compute_source_power(epochs, inv, freqs=DEFAULT_FREQS, n_cycles=None,
                        method="eLORETA", baseline=(2.4, 2.7),
                        baseline_mode="logratio", crop=None, decim=5,
                        n_jobs=1, morph=None):
    """Source-level induced power (per frequency), optional morph to fsaverage.

    Parameters
    ----------
    epochs : mne.Epochs  (single condition recommended)
    inv : mne.minimum_norm.InverseOperator
    freqs, n_cycles : Morlet parameters.
    method : str  inverse method (default 'eLORETA').
    baseline, baseline_mode : baseline correction of the induced power.
    crop : tuple | None  crop window after baseline.
    morph : mne.SourceMorph | None  if given, applied to each frequency STC.

    Returns
    -------
    power : np.ndarray, shape (n_vertices, n_freqs, n_times)  (morphed if morph given)
    vertices : list of (lh_vertno, rh_vertno) of the *output* space
    """
    if n_cycles is None:
        n_cycles = default_cycles(freqs)
    power = mne.minimum_norm.source_induced_power(
        epochs, inv, freqs, method=method, n_cycles=n_cycles, decim=decim,
        use_fft=True, baseline=baseline, baseline_mode=baseline_mode,
        pca=True, n_jobs=n_jobs, return_plv=False, verbose="error",
    )  # (n_vertices, n_freqs, n_times)
    vertices = [inv["src"][0]["vertno"], inv["src"][1]["vertno"]]
    sfreq = epochs.info["sfreq"] / decim
    tmin = epochs.times[0]
    if crop is not None or morph is not None:
        stacked = []
        for fi in range(len(freqs)):
            stc = mne.SourceEstimate(
                data=power[:, fi, :], vertices=vertices,
                tmin=tmin, tstep=1.0 / sfreq, subject=epochs.info["subject_info"]["subject_id"] if epochs.info.get("subject_info") else None,
            )
            if crop is not None:
                stc = stc.crop(*crop)
            if morph is not None:
                stc = morph.apply(stc)
            stacked.append(stc.data)
        power = np.stack(stacked, axis=1)
        if morph is not None:
            vertices = [morph.vertices[0], morph.vertices[1]]
    return power, vertices