"""BIDS / derivatives IO helpers for reading preprocessed epochs and inverse operators.

These consolidate the repeated ``read_epochs`` / ``read_inverse`` helpers found in every
analysis script of the project.
"""

import os
import mne
from mne_bids import BIDSPath


def default_paths(bids_root):
    """Return the conventional derivative directory layout used across the project.

    Parameters
    ----------
    bids_root : str
        Root of the BIDS MEG dataset.

    Returns
    -------
    dict with keys: bids_root, preproc_root, fs_root.
    """
    return {
        "bids_root": bids_root,
        "preproc_root": os.path.join(bids_root, "derivatives/preprocessing"),
        "fs_root": os.path.join(bids_root, "derivatives/FreeSurfer"),
    }


def read_epochs(subject, deriv_root, session="01", task="semvstr",
                epo_suffix="epo", ch_type="meg", preload=True):
    """Read a subject's preprocessed epochs from the derivatives folder.

    Parameters
    ----------
    subject : str
        BIDS subject label.
    deriv_root : str
        Directory holding the *-epo.fif derivative files.
    session, task, epo_suffix : str
        BIDS entities (defaults match the project: session 01, task semvstr).
    ch_type : str | None
        Channel type to pick (e.g. 'meg', 'grad', 'mag'). None keeps all.
    preload : bool
        Whether to load data into memory.

    Returns
    -------
    mne.Epochs
    """
    epo_path = BIDSPath(
        subject=subject, session=session, task=task,
        suffix=epo_suffix, datatype="meg", root=deriv_root,
        extension=".fif", check=False,
    )
    epochs = mne.read_epochs(epo_path, preload=preload, verbose="error")
    if ch_type is not None:
        epochs = epochs.pick(ch_type).copy()
    return epochs


def read_inverse(subject, preproc_root, session="01", task="semvstr",
                 description="1to45MEG"):
    """Read a subject's inverse operator.

    Parameters
    ----------
    subject : str
    preproc_root : str
        Folder containing the *-inv.fif files.
    description : str
        BIDS description tag of the covariance (default '1to45MEG').

    Returns
    -------
    mne.minimum_norm.InverseOperator
    """
    inv_path = BIDSPath(
        subject=subject, session=session, task=task,
        suffix="inv", datatype="meg", description=description,
        root=preproc_root, extension=".fif", check=False,
    )
    return mne.minimum_norm.read_inverse_operator(inv_path, verbose="error")


def read_fsaverage_src(fs_root, spacing=4):
    """Read the canonical fsaverage source-space file used for morphing."""
    path = os.path.join(fs_root, "fsaverage", "bem",
                        f"fsaverage-ico-{spacing}-src.fif")
    return mne.read_source_spaces(path, verbose="error")