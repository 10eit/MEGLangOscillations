"""Source-space utilities: morphing, label time-courses, medial wall, adjacency."""

import numpy as np
import mne


def get_medial_vertices(fs_root, atlas="aparc_sub", specifier="unknown",
                        ico_sample=4, subject="fsaverage"):
    """Return vertex indices of the medial wall (e.g. the 'unknown' FreeSurfer label).

    Parameters
    ----------
    fs_root : str
        FreeSurfer subjects directory.
    atlas : str
        Annotation parcellation name.
    specifier : str
        Substring identifying the medial-wall labels (e.g. 'unknown').
    ico_sample : int
        Ico-sampling of fsaverage (4 -> 2562 vertices/hem, 5 -> 10242 vertices/hem).
    subject : str
        Usually 'fsaverage'.

    Returns
    -------
    medial_vertices : np.ndarray (n_medial,)
        Concatenated LH/RH vertex indices (RH offset by hemisphere vertex count).
    """
    labels = mne.read_labels_from_annot(subject, parc=atlas, subjects_dir=fs_root)
    medial_wall = [lab for lab in labels if specifier in lab.name]
    upper = 10242 if ico_sample == 5 else 2562  # ico-5 / ico-4
    lh = medial_wall[0].get_vertices_used(np.arange(0, upper))
    rh = medial_wall[1].get_vertices_used(np.arange(0, upper)) + upper
    return np.concatenate([lh, rh])


def source_adjacency(src):
    """Return the sparse source-space adjacency matrix (vertices)."""
    return mne.spatial_src_adjacency(src)


def compute_morph(inv, subject_from, fs_root, subject_to="fsaverage", spacing=4):
    """Build a source morph operator from a subject to fsaverage.

    Parameters
    ----------
    inv : mne.minimum_norm.InverseOperator
        Inverse operator providing the source space to morph from.
    subject_from, subject_to : str
    fs_root : str
        FreeSurfer subjects directory.
    spacing : int
        Target ico-sampling on fsaverage (default 4).

    Returns
    -------
    mne.SourceMorph
    """
    return mne.compute_source_morph(
        src=inv["src"], subject_from=subject_from,
        subject_to=subject_to, subjects_dir=fs_root,
        spacing=spacing, verbose="error",
    )


def extract_label_time_course(stcs, labels, src, **kwargs):
    """Wrap ``mne.extract_label_time_course`` with project defaults silenced.

    Returns array of shape (n_trials, n_labels, n_timepoints).
    """
    kwargs.setdefault("verbose", "error")
    return np.array(mne.extract_label_time_course(stcs, labels, src, **kwargs))


def read_schaefer_labels(fs_root, parc="Schaefer2018_200Parcels_7Networks_order",
                         subject="fsaverage"):
    """Read a cortical parcellation (default Schaefer-200 7Networks) for fsaverage."""
    return mne.read_labels_from_annot(subject, parc=parc, subjects_dir=fs_root)


def select_labels(all_label_names, labels_in_atlas):
    """Return the subset of label objects whose ``name`` is in ``all_label_names``."""
    return [lab for lab in labels_in_atlas if lab.name in all_label_names]