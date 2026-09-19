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


def merge_labels(label_names, atlas_labels, name: str | None = None):
    """Merge several labels of one hemisphere into a single :class:`mne.Label`.

    Parameters
    ----------
    label_names : sequence of str
        Label names to combine (they must exist in ``atlas_labels``).
    atlas_labels : sequence of mne.Label
        Labels of one annotation (e.g. ``aparc.a2005s``).
    name : str | None
        Name of the merged label (defaults to the first name).

    Returns
    -------
    mne.Label
    """
    selected = [lab for lab in atlas_labels if lab.name in set(label_names)]
    found = {lab.name for lab in selected}
    missing = set(label_names) - found
    if missing:
        raise KeyError(f"labels not found in the annotation: {sorted(missing)}")
    merged = selected[0]
    for label in selected[1:]:
        merged = merged + label
    if name is not None:
        merged.name = name
    return merged


def vertices_from_labels(labels, *, n_per_hemi: int = 2562) -> np.ndarray:
    """Vertex indices of one or more labels in the concatenated (lh, rh) space.

    Parameters
    ----------
    labels : mne.Label | sequence of mne.Label
    n_per_hemi : int
        Number of vertices per hemisphere of the annotation the labels come from
        (2562 for ico-4 fsaverage, 10242 for ico-5, 163842 for the full surface).

    Returns
    -------
    np.ndarray of int
        Right-hemisphere indices are offset by ``n_per_hemi``.
    """
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
    out = []
    for label in labels:
        index = np.arange(n_per_hemi)
        vertices = label.get_vertices_used(index)
        out.extend(vertices + (0 if label.hemi == "lh" else n_per_hemi))
    return np.array(sorted(set(int(v) for v in out)), dtype=int)


def vertex_mask(labels, *, n_per_hemi: int = 2562) -> np.ndarray:
    """Boolean mask over ``2 * n_per_hemi`` vertices marking the given labels."""
    mask = np.zeros(2 * n_per_hemi, dtype=bool)
    mask[vertices_from_labels(labels, n_per_hemi=n_per_hemi)] = True
    return mask


def label_time_courses(stcs, labels, src, *, mode: str = "mean_flip", scale: float = 1.0,
                       return_labels: bool = False):
    """Extract region time courses from source estimates.

    ``mode='mean_flip'`` averages the vertices of a label with the sign flip that
    preserves the orientation of the amplitude (the standard choice for evoked
    responses).

    Parameters
    ----------
    stcs : mne.SourceEstimate | array_like
        One source estimate or a stack of them.
    labels : sequence of mne.Label
    src : mne.SourceSpaces
    scale : float
        Multiplicative factor applied after extraction (e.g. ``1e12`` to move from T
        to pT-like units).
    return_labels : bool
        Also return the extracted label objects.

    Returns
    -------
    np.ndarray, shape (n_stcs, n_labels, n_times) (after ``scale``)
    """
    extracted = extract_label_time_course(stcs, labels, src, mode=mode)
    extracted = np.asarray(extracted, dtype=float) * float(scale)
    if return_labels:
        return extracted, list(labels)
    return extracted