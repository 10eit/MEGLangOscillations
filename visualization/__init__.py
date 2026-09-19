"""Publication plotting toolkit.

Modules
-------
style           rcParams, millimetre sizes, palette, significance annotations, savefig.
series          mean +/- error time courses, temporal-generalization matrices.
distributions   box / violin with subject points, KDEs, scatter + fit.
posterior       posterior KDEs, information-criterion ranking / pointwise plots.
brains          source-space surface plots and standalone colour bars.

The modules only depend on matplotlib / numpy / scipy (plus optional ``mne`` for the
surface plots).  Analysis outputs are produced by :mod:`kit`; the plotting functions
take plain arrays or the DataFrames returned there.
"""

from . import style
from . import series
from . import distributions
from . import posterior
from . import brains

__all__ = ["style", "series", "distributions", "posterior", "brains"]
