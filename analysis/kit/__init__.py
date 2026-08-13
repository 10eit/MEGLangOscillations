"""
kit
---
A small MEG analysis toolkit consolidated from the project's analysis scripts.

Modules
-------
io          BIDS / derivative IO helpers (epochs, inverse operators).
source      Source-space utilities (morphing, label time courses, medial wall).
power       Sensor- and source-level spectral power (Morlet / induced power).
decoding    Sliding & generalizing (cross-temporal) decoding, permutations.
searchlight Vertex searchlight decoding & searchlight RSA.
rdm         Representational dissimilarity matrices (Euclidean + Crossnobis).
rsa         RSA model evaluation (Spearman correlations with model RDMs).
stats       Group-level statistics (one-sample t / chance, FDR correction).
"""

from . import io
from . import source
from . import power
from . import decoding
from . import searchlight
from . import rdm
from . import rsa
from . import stats

__all__ = ["io", "source", "power", "decoding", "searchlight", "rdm", "rsa", "stats"]