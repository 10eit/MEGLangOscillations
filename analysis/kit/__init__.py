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
rsa         RSA: model RDMs, RDM correlations, model comparisons.
fitting     Latent-RDM ("cognitive effort") model fitting and Gaussian peak fitting.
cluster     Cluster-based permutation tests (1-D, vertex, spatio-temporal, surrogate).
stats       Group-level statistics (tests, effect sizes, HDI, FDR, bootstrap).
behavior    Behavioural parsing and derived indices (accuracy, RT, choice tendency).
hddm        Hierarchical drift-diffusion modelling (data prep, sampling, comparison).
embedding   Text / audio embeddings and cosine similarity.
"""

from . import io
from . import source
from . import power
from . import decoding
from . import searchlight
from . import rdm
from . import rsa
from . import fitting
from . import cluster
from . import stats
from . import behavior
from . import hddm
from . import embedding

__all__ = [
    "io",
    "source",
    "power",
    "decoding",
    "searchlight",
    "rdm",
    "rsa",
    "fitting",
    "cluster",
    "stats",
    "behavior",
    "hddm",
    "embedding",
]
