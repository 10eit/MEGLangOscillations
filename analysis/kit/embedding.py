"""Text and audio embeddings and cosine similarity.

Typical use is trial-level similarity between two word forms:

.. code-block:: python

    from kit.embedding import pairwise_similarity
    semantic = pairwise_similarity(list(zip(prev_words, words)), kind="text")
    phonetic = pairwise_similarity(list(zip(prev_paths, paths)), kind="audio")

Each unique item is embedded once, so a trial table with repeated words costs one
forward pass per distinct word instead of one per trial.

The heavy backends (``torch``, ``transformers``, ``librosa``) are imported lazily and
the loaded models are cached, so this module stays importable without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "DEFAULT_TEXT_MODEL",
    "DEFAULT_AUDIO_MODEL",
    "resolve_device",
    "text_embeddings",
    "audio_embeddings",
    "load_audio",
    "cosine_similarity",
    "similarity_matrix",
    "pairwise_similarity",
]

DEFAULT_TEXT_MODEL = "bert-base-chinese"
DEFAULT_AUDIO_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"

_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# device / model loading
# --------------------------------------------------------------------------- #
def resolve_device(device=None) -> str:
    """Return ``'cuda'`` when available (or the requested device), else ``'cpu'``."""
    if device is not None:
        return str(device)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch is required for embeddings
        return "cpu"


def _load_text_model(model_name: str, device: str):
    key = ("text", model_name, device)
    if key not in _CACHE:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        _CACHE[key] = (tokenizer, model)
    return _CACHE[key]


def _load_audio_model(model_name: str, device: str):
    key = ("audio", model_name, device)
    if key not in _CACHE:
        from transformers import AutoFeatureExtractor, AutoModel

        processor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        _CACHE[key] = (processor, model)
    return _CACHE[key]


def _as_list(items):
    if isinstance(items, str) or isinstance(items, Path):
        return [items]
    return list(items)


def _pool(hidden, attention_mask=None, pooling: str = "mean"):
    """Pool ``(batch, tokens, dim)`` hidden states into ``(batch, dim)``."""
    if pooling == "cls":
        return hidden[:, 0]
    if pooling == "mean":
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    raise ValueError(f"unknown pooling {pooling!r}")


def _finalize(vectors: np.ndarray, normalize: bool) -> np.ndarray:
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)
    return vectors


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
def text_embeddings(texts, *, model_name: str = DEFAULT_TEXT_MODEL, pooling: str = "mean",
                    batch_size: int = 32, max_length: int = 64, device=None,
                    normalize: bool = False) -> np.ndarray:
    """Contextual text embeddings from a HuggingFace encoder.

    Parameters
    ----------
    texts : str | sequence of str
    model_name : str
        Any HuggingFace encoder (defaults to a Chinese BERT).
    pooling : {'mean', 'cls'}
    batch_size, max_length, device : int / int / str | None
    normalize : bool
        L2-normalise the embeddings (then cosine similarity is a dot product).

    Returns
    -------
    np.ndarray, shape (n_texts, hidden_size)
    """
    import torch

    texts = _as_list(texts)
    device = resolve_device(device)
    tokenizer, model = _load_text_model(model_name, device)

    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**encoded).last_hidden_state
            out.append(_pool(hidden, encoded.get("attention_mask"), pooling).cpu().numpy())
    return _finalize(np.vstack(out), normalize)


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #
def load_audio(source, *, sr: int = 16000) -> tuple:
    """Load an audio file (or pass through an array) as ``(waveform, sr)``.

    ``source`` may be a path, ``(waveform, sr)`` or a 1-D array (then ``sr`` is used).
    """
    if isinstance(source, tuple):
        waveform, rate = source
        return np.asarray(waveform, dtype=np.float32).ravel(), int(rate)
    if isinstance(source, (str, Path)):
        try:
            import librosa

            waveform, rate = librosa.load(str(source), sr=sr, mono=True)
        except ImportError:  # pragma: no cover - soundfile fallback
            import soundfile as sf

            waveform, rate = sf.read(str(source))
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
        return np.asarray(waveform, dtype=np.float32).ravel(), int(rate)
    return np.asarray(source, dtype=np.float32).ravel(), int(sr)


def audio_embeddings(audio: Sequence, *, sr: int = 16000,
                     model_name: str = DEFAULT_AUDIO_MODEL, pooling: str = "mean",
                     batch_size: int = 8, device=None,
                     normalize: bool = False) -> np.ndarray:
    """Speech embeddings from a HuggingFace audio encoder.

    Parameters
    ----------
    audio : sequence
        Items accepted by :func:`load_audio` (paths, ``(waveform, sr)`` tuples or 1-D
        arrays assumed to be sampled at ``sr``).
    sr : int
        Target sampling rate used when loading files and for raw arrays.
    model_name, pooling, batch_size, device, normalize : see :func:`text_embeddings`.
        ``pooling='mean'`` averages the hidden states over time, i.e. one vector per
        word/file.

    Returns
    -------
    np.ndarray, shape (n_items, hidden_size)
    """
    import torch

    audio = _as_list(audio)
    device = resolve_device(device)
    processor, model = _load_audio_model(model_name, device)
    waveforms = [load_audio(item, sr=sr)[0] for item in audio]

    out = []
    with torch.no_grad():
        for start in range(0, len(waveforms), batch_size):
            batch = waveforms[start:start + batch_size]
            encoded = processor(batch, sampling_rate=sr, return_tensors="pt",
                                padding=True).to(device)
            hidden = model(**encoded).last_hidden_state
            out.append(_pool(hidden, encoded.get("attention_mask"), pooling).cpu().numpy())
    return _finalize(np.vstack(out), normalize)


# --------------------------------------------------------------------------- #
# similarity
# --------------------------------------------------------------------------- #
def cosine_similarity(a, b=None) -> np.ndarray:
    """Cosine similarity between the rows of ``a`` and ``b`` (default: within ``a``)."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = a if b is None else np.atleast_2d(np.asarray(b, dtype=float))
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a @ b.T


def similarity_matrix(items_a, items_b=None, *, kind: str = "text",
                      embed_kwargs: dict | None = None, **kwargs) -> np.ndarray:
    """Cosine similarity matrix between two sets of items (embeddings under the hood)."""
    embed_kwargs = dict(embed_kwargs or {})
    embed_kwargs.update(kwargs)
    embed = text_embeddings if kind == "text" else audio_embeddings
    emb_a = embed(items_a, normalize=True, **embed_kwargs)
    if items_b is None:
        return emb_a @ emb_a.T
    emb_b = embed(items_b, normalize=True, **embed_kwargs)
    return emb_a @ emb_b.T


def pairwise_similarity(pairs: Iterable, *, kind: str = "text",
                        embed_kwargs: dict | None = None, **kwargs) -> np.ndarray:
    """Cosine similarity for an ordered list of ``(a, b)`` item pairs.

    Each distinct item is embedded once, which keeps trial-level similarity cheap.

    Parameters
    ----------
    pairs : iterable of (item_a, item_b)
        Items must be hashable (word strings, audio paths, ...).
    kind : {'text', 'audio'}
    embed_kwargs, kwargs : forwarded to the embedding function.

    Returns
    -------
    np.ndarray, shape (n_pairs,)
    """
    pairs = [(a, b) for a, b in pairs]
    unique = list(dict.fromkeys([item for pair in pairs for item in pair]))
    index = {item: i for i, item in enumerate(unique)}

    embed_kwargs = dict(embed_kwargs or {})
    embed_kwargs.update(kwargs)
    embed = text_embeddings if kind == "text" else audio_embeddings
    embeddings = embed(unique, normalize=True, **embed_kwargs)

    left = embeddings[[index[a] for a, _ in pairs]]
    right = embeddings[[index[b] for _, b in pairs]]
    return np.einsum("ij,ij->i", left, right)
