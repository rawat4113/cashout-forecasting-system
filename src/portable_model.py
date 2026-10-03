"""
Portable, dependency-free model runtime.

Why this exists
---------------
* A scikit-learn `.joblib` is a Python pickle: it ties inference to one scikit-learn version
  and, because it embeds raw numpy buffers, to the byte order of the machine that wrote it.
  IBM Z / LinuxONE (s390x) is BIG-endian; most laptops (x86-64 / ARM64) are little-endian.
* So the trained HistGradientBoosting trees are exported once into a flat `.npz` with
  EXPLICIT little-endian dtypes ('<f8', '<i4'). On load they are converted to the host's native
  byte order, and inference needs only numpy -- no scikit-learn on the serving node.

Training can stay anywhere (laptop / cloud); only this small artifact moves to the Z system.
`tests/test_portable_model.py` asserts bit-for-bit-close parity with scikit-learn.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

_FORMAT_VERSION = 1


def export_hgb(model, path, feature_names) -> str:
    """Flatten a fitted HistGradientBoostingClassifier (binary) into a portable npz. Returns sha256."""
    if len(model.classes_) != 2:
        raise ValueError("only binary classifiers are supported")
    feats, thr, left, right, val, miss, roots = [], [], [], [], [], [], []
    offset = 0
    for it in model._predictors:
        tree = it[0]
        nodes = tree.nodes
        n = len(nodes)
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits are not supported by the portable runtime")
        is_leaf = nodes["is_leaf"].astype(bool)
        idx = np.arange(n) + offset
        feats.append(np.where(is_leaf, 0, nodes["feature_idx"]))
        thr.append(np.where(is_leaf, np.inf, nodes["num_threshold"]))
        # leaves point to themselves, so a fixed number of traversal steps is always safe
        left.append(np.where(is_leaf, idx, nodes["left"].astype(np.int64) + offset))
        right.append(np.where(is_leaf, idx, nodes["right"].astype(np.int64) + offset))
        val.append(nodes["value"])
        miss.append(nodes["missing_go_to_left"].astype(np.uint8))
        roots.append(offset)
        offset += n
    max_depth = max(int(it[0].nodes["depth"].max()) for it in model._predictors)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.array([_FORMAT_VERSION], dtype="<i4"),
        feature_idx=np.concatenate(feats).astype("<i4"),
        threshold=np.concatenate(thr).astype("<f8"),
        left=np.concatenate(left).astype("<i4"),
        right=np.concatenate(right).astype("<i4"),
        value=np.concatenate(val).astype("<f8"),
        missing_left=np.concatenate(miss).astype("u1"),
        roots=np.array(roots, dtype="<i4"),
        baseline=np.array([float(np.ravel(model._baseline_prediction)[0])], dtype="<f8"),
        max_depth=np.array([max_depth], dtype="<i4"),
        feature_names=np.array(list(feature_names)),
    )
    return file_sha256(path)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PortableGBM:
    """Numpy-only gradient-boosted-tree scorer (vectorised over rows AND trees)."""

    def __init__(self, arrays: dict, sha256: str = ""):
        # astype() with a native dtype performs the byte-swap on big-endian hosts (s390x)
        self.feature_idx = arrays["feature_idx"].astype(np.int64)
        self.threshold = arrays["threshold"].astype(np.float64)
        self.left = arrays["left"].astype(np.int64)
        self.right = arrays["right"].astype(np.int64)
        self.value = arrays["value"].astype(np.float64)
        self.missing_left = arrays["missing_left"].astype(bool)
        self.roots = arrays["roots"].astype(np.int64)
        self.baseline = float(arrays["baseline"][0])
        self.max_depth = int(arrays["max_depth"][0])
        self.feature_names = [str(x) for x in arrays["feature_names"]]
        self.n_features = len(self.feature_names)
        self.sha256 = sha256

    @classmethod
    def load(cls, path) -> "PortableGBM":
        with np.load(path, allow_pickle=False) as z:     # allow_pickle=False: no code execution on load
            arrays = {k: z[k] for k in z.files}
        if int(arrays["format_version"][0]) != _FORMAT_VERSION:
            raise ValueError("unsupported portable-model format")
        return cls(arrays, file_sha256(path))

    def raw_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float64)
        n = X.shape[0]
        node = np.broadcast_to(self.roots[:, None], (len(self.roots), n)).copy()   # (trees, rows)
        rows = np.arange(n)[None, :]
        for _ in range(self.max_depth):
            x = X[rows, self.feature_idx[node]]
            go_left = np.where(np.isnan(x), self.missing_left[node], x <= self.threshold[node])
            node = np.where(go_left, self.left[node], self.right[node])
        return self.baseline + self.value[node].sum(axis=0)

    def predict_proba1(self, X: np.ndarray) -> np.ndarray:
        """P(class 1) per row."""
        return 1.0 / (1.0 + np.exp(-self.raw_predict(X)))
