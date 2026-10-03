import joblib
import numpy as np

from src import config as C
from src.common import load_panel
from src.features import FEATURE_COLUMNS
from src.portable_model import PortableGBM, export_hgb


def test_portable_matches_sklearn(tmp_path):
    model = joblib.load(C.MODEL_DIR / "hgb_model.joblib")
    path = tmp_path / "m.npz"
    export_hgb(model, path, FEATURE_COLUMNS)
    pm = PortableGBM.load(path)
    _, _, panel = load_panel()
    X = panel[FEATURE_COLUMNS].to_numpy()[::7]
    ref = model.predict_proba(X)[:, 1]
    got = pm.predict_proba1(X)
    assert np.max(np.abs(ref - got)) < 1e-9


def test_big_endian_roundtrip(tmp_path):
    """Simulate an s390x host: store every array byte-swapped (big-endian) and score again."""
    model = joblib.load(C.MODEL_DIR / "hgb_model.joblib")
    path = tmp_path / "m.npz"
    export_hgb(model, path, FEATURE_COLUMNS)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    swapped = {k: (v.astype(v.dtype.newbyteorder(">")) if v.dtype.kind in "fiu" and v.dtype.itemsize > 1 else v)
               for k, v in arrays.items()}
    assert swapped["threshold"].dtype.byteorder in (">", "=") and swapped["threshold"].dtype.str.startswith(">")
    a, b = PortableGBM(arrays), PortableGBM(swapped)
    X = np.random.default_rng(0).random((50, len(FEATURE_COLUMNS))) * 10
    assert np.allclose(a.predict_proba1(X), b.predict_proba1(X))
