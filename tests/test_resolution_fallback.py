import builtins
import sys

import numpy as np

from pemoin.utils.geometry_export import _resize_array as geometry_resize_array
from pemoin.utils.resolution import _resize_array as resolution_resize_array


def _block_optional_imaging_imports(monkeypatch) -> None:
    original_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cv2" or name == "PIL" or name.startswith("PIL."):
            raise ModuleNotFoundError(f"blocked optional dependency: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    sys.modules.pop("cv2", None)
    for module_name in list(sys.modules):
        if module_name == "PIL" or module_name.startswith("PIL."):
            sys.modules.pop(module_name, None)


def test_resolution_resize_array_uses_numpy_fallback_without_cv2_or_pil(monkeypatch) -> None:
    _block_optional_imaging_imports(monkeypatch)
    rgba = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)

    resized = resolution_resize_array(rgba, (5, 7), interpolation="bilinear")

    assert resized.shape == (5, 7, 4)
    assert resized.dtype == np.uint8
    assert np.array_equal(resized[0, 0], rgba[0, 0])
    assert np.array_equal(resized[-1, -1], rgba[-1, -1])


def test_resolution_resize_array_preserves_float_depth_with_numpy_fallback(monkeypatch) -> None:
    _block_optional_imaging_imports(monkeypatch)
    depth = np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=np.float32)

    resized = resolution_resize_array(depth, (4, 6), interpolation="bilinear")

    assert resized.shape == (4, 6)
    assert resized.dtype == np.float32
    assert float(resized.min()) >= float(depth.min())
    assert float(resized.max()) <= float(depth.max())


def test_geometry_export_resize_array_uses_numpy_fallback_without_cv2_or_pil(monkeypatch) -> None:
    _block_optional_imaging_imports(monkeypatch)
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    resized = geometry_resize_array(depth, 5, 3, interpolation="nearest")

    assert resized.shape == (3, 5)
    assert resized.dtype == np.float32
    assert set(np.unique(resized).tolist()) <= {1.0, 2.0, 3.0, 4.0}
