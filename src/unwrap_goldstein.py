import ctypes
import numpy as np
import os

# Compute absolute path relative to this file
dll_path = os.path.join(os.path.dirname(__file__), "qgpu.dll")
lib = ctypes.CDLL(dll_path)

lib.unwrap_phase_QGPU.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int
]
lib.unwrap_phase_QGPU.restype = None

def unwrap_phase_qgpu(wrapped: np.ndarray) -> np.ndarray:
    h, w = wrapped.shape
    wrapped = wrapped.astype(np.float32, order="C")
    unwrapped = np.zeros_like(wrapped, dtype=np.float32)
    lib.unwrap_phase_QGPU(wrapped, unwrapped, w, h)
    return unwrapped
