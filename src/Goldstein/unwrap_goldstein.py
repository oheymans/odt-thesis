import ctypes
import numpy as np
import os

# Path to DLL
dll_path = os.path.join(os.path.dirname(__file__), "qgpu.dll")
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"Cannot find DLL at {dll_path}")

lib = ctypes.CDLL(dll_path)

# Function signature
lib.unwrap_phase_QGPU.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),  # phase
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),  # output
    ctypes.c_int,  # xsize
    ctypes.c_int   # ysize
]
lib.unwrap_phase_QGPU.restype = None

def unwrap_phase_qgpu(phase: np.ndarray) -> np.ndarray:
    """Python wrapper for QGPU phase unwrapping"""
    phase = np.ascontiguousarray(phase.astype(np.float32))
    out = np.zeros_like(phase, dtype=np.float32)
    lib.unwrap_phase_QGPU(phase, out, phase.shape[1], phase.shape[0])
    return out
