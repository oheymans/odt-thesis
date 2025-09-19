import ctypes
import numpy as np
import os

# Path to DLL (relative to project root)
dll_path = os.path.join(os.path.dirname(__file__), "arm.dll")
lib = ctypes.CDLL(dll_path)

# Define ctypes signature
lib.unwrap_phase_ARM.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),  # wrapped
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),  # unwrapped
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),  # mask
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),  # iW
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),  # jW
    ctypes.c_int, ctypes.c_int,          # rows, cols
    ctypes.c_double, ctypes.c_double,    # mu, lambda
    ctypes.c_int, ctypes.c_int           # numIter, ban_OmegaInit
]
lib.unwrap_phase_ARM.restype = None

def unwrap_phase_ARM(phi_wrap, mask=None, mu=0.1, lam=0.01, nIter=500, ban_OmegaInit=1):
    """Python wrapper for the C ARM unwrapping algorithm"""
    rows, cols = phi_wrap.shape
    phi_wrap = np.ascontiguousarray(phi_wrap, dtype=np.float64)

    if mask is None:
        mask = np.ones_like(phi_wrap, dtype=np.float64)
    else:
        mask = np.ascontiguousarray(mask, dtype=np.float64)

    unwrapped = np.zeros_like(phi_wrap, dtype=np.float64)
    iW = np.zeros_like(phi_wrap, dtype=np.float64)
    jW = np.zeros_like(phi_wrap, dtype=np.float64)

    lib.unwrap_phase_ARM(phi_wrap, unwrapped, mask, iW, jW,
                         rows, cols, mu, lam, nIter, ban_OmegaInit)

    return unwrapped
