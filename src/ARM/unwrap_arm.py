import ctypes
import numpy as np
import os

# Path to DLL (relative to this file)
dll_path = os.path.join(os.path.dirname(__file__), "arm.dll")
lib = ctypes.CDLL(dll_path)

# Define ctypes signature for UnwrapARM
lib.UnwrapARM.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),  # G (wrapped)
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),  # Msk
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),  # iW
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),  # jW
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),  # F (unwrapped)
    ctypes.c_double, ctypes.c_double,  # mu, lambda
    ctypes.c_int, ctypes.c_int,        # numIter, ban_OmegaInit
    ctypes.c_int, ctypes.c_int         # imageW (cols), imageH (rows)
]
# default: pointer, will override inside function
lib.UnwrapARM.restype = ctypes.POINTER(ctypes.c_double)


def unwrap_phase_ARM(phi_wrap, mask=None, mu=1.0, lam=0.1, nIter=500, ban_OmegaInit=1):
    """Python wrapper for ARM phase unwrapping (2D)."""
    rows, cols = phi_wrap.shape
    phi_wrap = np.ascontiguousarray(phi_wrap, dtype=np.float64)

    if mask is None:
        mask = np.ones_like(phi_wrap, dtype=np.float64)
    else:
        mask = np.ascontiguousarray(mask, dtype=np.float64)

    iW = np.zeros_like(phi_wrap, dtype=np.float64)
    jW = np.zeros_like(phi_wrap, dtype=np.float64)
    F = np.zeros_like(phi_wrap, dtype=np.float64)  # output buffer

    # set correct return type dynamically now that we know rows, cols
    lib.UnwrapARM.restype = np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=2, shape=(rows, cols)
    )

    unwrapped = lib.UnwrapARM(
        phi_wrap, mask, iW, jW, F,
        float(mu), float(lam), int(nIter), int(ban_OmegaInit),
        cols, rows
    )

    # copy to ensure Python owns the memory
    return np.array(unwrapped, copy=True)
