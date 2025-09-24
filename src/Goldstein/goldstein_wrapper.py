import numpy as np
import ctypes
import os

# Load DLL (adjust path if needed)
dll_path = r"C:\Users\oheymans\odt-thesis\src\Goldstein\goldstein.dll"
goldstein = ctypes.CDLL(dll_path)

# Declare argument types and return type
goldstein.goldstein_unwrap_array.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),  # phase
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),  # soln
    ctypes.c_int,  # xsize (width)
    ctypes.c_int   # ysize (height)
]
goldstein.goldstein_unwrap_array.restype = ctypes.c_int

def goldstein_unwrap(wrapped_phase: np.ndarray) -> np.ndarray:
    """
    Run the Goldstein phase unwrapping algorithm via DLL.
    """
    ysize, xsize = wrapped_phase.shape
    wrapped_phase = wrapped_phase.astype(np.float32, order="C")
    soln = np.zeros_like(wrapped_phase, dtype=np.float32, order="C")

    num_pieces = goldstein.goldstein_unwrap_array(
        wrapped_phase.ravel(),
        soln.ravel(),
        xsize,
        ysize
    )
    print(f"Goldstein finished, {num_pieces} disconnected regions")

    return soln
