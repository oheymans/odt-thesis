import numpy as np
from sklearn.metrics import root_mean_squared_error

def evaluate_phase(phi_true, phi_unwrapped, mask=None):
    if mask is None:
        mask = np.ones_like(phi_true, dtype=bool)

    t = phi_true[mask]
    u = phi_unwrapped[mask]

    offset = np.mean(u - t)
    u_corr = u - offset

    rmse = root_mean_squared_error(t, u_corr)
    return rmse
