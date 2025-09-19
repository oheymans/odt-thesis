import numpy as np
from scipy.fftpack import dctn, idctn
def unwrap_phase_weighted(
        psi: np.ndarray, weight: np.ndarray = None, kmax: int = 100) -> np.ndarray:
    """
    Perform 2D phase unwrapping using the weighted least-squares method 
    described by Ghiglia and Romero (1994), using DCT-based Poisson solvers.

    This algorithm is particularly suited for recovering unwrapped phase 
    from wrapped noisy phase data. If a weight map is provided, it helps 
    guide the unwrapping by emphasizing regions with higher confidence.

    Parameters
    ----------
    psi : np.ndarray
        2D wrapped phase array (in radians).
    
    weight : np.ndarray or None, optional
        2D array of confidence weights, same shape as psi. If None, 
        unweighted unwrapping is performed. Default is None.
    
    kmax : int, optional
        Maximum number of iterations for the conjugate gradient solver. Default is 100.

    Returns
    -------
    phi : np.ndarray
        2D array of unwrapped phase values.

    References
    ----------
    Ghiglia, D. C., & Romero, L. A. (1994). 
    "Robust two-dimensional weighted and unweighted phase unwrapping that uses 
    fast transforms and iterative methods." JOSA A, 11(1), 107–117.
    https://doi.org/10.1364/JOSAA.11.000107
    """
    def _wrap_to_pi(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def _precompute_poisson_scaling(shape):
        N, M = shape
        I, J = np.ogrid[0:N, 0:M]
        scale = 2 * (np.cos(np.pi * I / M) + np.cos(np.pi * J / N) - 2)
        scale[0, 0] = 1.0  # avoid divide-by-zero
        return scale

    def _solve_poisson_dct(rho, scale):
        dct_rhs = dctn(rho, norm='ortho')
        phi_dct = dct_rhs / scale
        phi_dct[0, 0] = 0  # preserve mean (DC component)
        return idctn(phi_dct, norm='ortho')

    def _apply_Q(p, WWx, WWy):
        dp_x = np.diff(p, axis=1)
        dp_y = np.diff(p, axis=0)
        Wdp_x = WWx * dp_x
        Wdp_y = WWy * dp_y
        Qx = np.diff(Wdp_x, axis=1, prepend=0, append=0)
        Qy = np.diff(Wdp_y, axis=0, prepend=0, append=0)
        return Qx + Qy

    # Compute wrapped gradients
    dx = _wrap_to_pi(np.diff(psi, axis=1))
    dy = _wrap_to_pi(np.diff(psi, axis=0))

    # Weight matrix
    WW = np.ones_like(psi) if weight is None else weight ** 2
    WWx = np.minimum(WW[:, :-1], WW[:, 1:])
    WWy = np.minimum(WW[:-1, :], WW[1:, :])
    WWdx = WWx * dx
    WWdy = WWy * dy

    # Compute initial residual (right-hand side of Poisson)
    WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)
    rk = WWdx2 + WWdy2
    norm_r0 = np.linalg.norm(rk)

    # Initialize
    phi = np.zeros_like(psi)
    scale = _precompute_poisson_scaling(psi.shape)
    rkzk_prev = None

    for k in range(1, kmax + 1):
        zk = _solve_poisson_dct(rk, scale)
        rkzk = np.tensordot(rk, zk)

        if k == 1:
            pk = zk
        else:
            beta = rkzk / rkzk_prev
            pk = zk + beta * pk

        Qpk = _apply_Q(pk, WWx, WWy)
        alpha = rkzk / np.tensordot(pk, Qpk)

        phi += alpha * pk
        rk -= alpha * Qpk
        rkzk_prev = rkzk

        if np.linalg.norm(rk) < 1e-9 * norm_r0:
            break

    return phi