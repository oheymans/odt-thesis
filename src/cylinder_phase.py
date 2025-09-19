import numpy as np

def make_grid(Nx, Ny, px, py):
    """
    Build a 2D grid centered around 0.
    Returns meshgrid arrays (X, Y).
    """
    x = (np.arange(Nx) - Nx/2) * px
    y = (np.arange(Ny) - Ny/2) * py
    return np.meshgrid(x, y, indexing='xy')


def cylinder_phase_perp_to_z_3d(
    Nx, Ny, px, py,           # grid (µm/px)
    R,                        # cylinder radius in (x,z) (µm)
    L,                        # cylinder length along y (µm)
    lam, n_med,               # wavelength (µm), medium index
    n_obj,                    # scalar, 2D array, or callable n_obj(X,Y,z)
    pz=0.2,                   # z step (µm)
):
    """
    Cylinder axis = y
    Cross-section: x^2 + z^2 <= R^2
    Finite length: |y| <= L/2
    Integration: along z
    """

    # Build grid in (x,y) plane = projection plane
    X, Y = make_grid(Nx, Ny, px, py)

    # Enforce finite length along y
    inside_length = (np.abs(Y) <= L/2)

    # z samples across the radius
    z_axis = np.arange(-R, R + 0.5*pz, pz, dtype=float)

    accum = np.zeros_like(X, dtype=float)

    for z in z_axis:
        # cross-section circle in (x,z)
        cross_ok = (X**2 + z**2) <= (R*R)
        mask = inside_length & cross_ok

        if np.isscalar(n_obj):
            dn_slice = (float(n_obj) - n_med) * mask.astype(float)
        else:
            if callable(n_obj):
                n_map = np.asarray(n_obj(X, Y, z), dtype=float)
            else:
                n_map = np.asarray(n_obj, dtype=float)
            if n_map.shape != X.shape:
                raise ValueError("n_obj must produce shape (Ny, Nx).")
            dn_slice = (n_map - n_med) * mask.astype(float)

        accum += dn_slice

    # Phase
    phi_true = (2*np.pi/lam) * accum * pz
    phi_wrapped = np.angle(np.exp(1j*phi_true))

    # Mask in projection (x,y plane)
    tissue_mask = inside_length & (X**2 <= R**2)
    medium_mask = ~tissue_mask

    return phi_true, phi_wrapped, tissue_mask, medium_mask
