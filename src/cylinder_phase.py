
import numpy as np

def make_grid(Nx, Ny, px, py):
    #Centered around 0  
    x = (np.arange(Nx) - Nx/2) * px
    y = (np.arange(Ny) - Ny/2) * py
    return np.meshgrid(x, y, indexing='xy')

def cylinder_phase_perp_to_z_3d(
    Nx, Ny, px, py,           # grid (µm/px)
    R,                        # cylinder radius (µm)
    L,                        # length of cylinder
    lam, n_med,               # wavelength (µm), medium index
    n_obj,                    # scalar, 2D array, or callable n_obj(X,Y,z)
    pz=0.2,                   # z step (µm)
):
    """
    Phase for a cylinder whose axis lies along x (⊥ z), allowing n to depend on z.
    φ(x,y) = (2π/λ) * ∫ [n_obj(x,y,z) - n_med] dz over points inside the circular cross-section.

    n_obj can be: scalar, 3D function
    """
    # Build grid in x,z plane (Ny=Nz now!)
    X, Z = make_grid(Nx, Ny, px, py)  # here Y is "z-axis" in plotting
    tissue_mask = (X**2 + Z**2 <= R**2)
    # Apply finite cylinder length along x
    tissue_mask &= (np.abs(X) <= L/2)
    medium_mask = ~tissue_mask

    # y samples for integration (beam along y)
    y_axis = np.arange(-R, R + 0.5*pz, pz, dtype=float)

    # finite length along z (optional)
    mask_z = (np.abs(Z) <= (L/2.0)).astype(float)

    accum = np.zeros_like(X, dtype=float)
    for y in y_axis:
        mask_xz = ((X**2 + y**2) <= R*R).astype(float)
        mask = mask_xz * mask_z

        if np.isscalar(n_obj):
            dn_slice = (float(n_obj) - n_med) * mask
        else:
            if callable(n_obj):
                try:
                    n_map = np.asarray(n_obj(X, Z, y), dtype=float)
                except TypeError:
                    n_map = np.asarray(n_obj(X, Z), dtype=float)
            else:
                n_map = np.asarray(n_obj, dtype=float)
            if n_map.shape != X.shape:
                raise ValueError("n_obj must produce shape (Ny, Nx).")
            dn_slice = (n_map - n_med) * mask

        accum += dn_slice

    phi_true = (2*np.pi/lam) * accum * pz
    phi_wrapped = np.angle(np.exp(1j*phi_true))
    return phi_true, phi_wrapped, tissue_mask, medium_mask

