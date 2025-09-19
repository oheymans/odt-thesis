def background_fit(phase, mask, deg=1):
    y, x = np.mgrid[0:phase.shape[0], 0:phase.shape[1]]
    x_data, y_data = x[mask], y[mask]
    z_data = phase[mask]
    A = np.vstack([x_data**m * y_data**n for m in range(deg+1) for n in range(deg+1-m)]).T
    coeffs, *_ = np.linalg.lstsq(A, z_data, rcond=None)
    bg = np.zeros_like(phase, dtype=float)
    idx = 0
    for m in range(deg+1):
        for n in range(deg+1-m):
            bg += coeffs[idx] * (x**m) * (y**n)
            idx += 1
    return bg


"from the package - ability to choose trust worthy pixels"
#mask = np.zeros_like(phi_true, dtype=bool)
#mask[phi_true.shape[0] // 2, :] = True
#phi_unwrapped = unwrap_phase(phi_wrap)

"""
background correction
bg_wrap = background_fit(phi_wrap, medium_mask, deg=1)
phi_wrap_corr = phi_wrap - bg_wrap
unwrapped_for_plot = phi_unwrapped
bg_for_plot   = bg_unwrap
final_for_plot = phi_corrected
if not apply_correction:
    wrap_corr_for_plot = phi_wrap
    final_for_plot     = phi_unwrapped
else:
    wrap_corr_for_plot = phi_wrap_corr
    final_for_plot     = phi_corrected

bg_unwrap = background_fit(phi_unwrapped, medium_mask, deg=1)
phi_corrected = phi_unwrapped - bg_unwrap

"""