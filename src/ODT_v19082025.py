import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt
from matplotlib import widgets
from datetime import datetime
import sys 
import threading
import numpy as np 
from ctypes import *
import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from matplotlib.colors import ListedColormap
from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase
import skimage.measure
import time
from scipy.io import savemat
import cmath
from warnings import warn
import scipy.io
from skimage.transform import iradon
from skimage.transform import iradon_sart
from skimage.registration import phase_cross_correlation
from numpy import diff
from skimage.transform import radon, rescale
from concurrent.futures import ThreadPoolExecutor
import itertools
import os, re, ast, gc
import cv2
import pickle
import multiprocessing
import skimage.filters as filters
import skimage.morphology as morphology
import skimage.segmentation as segmentation
from scipy import signal, optimize, ndimage, interpolate, fft 
from joblib import Parallel, delayed
from PIL import Image
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import warnings
from scipy import fftpack
from scipy.interpolate import griddata
from scipy.interpolate import SmoothBivariateSpline
from scipy import ndimage
from scipy.interpolate import interp2d 
from scipy.ndimage import center_of_mass
from skimage.filters import sobel
from skimage.exposure import rescale_intensity
from scipy.ndimage import laplace
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from scipy.stats import zscore
from scipy.interpolate import interp1d
from skimage import transform 
import tifffile as tiff
import matplotlib
from scipy.optimize import curve_fit
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.fft import dctn, idctn
from matplotlib.widgets import RectangleSelector
from scipy import fftpack
import zarr
from numcodecs import Blosc
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops 
from scipy.ndimage import median_filter
import platform
import pickle
from skimage.feature import peak_local_max
from skimage import transform

def FFT_filtering(
    hologram: np.ndarray,
    pixelsize: Tuple[float, float],
    center_padding: int,
    sample_filter_size: float,
    first_order_peak: Optional[Tuple[int, int]] = None,
    reference_hologram: Optional[np.ndarray] = None,
    low_frequency_filter: Optional[float] = None,
    filter_type: str = 'Linear',
    smoothing_sigma: Optional[float] = None,
    wavelength: Optional[float] = None,
    propagation_distance: Optional[float] = None,
    dtype: type = np.float64,
    edge_remove=True,      
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Extracts the wrapped phase from a hologram using FFT filtering and optionally applies
    numerical propagation using the Angular Spectrum Method (ASM). A reference hologram can
    be provided to isolate the sample phase from the background.

    Parameters
    ----------
    hologram : np.ndarray
        2D interferogram containing the object and reference beams.
    pixelsize : tuple of float
        Pixel size in (dy, dx) directions in millimeters.
    center_padding : int
        Padding size to exclude the DC component during automatic peak detection.
    sample_filter_size : float
        Radius of the frequency filter to extract the first-order diffraction peak (in mm^-1).
    first_order_peak : tuple of int, optional
        Coordinates (y, x) of the first-order peak in the Fourier domain. If None, peak is auto-detected.
    reference_hologram : np.ndarray, optional
        Reference hologram (no sample) used to remove background phase distortions.
    low_frequency_filter : float, optional
        Radius of a low-pass filter for optional phase background subtraction (in mm^-1).
    filter_type : str, default='Linear'
        Type of filter used: 'Linear' or 'Gaussian'.
    smoothing_sigma : float, optional
        Sigma for Gaussian smoothing applied to peak detection (if enabled).
    wavelength : float, optional
        Wavelength in millimeters, required for angular spectrum propagation.
    propagation_distance : float, optional
        Distance in millimeters to propagate the field. Positive = forward, negative = backward.

    Returns
    -------
    phase_wrapped : np.ndarray
        Wrapped phase retrieved from the hologram.
    optional_outputs : dict
        Dictionary containing detailed intermediate and final outputs:
        - 'first_order_peak': Detected or given peak location
        - 'hf_phase': Phase with low-frequency background subtracted (if available)
        - 'phase_no_propagation': Phase before propagation
        - 'extend_spatial': [x_min, x_max, y_min, y_max] in mm
        - 'extend_frequency': [fx_min, fx_max, fy_min, fy_max] in mm^-1
        - 'complex_field': Complex reconstructed field after filtering and propagation
    """
    warnings.simplefilter("ignore", RuntimeWarning)

    hologram = hologram.astype(dtype)
    if reference_hologram is not None:
        reference_hologram = reference_hologram.astype(dtype)

    def shift_data(data, shift):
        dy, dx = map(int, shift)
        return np.roll(np.roll(data, dy, axis=0), dx, axis=1)

    def filtering(data, radius, center):
        if filter_type == 'Linear':
            mask = np.sqrt((FxGrid - center[0])**2 + (FyGrid - center[1])**2) < radius
        elif filter_type == 'Gaussian':
            sigma = radius / 3
            mask = np.exp(-((FxGrid - center[0])**2 + (FyGrid - center[1])**2) / (2 * sigma**2))
        else:
            raise ValueError("Unsupported filter_type. Use 'Linear' or 'Gaussian'.")
        return data * mask + 1e-6

    def angular_spectrum_propagation(field, zshift):
        k = 2 * np.pi / wavelength
        H = np.exp(1j * zshift * np.sqrt(np.maximum(0, k**2 - (2*np.pi*FxGrid)**2 - (2*np.pi*FyGrid)**2)))
        return np.fft.ifft2(np.fft.fft2(field) * H)

    Ny, Nx = hologram.shape
    dy, dx = pixelsize
    y = np.linspace(-Ny * dy / 2, Ny * dy / 2 - dy, Ny)
    x = np.linspace(-Nx * dx / 2, Nx * dx / 2 - dx, Nx)
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    FxGrid, FyGrid = np.meshgrid(fx, np.flip(fy))

    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(hologram)))

    if first_order_peak is None:
        spectrum_log = np.log10(np.abs(spectrum) / np.max(np.abs(spectrum)))

        if edge_remove:
            spectrum_log[:center_padding, :] = np.nan
            spectrum_log[-center_padding:, :] = np.nan
            spectrum_log[:, :center_padding] = np.nan
            spectrum_log[:, -center_padding:] = np.nan

        spectrum_log[FyGrid <= center_padding] = np.nan

        peak_idx_flat = np.nanargmax(spectrum_log)
        first_order_peak = np.unravel_index(peak_idx_flat, spectrum_log.shape)

        if smoothing_sigma is not None:
            spectrum_log = filtering(spectrum_log, sample_filter_size / 3, (FxGrid[first_order_peak], FyGrid[first_order_peak]))
            spectrum_log[spectrum_log == 1e-6] = np.nan
            _, _, _, max_loc = cv2.minMaxLoc(ndimage.gaussian_filter(spectrum_log, sigma=smoothing_sigma))
            first_order_peak = (max_loc[1], max_loc[0])

    filter_center = (FxGrid[first_order_peak], FyGrid[first_order_peak])

    spectrum_filtered = filtering(spectrum, sample_filter_size, filter_center)
    spectrum_centered = shift_data(spectrum_filtered, [Ny // 2 - first_order_peak[0], Nx // 2 - first_order_peak[1]])
    hologram_filtered = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(spectrum_centered)))

    hologram_refexp = None
    if reference_hologram is not None:
        spectrum_ref = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_hologram)))
        spectrum_ref_filtered = filtering(spectrum_ref, sample_filter_size, filter_center)
        spectrum_ref_centered = shift_data(spectrum_ref_filtered, [Ny // 2 - first_order_peak[0], Nx // 2 - first_order_peak[1]])
        hologram_refexp = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(spectrum_ref_centered)))
        hologram_sample = hologram_filtered / hologram_refexp
    else:
        hologram_sample = hologram_filtered

    if propagation_distance is not None and propagation_distance != 0:
        hologram_sample = angular_spectrum_propagation(hologram_sample, propagation_distance)
        hologram_filtered = angular_spectrum_propagation(hologram_filtered, propagation_distance)

    phase_wrapped = np.angle(hologram_sample)
    intensity = np.abs(hologram_filtered) ** 2 
    if reference_hologram is not None:
        intensity -= np.abs(hologram_refexp) ** 2 


    if low_frequency_filter is not None:
        low_spectrum = filtering(spectrum, low_frequency_filter, filter_center)
        low_spectrum_centered = shift_data(low_spectrum, [Ny // 2 - first_order_peak[0], Nx // 2 - first_order_peak[1]])
        low_freq_hologram = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(low_spectrum_centered)))
        if reference_hologram is not None:
            spectrum_ref = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_hologram)))
            low_ref = filtering(spectrum_ref, low_frequency_filter, filter_center)
            low_ref_centered = shift_data(low_ref, [Ny // 2 - first_order_peak[0], Nx // 2 - first_order_peak[1]])
            low_ref_hologram = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(low_ref_centered)))
            low_freq_hologram /= low_ref_hologram
        if propagation_distance is not None and propagation_distance != 0:
            low_freq_hologram = angular_spectrum_propagation(low_freq_hologram, propagation_distance)

        hf_phase = np.angle(hologram_sample / low_freq_hologram)
    else:
        hf_phase = np.copy(phase_wrapped)
    
    extend_spatial = [x[0], x[-1], y[0], y[-1]]
    extend_frequency = [fx[0], fx[-1], fy[0], fy[-1]]

    optional_outputs = {
        'intensity':intensity,
        'hf_phase': hf_phase,
        'first_order_peak': first_order_peak,
        'extend_spatial': [e * 1e3 for e in extend_spatial],
        'extend_frequency': [f * 1e-3 for f in extend_frequency],
        'complex_field': hologram_sample,
        'spectrum':np.log10(np.abs(np.real(spectrum))/np.max(np.abs(np.real(spectrum)))), 
        'spectrum_filtered': np.log10(np.abs(np.real(spectrum_filtered))/np.max(np.abs(np.real(spectrum_filtered)))),
    }
    
    return phase_wrapped, optional_outputs

def hologram_load(
    filename: str,
    directory: str = None,
    delete_file: bool = False,
    dtype: np.dtype = np.float32,  
) -> np.ndarray:
    """
    Loads a hologram image or stack of images (TIFF) and returns a 3D NumPy array.

    Parameters
    ----------
    filename : str
        Name of the TIFF file to load.
    directory : str, optional
        Directory containing the file. If None, defaults to './hologram stack'.
    delete_file : bool, default=False
        If True, deletes the file after loading.
    dtype : np.dtype, default=np.float32
        Desired NumPy data type (e.g., np.uint8, np.uint16, np.float32, np.float16).

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (Nframes, height, width).
    """
    if directory is None:
        directory = "./hologram stack"
    filepath = os.path.join(directory, filename)
    image = Image.open(filepath)

    hologram_stack = []

    try:
        Nframes = image.n_frames
    except AttributeError:
        Nframes = 1

    for frame_ind in range(Nframes):
        if Nframes > 1:
            image.seek(frame_ind)
        frame_array = np.array(image).astype(dtype)  
        hologram_stack.append(frame_array)

    if delete_file:
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Warning: Could not delete file: {e}")

    hologram_array = np.array(hologram_stack)
    if hologram_array.ndim == 2:
        hologram_array = np.expand_dims(hologram_array, axis=0)

    return hologram_array

def image_segmentation(
    image,
    pixelsize,
    otsu_thresholding,
    scale_factor=1.0,
    blur_kernel=None,
    Fourier_filter_pass=None,
    border_remove=None,
    median_blur_size=None,
    morph_remove_obj=None,
    laser_profile=None,
    morph_openning=None,
    morph_closing=None,
    segmentation_check=False,
):

    """
    Perform tissue segmentation on phase contrast microscopy data.

    Parameters:
        image (np.ndarray): Input 2D grayscale image (phase contrast image).
        scale_factor (float): Scale to resize image for faster computation (0 < scale <= 1).
        blur_kernel (tuple): Tuple of two odd integers specifying kernel size for Gaussian blur.
        border_remove (int): Width in pixels to remove from each border to avoid edge artifacts.
        median_blur_size (int): Odd integer specifying size of the median filter kernel.
        otsu_thresholding (dict): Dictionary with keys:
            - 'classes': Number of intensity levels for multi-Otsu thresholding.
            - 'output_class': Index of the threshold class to be used.
        morph_remove_obj (dict): Dictionary with keys:
            - 'size': Minimum area (in pixels) of objects to keep.
            - 'connectivity': Pixel connectivity for object detection.
        laser_profile (dict): Dictionary with keys:
            - 'off_center': Tuple (x, y) specifying beam center offset in pixels.
            - 'radius': Radius of valid illumination area in pixels.
        morph_openning (dict): Dictionary with keys:
            - 'kernel_size': Tuple specifying kernel size for morphological opening.
            - 'iterations': Number of opening iterations.
        morph_closing (dict): Dictionary with keys:
            - 'kernel_size': Tuple specifying kernel size for morphological closing.
            - 'iterations': Number of closing iterations.
        segmentation_check (bool): If True, returns a matplotlib figure for debugging.

    Returns:
        segmented_image (np.ndarray): Boolean array with segmented tissue area.
        fig (matplotlib.figure.Figure or None): Visualization figure if segmentation_check is True.
    """

    def fourier_bandpass_filter(image, low_cutoff, high_cutoff):
        """
        Apply a band-pass filter in the Fourier domain to retain mid-range spatial frequencies in mm⁻¹.

        Parameters:
            image (2D np.ndarray): Input image.
            low_cutoff(float): Low-frequency cutoff in mm⁻¹.
            high_cutoff (float): High-frequency cutoff in mm⁻¹.

        Returns:
            filtered (2D np.ndarray): Filtered image (same size, float32 normalized to 0–255).
        """
        if low_cutoff >= high_cutoff:
            raise ValueError("low_cutoff_mm_inv must be less than high_cutoff_mm_inv")

        image = image.astype(np.float32)
        h, w = image.shape
        fy = np.fft.fftshift(np.fft.fftfreq(h, d=pixelsize[0]))
        fx = np.fft.fftshift(np.fft.fftfreq(w, d=pixelsize[1]))
        FX, FY = np.meshgrid(fx, fy)
        R = np.sqrt(FX**2 + FY**2)

        # Create band-pass mask in mm⁻¹
        mask = (R >= low_cutoff) & (R <= high_cutoff)

        # Apply FFT, filter, and inverse FFT
        F = np.fft.fftshift(np.fft.fft2(image))
        F_filtered = F * mask
        image_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))

        # Normalize result to 0–255
        image_filtered -= image_filtered.min()
        image_filtered /= (image_filtered.max() + 1e-8)
        image_filtered *= 255

        return image_filtered.astype(np.uint8)

    # --- Normalize image without creating multiple copies ---
    image = image.astype(np.float32)
    image -= np.nanmin(image)
    image /= np.nanmax(image) if np.nanmax(image) > 0 else 1
    image[np.isnan(image)] = 0
    image[np.isinf(image)] = 0
    image *= 255
    image = image.astype(np.uint8)
    if segmentation_check:
        original_image = np.copy(image)

    # --- Resize image if needed ---
    if scale_factor < 1:
        original_size = np.flip(np.shape(image))
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        if blur_kernel:
            blur_kernel = (max(int(blur_kernel[0] * scale_factor), 1), max(int(blur_kernel[1] * scale_factor), 1))
        if median_blur_size:
            median_blur_size = 2 * int(int(median_blur_size * scale_factor) / 2) + 1
        if border_remove:
            border_remove = int(border_remove * scale_factor)
        if morph_remove_obj:
            morph_remove_obj = morph_remove_obj.copy()
            morph_remove_obj['size'] = int(morph_remove_obj['size'] * scale_factor)
            morph_remove_obj['connectivity'] = int(morph_remove_obj['connectivity'] * scale_factor)
        if morph_openning:
            morph_openning = morph_openning.copy()
            morph_openning['kernel_size'] = (
                max(int(morph_openning['kernel_size'][0] * scale_factor), 1),
                max(int(morph_openning['kernel_size'][1] * scale_factor), 1)
            )
            if morph_openning['max_distance'] is not None: morph_openning['max_distance'] = int(morph_openning['max_distance'] * scale_factor)
        if morph_closing:
            morph_closing = morph_closing.copy()
            morph_closing['kernel_size'] = (
                max(int(morph_closing['kernel_size'][0] * scale_factor), 1),
                max(int(morph_closing['kernel_size'][1] * scale_factor), 1)
            )
            if morph_closing['max_distance'] is not None: morph_closing['max_distance'] = int(morph_closing['max_distance'] * scale_factor)

    # --- Filtering ---
    if blur_kernel:
        image = cv2.blur(image, blur_kernel)
    if median_blur_size:
        image = cv2.medianBlur(image, median_blur_size)
    if segmentation_check:
        filtered_image = np.copy(image)

    # Fourier high pass filter
    hp_filtered = np.zeros_like(image)
    if Fourier_filter_pass:
        image = fourier_bandpass_filter(image, low_cutoff=Fourier_filter_pass[0], high_cutoff=Fourier_filter_pass[1])
        if segmentation_check:
            hp_filtered = np.copy(image)

    # --- Apply laser mask ---
    h, w = image.shape
    if laser_profile:
        Y, X = np.ogrid[:h, :w]
        cx = 0.5 * w + scale_factor * laser_profile['off_center'][0]
        cy = 0.5 * h + scale_factor * laser_profile['off_center'][1]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        laser_mask = (dist <= (scale_factor * laser_profile['radius'])).astype(int)
        image = image * laser_mask
        laser_mask = laser_mask > 0
    else:
        laser_mask = np.ones_like(image, dtype=bool)

    # --- Thresholding ---
    thresholds = filters.threshold_multiotsu(image, classes=otsu_thresholding['classes'])
    output_adjust = np.copy(otsu_thresholding['output_adjust'])
    remask = True
    output_changed = 0 
    diff_area = 0
    
    while remask:
        
        tval = thresholds[otsu_thresholding['output_class']] * output_adjust
        if otsu_thresholding['filter'] == 'Max':
            mask = image > tval
            if np.mean(mask) > 0.5:
                mask = image < tval
        elif otsu_thresholding['filter'] == 'High':
            mask = image > tval
        elif otsu_thresholding['filter'] == 'Low':
            mask = image < tval
        elif otsu_thresholding['filter'] == 'midmax':
            midline = image[:,int(w/2)] > tval
            mask = image > tval
            if np.mean(midline) > 0.5:
                mask = image < tval

        mask = mask & laser_mask

        if segmentation_check:
            threshold_image = np.copy(mask)

        # --- Remove border ---
        if border_remove:
            mask[:border_remove, :] = False
            mask[-border_remove:, :] = False
            mask[:, :border_remove] = False
            mask[:, -border_remove:] = False
        if segmentation_check:
            border_image = np.copy(mask)

        # --- Morphological Opening ---
        mask = mask.astype(np.uint8)
        morph_openning_image = np.zeros_like(mask)
        if morph_openning:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones(morph_openning['kernel_size'], np.uint8),
                                    iterations=morph_openning['iterations'])
            # distance and grouping transform
            if morph_openning['max_distance']:
                distance_map = distance_transform_edt(mask) 
                distance_image = distance_map > morph_openning['max_distance']
                mask = (distance_image).astype(bool)
            if segmentation_check:
                morph_openning_image = np.copy(mask)

        # --- Remove small objects ---
        remove_obj_image = np.zeros_like(mask)
        if morph_remove_obj:
            mask = morphology.remove_small_objects(mask, min_size=morph_remove_obj['size'],
                                                connectivity=morph_remove_obj['connectivity'])
            mask = morphology.remove_small_holes(mask)
            mask = mask.astype(np.uint8)
            if segmentation_check:
                remove_obj_image = np.copy(mask)

        # --- Morphological Closing with padding (horizontal only) ---
        close_distance_image = np.zeros_like(mask)
        if morph_closing:
            pad = int(morph_closing['kernel_size'][1] * 2)
            padded = cv2.copyMakeBorder(mask, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
            closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE,
                                    np.ones(morph_closing['kernel_size'], np.uint8),
                                    iterations=morph_closing['iterations'])
            mask = closed[:, pad:-pad]

            # distance and grouping transform
            if morph_closing['max_distance']:
                distance_map = distance_transform_edt(mask) 
                distance_mask = distance_map > morph_closing['max_distance']
                distance_label = label(distance_mask)
                distance_regions = regionprops(distance_label)
                if distance_regions:
                    largest_region = max(distance_regions, key=lambda r: r.area)
                    distance_image = distance_label == largest_region.label
                else:
                    distance_image = distance_mask
                mask = (distance_image).astype(np.uint8)
                if segmentation_check:
                    close_distance_image = np.copy(mask)
                    
        remask = False
        if scale_factor < 1:
            mask = cv2.resize(mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        if otsu_thresholding['dynamic_adjust']['perform']:
            if otsu_thresholding['dynamic_adjust']['reference_area'] is not None:
                if output_changed < otsu_thresholding['dynamic_adjust']['max_change']:

                    area = np.sum(mask)
                    ref_area = otsu_thresholding['dynamic_adjust']['reference_area']
                    diff_area =  (area - ref_area)/ref_area * 100
                    if np.abs(diff_area) > otsu_thresholding['dynamic_adjust']['max_dif']:
                        output_adjust -= np.sign(diff_area) * otsu_thresholding['dynamic_adjust']['step']
                        output_changed += otsu_thresholding['dynamic_adjust']['step']  
                        remask = True    
            else:
                return np.sum(mask), mask
        
    # Resize segmented mask to match the original wrapped phase shape
    if scale_factor < 1:
        laser_mask = cv2.resize(laser_mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        if segmentation_check:
            filtered_image = cv2.resize(filtered_image.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)

    tissue_mask = mask.astype(bool)
    medium_mask = ~tissue_mask & laser_mask

    fig = None
    if segmentation_check:
        fig, ax = plt.subplots(3, 3, figsize=(15, 8))

        ax[0, 0].imshow(original_image, aspect='auto', cmap='gray')
        ax[0, 0].set_title('Original Image')
        ax[0, 0].axis('off')

        ax[0, 1].imshow(filtered_image*laser_mask, aspect='auto', cmap='gray')
        ax[0, 1].set_title('Guassian-Median with laser mask')
        ax[0, 1].axis('off')

        ax[0, 2].imshow(hp_filtered, aspect='auto', cmap='gray')
        ax[0, 2].set_title('Fourier high-pass filtered')
        ax[0, 2].axis('off')

        ax[1, 0].imshow(threshold_image, aspect='auto', cmap='gray')
        ax[1, 0].set_title('Thresholding')
        ax[1, 0].axis('off')

        ax[1, 2].imshow(remove_obj_image, aspect='auto', cmap='gray')
        ax[1, 2].set_title('Removed Objects')
        ax[1, 2].axis('off')

        ax[1, 1].imshow(morph_openning_image, aspect='auto', cmap='gray')
        ax[1, 1].set_title('Opened transform with ')
        ax[1, 1].axis('off')

        ax[2, 0].imshow(close_distance_image, aspect='auto', cmap='gray')
        ax[2, 0].set_title('Distance mask')
        ax[2, 0].axis('off')

        ax[2, 1].imshow(tissue_mask, aspect='auto', cmap='gray')
        ax[2, 1].set_title('Segmented Tissue')
        ax[2, 1].axis('off')

        ax[2, 2].imshow(medium_mask, aspect='auto', cmap='gray')
        ax[2, 2].set_title('Medium Mask')
        ax[2, 2].axis('off')

        plt.tight_layout()

    return tissue_mask, medium_mask, laser_mask, fig, output_changed, diff_area

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

class ODT_Collecting:

    def __init__(self, params):

        print('***** Optical Diffraction Tomography ******')

        import pylablib as pll
        pll.par["devices/dlls/niimaq"] = "path/to/dlls"

        from pylablib.devices import IMAQ
        from pylablib.devices import Thorlabs

        self.params = params
        self.camera = IMAQ.IMAQCamera()
        self.stage = Thorlabs.KinesisMotor("27500615")

        self.script_dir = os.getcwd()
        if self.params.processing_dir is None:
            self.params.processing_dir = os.getcwd()
        else:
            if platform.system() == 'Linux':
                self.params.processing_dir = self.params.processing_dir.replace("\\", "/").replace("//", "/").replace("tudelft.net", "/tudelft.net")

        self.params.hologram_stack_dir = os.path.join(self.params.processing_dir, 'hologram stack')
        os.makedirs(self.params.hologram_stack_dir, exist_ok=True) 
            
        self.hologram_stack = []
        self.reference_stack = []
        self.intensity_stack = []
        self.open_region = None

    def sample_alignment(self):

        def FOV_check(rotate=True):

            if rotate:
                # Start continuous rotation
                self.stage.jog("+",kind="continuous")  # 1 = forward direction

            window_name = "Sample Position Check"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, figsize[0], figsize[1])

            while True:
                hologram = self.camera.grab(1)[0]
                Normal_hologram = cv2.normalize(hologram, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                angle = self.stage.get_position() / 1638.4
                cv2.setWindowTitle(window_name, f"{window_name} | Angle: {angle:.1f}° | Max Intensity: {np.nanmax(hologram)}")
                cv2.imshow(window_name, Normal_hologram)

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):  
                    self.stage.stop()  # Always stop motion on exit
                    cv2.destroyAllWindows()
                    break
                else:
                    if rotate:
                        if key == ord('s'): 
                            self.stage.stop()  # Always stop motion on exit 
                            print('    The stage is stopped')
                        elif key == ord('c'): 
                            self.stage.jog("+",kind="continuous")  # 1 = forward direction 
                            print('    The stage is running again')
    
        zoom_step = self.params.collect_data['alignment']['zoom_step']
        move_step = self.params.collect_data['alignment']['move_step']
        overlay_ratio =self.params.collect_data['alignment']['overlay_ratio']
        figsize = self.params.collect_data['checking_figsize']
        gap_width = self.params.collect_data['alignment']['gap_width']
    

        if self.params.collect_data['precheck']:
            print(' Checking the sample position within the FOV ---------------------------------------------------------------------------')

            print('    Make sure that the glass is within the FOV during a full rotation.')
            print('    Make sure that the back-reflected light is aligned with the sample beam.')
            print('    👉 Press "q" in the image window to stop rotation and quit checking.')
            print('    👉 Press "s" in the image window to temporaly stop the stage.')
            print('    👉 Press "c" in the image window to restart stage jogging.')

            FOV_check(rotate=False)

            print('    Done!')

        if self.params.collect_data['alignment']['perform']:

            print(' Correcting sample misalignment and stage offset -------------------------------------------------------------------------')

            print('    Make sure that the reference beam is blocked.')
            print('    Make sure that the glass sample holder is within the FOV.')
            print('    In XY plane: correct angular and axial misalignments using the "Upper-rotational-stage" and "Y-screw", respectively.')
            print('    In YZ plane: correct angular and axial misalignments using the "Pitch-screw" and "X-screw", respectively.')
            print('    For correcting offset use the "Upper-axial-stage".')
            print('    Note: The actual shift is half of the detected shift shown on the left image panel')

            while True:
                theta_inspection = None
                respond = input(" Please determine the detection plane: 0:'xy' 1:'xz' 2:'xyz' 3:'XY-offset' 4:'XZ-offset' q:quit ").strip().lower()

                if respond == '0':
                    theta_inspection = [0, 180, False]
                    notif = f' 🔧 Correct the Sample Mislignment Along XY PLane. (image control: q=quit, +=zoom in, -=zoom out, calculater_arrows=move)'
                elif respond == '1':
                    theta_inspection = [90, 270, False]
                    notif = f' 🔧 Correct the Sample Mislignment Along XZ PLane. (image control: q=quit, +=zoom in, -=zoom out, calculater_arrows=move)'
                elif respond == '2':
                    invalid_angle = True
                    while invalid_angle:
                        respond = input("         please determine the referance angle between 0 and 180: ").strip().lower()
                        try:
                            theta_i = int(respond)
                            if theta_i > 0 and theta_i < 180:
                                theta_inspection = [theta_i, int(theta_i+180), False]
                                notif = f' 🔧 Check the Sample Mislignment Along XYZ PLane. (image control: q=quit, +=zoom in, -=zoom out, calculater_arrows=move)'
                                invalid_angle = False
                        except:
                            print('         The entered value is not valid.')
                elif respond == '3':
                    theta_inspection = [0, 180, True]
                    notif = f' 🔧 Correct the Stage Offset in XY Plane. (image control: q=quit, +=zoom in, -=zoom out, calculater_arrows=move)'
                elif respond == '4':
                    theta_inspection = [90, 270, True]
                    notif = f' 🔧 Correct the Stage Offset in XZ Plane. (image control: q=quit, +=zoom in, -=zoom out, calculater_arrows=move)'
                elif respond == 'q':
                    break
                else:
                    print('         The entered value is not valid.')
                    continue

                self.stage.move_to(1638.4*theta_inspection[1])
                while self.stage.is_moving():
                    pass
                image_i = self.camera.grab(1)[0]
                if theta_inspection[2]:
                    image_i = np.fliplr(image_i)
                image_i = cv2.normalize(image_i, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                self.stage.move_to(1638.4*theta_inspection[0])

                zoom_factor = 1.0
                move_x, move_y = 0, 0  # Center position offset in pixels

                cv2.namedWindow(notif, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(notif, 2*figsize[0], figsize[1])
                finish_move = False
                
                gap = np.zeros((image_i.shape[0], gap_width), dtype=np.uint8)  # gray gap

                while True:

                    live_image = self.camera.grab(1)[0]
                    live_image = cv2.normalize(live_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    overlay = cv2.addWeighted(image_i, overlay_ratio['theta+pi'], live_image, overlay_ratio['live'], 0)

                    if not finish_move:
                        if not self.stage.is_moving():
                            finish_move = True
                            print(notif)
                        detected_shift = cv2.addWeighted(image_i, overlay_ratio['theta+pi'], live_image, overlay_ratio['theta'], 0)

                    h, w = overlay.shape
                    desplay_shift = np.copy(detected_shift)
                    if zoom_factor > 1.0:
                        new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)

                        max_x = (w - new_w) // 2
                        max_y = (h - new_h) // 2
                        move_x = np.clip(move_x, -max_x, max_x)
                        move_y = np.clip(move_y, -max_y, max_y)

                        center_y, center_x = h // 2 + move_y, w // 2 + move_x
                        y1 = int(center_y - new_h // 2)
                        y2 = y1 + new_h
                        x1 = int(center_x - new_w // 2)
                        x2 = x1 + new_w

                        overlay = overlay[y1:y2, x1:x2]
                        desplay_shift = detected_shift[y1:y2, x1:x2]

                        overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)
                        desplay_shift = cv2.resize(desplay_shift, (w, h), interpolation=cv2.INTER_LINEAR)

                    combined = np.hstack([desplay_shift, gap, overlay])
                    combined = cv2.resize(combined, (2*figsize[0], figsize[1]), interpolation=cv2.INTER_LINEAR)

                    cv2.imshow(notif, combined)

                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('+') or key == ord('='):
                        zoom_factor = min(zoom_factor * (1 + zoom_step), 5.0)
                    elif key == ord('-') or key == ord('_'):
                        zoom_factor = max(zoom_factor / (1 + zoom_step), 1.0)
                    elif key == 52:  # Left arrow
                        move_x -= move_step
                    elif key == 54:  # Right arrow
                        move_x += move_step
                    elif key == 56:  # Up arrow
                        move_y -= move_step
                    elif key == 50:  # Down arrow
                        move_y += move_step

                cv2.destroyAllWindows()

            print('    Done!')

        print(' Checking the sample position within the FOV ---------------------------------------------------------------------------')

        print('    Step 1: make sure that the sample is within the FOV during a full rotation.')
        print('    Step 1: make sure that the sample is in focus.')
        print('    Step 1: make sure that the glass sample holder is NOT within the FOV.')
        print('    Step 1: make sure that the back-reflected light is aligned with the sample beam.')
        print('    Step 2: make sure that the reference beam is NOT blocked.')
        print('    Step 2: make sure that the maximum intensity is stable within the range of 200–240. Use the filter wheel to adjust the intensity.')
        print('    👉 Press "q" in the image window to stop rotation and quit checking.')
        print('    👉 Press "s" in the image window to temporaly stop the stage.')
        print('    👉 Press "c" in the image window to restart stage jogging.')

        FOV_check(rotate=False)

        print('    Done!')

    def collect_all_projections(self):
             
        def hologram_check(hologram):
            
            verified = True
            error = ''
            deviation = 0

            # check phase data
            if self.params.collect_data['hologram_check']['perform']:
                if self.hologram_check_reference is not None:

                    self.maxstd = self.params.collect_data['hologram_check']['maxstd_dev']

                    deviation = np.std(hologram) - self.hologram_check_reference

                    if np.abs(deviation) > self.maxstd:
                        verified = False
                        error = f'devstd-{deviation:.1f}' 
                
                if verified:
                    self.hologram_check_reference = np.std(hologram) 

            return verified,error,np.abs(deviation)
        
        print('Collecting data -------------------------------------------------------------------------')

        start_time = time.time()
        self.hologram_check_reference = None
 
        # collect the data
        theta = self.params.collect_data['theta_range'][0]
        self.stage.move_to(1638.4*theta)
        start_move = time.time()
        time_out = self.params.collect_data['stage_timeout'][0] 
        while self.stage.is_moving():
            if time.time() - start_move >= time_out:
                print(f"Stage move is manually terminated after {time_out} sec")
                break

        while theta <= self.params.collect_data['theta_range'][1]:

            if theta == self.params.collect_data['theta_range'][0]:
                set_ind = 0
            else: set_ind = 1
            Nframes = self.params.collect_data['hologram_stack'][set_ind]

            self.hologram_stack = []
            for repeat_ind in range(Nframes):

                verified = False
                iteraion = 0
                while verified is False:
                    hologram = self.camera.grab(1)[0]
                    verified,error,deviation = hologram_check(hologram)
                    iteraion+=1

                    delta_thata = self.params.collect_data['theta_range'][1]-self.params.collect_data['theta_range'][0]
                    collected_delta_theta = (theta+self.params.collect_data['theta_inc'] - self.params.collect_data['theta_range'][0])
                    total_steps = delta_thata / self.params.collect_data['theta_inc']
                    collected_steps = collected_delta_theta / self.params.collect_data['theta_inc']
                    elapsed_time = time.time()-start_time
                    time_left = total_steps/collected_steps*elapsed_time

                    progress = collected_steps/total_steps*100
                    print(f"\rCollecting projections: progress {progress:.1f}%, Angle:{theta:.2f} deg, Iteration: {iteraion}, Error: {error}, MaxI: {np.max(hologram)}, STD_Dev: {deviation:.2f}, Elapsed time: {elapsed_time/60:.1f} min, Time remained: {time_left/3600:.2f} hours)", end='', flush=True)

                # creating a stack of multiple holograms for every angle
                self.hologram_stack.append(hologram)
                
            # saving the created stack of holograms as a multi-frame TIFF file
            hologram_stack_filename = f"Angle-{theta:.3f}.tiff"
            hologram_images_to_save = [Image.fromarray(img) for img in self.hologram_stack]
            hologram_images_to_save[0].save(os.path.join(self.params.hologram_stack_dir, hologram_stack_filename), 
                                            save_all=True, append_images=hologram_images_to_save[1:])            

            # prepare the stage for the next projection
            self.hologram_stack = []
            start_move = time.time()
            self.stage.move_by(1638.4 * self.params.collect_data['theta_inc'])

            start_move = time.time()
            time_out = self.params.collect_data['stage_timeout'][1] 
            while self.stage.is_moving():
                if time.time() - start_move >= time_out:
                    print(f"Stage move is manually terminated after {time_out} sec")
                    break

            theta = self.stage.get_position()/1638.4

            # if theta - focus_checked < check_focus_interval:
            #     self.focus_sample()
            #     focus_checked = int(theta)

    def collect_reference_phase(self,position=None):

        print('\n  Collecting Reference experiment -------------------------------------------------------------------------\n')
    
        def check_sample(window_name):
            
            print('    👉 Press "q" in the image window to continue.\n')
            figsize = self.params.collect_data['checking_figsize']
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, figsize[0], figsize[1])

            while True:
                hologram = self.camera.grab(1)[0]

                # Step 1: Phase extraction via FFT filtering
                phase_wrapped, optional_output = FFT_filtering(
                    hologram=hologram,
                    pixelsize=self.params.pixelsizes,
                    wavelength=self.params.wavelength,
                    propagation_distance=None,
                    reference_hologram=None,
                    **self.params.FFT_filtering
                )
                image = optional_output['hf_phase']
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imshow(window_name, image)

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'): 
                    cv2.destroyWindow(window_name) 
                    break

        print('    Please make sure that the sample is removed from the FOV')
        check_sample(window_name="Remove sample from FOV")

        Nframes = self.params.collect_data['reference_phase']['stack']
        sleep_time = self.params.collect_data['reference_phase']['cool_down']
        maxstd = self.params.collect_data['reference_phase']['maxstd']

        # collecting reference phase
        self.reference_stack = []
        start_time = time.time()
        for repeat_ind in range(Nframes):

            verified = False
            iteraion = 0
            while verified is False:
                hologram = self.camera.grab(1)[0]

                phase_wrapped, optional_output = FFT_filtering(
                    hologram=hologram,
                    pixelsize=self.params.pixelsizes,
                    wavelength=self.params.wavelength,
                    propagation_distance=None,
                    reference_hologram=None,
                    **self.params.FFT_filtering
                )
                image = optional_output['hf_phase']
                
                if np.std(image) < maxstd:
                    verified = True

                iteraion+=1

                progress = repeat_ind/Nframes*100
                elapsed_time = time.time() - start_time
                print(f"\r Collecting refrerence phase: progress {progress:.1f}%, Iteration: {iteraion}, Phase_STD: {np.std(image)}, MaxI: {np.max(hologram)}, Elapsed time: {elapsed_time/60:.1f} min", end='', flush=True)

            # creating a stack of multiple holograms for every angle
            self.reference_stack.append(hologram)
            time.sleep(sleep_time)
            
        # saving the created stack of holograms as a multi-frame TIFF file
        if position is not None:
            reference_stack_name = f"Reference_phase_{position}.tiff"
        else: reference_stack_name = f"Reference_phase.tiff"
        hologram_images_to_save = [Image.fromarray(img) for img in self.reference_stack]
        hologram_images_to_save[0].save(os.path.join(self.params.hologram_stack_dir, reference_stack_name), 
                                        save_all=True, append_images=hologram_images_to_save[1:])  

        if position == 'before':
            print('\n    Please make sure that the sample is within from the FOV')
            check_sample(window_name="Move sample into FOV")

        cv2.destroyAllWindows()

class ODT_processing:

    def __init__(self, params):

        print('\n   ***** Optical Diffraction Tomography ******\n')

        self.params = params
        self.phase_stack = None
        self.finished_phase_processing = True
        self.finished_phase_reconstruction = True

        warnings.filterwarnings("ignore")

        self.script_dir = os.getcwd()
        if self.params.processing_dir is None:
            self.params.processing_dir = os.getcwd()
        else:
            if platform.system() == 'Linux':
                self.params.processing_dir = self.params.processing_dir.replace("\\", "/").replace("//", "/").replace("tudelft.net", "/tudelft.net")
            else:
                self.params.processing_dir = self.params.processing_dir.replace("/tudelft.net/", "\\\\tudelft.net\\").replace("/", "\\")

        self.params.hologram_stack_dir = os.path.join(self.params.processing_dir, 'hologram stack')

        print(f" ℹ️  Info: Processing directory is set to:\n     {self.params.processing_dir}")

        try:
            os.chdir(self.params.processing_dir)
        except Exception as e:
            print(e)
            print(" Note: The current directory is set as the processing directory!!!!!!!")
            self.params.processing_dir = os.getcwd()
            os.chdir(self.params.processing_dir)

    def get_saved_phase(self,delta_theta):

        # read the total order of the collected angles 
        #       Not: (due to uncertainty in stage movement the actual angle is different from the requested angle)
        try:
            collected_angles = np.load('collected_angles.npy')
        except:
            # make sure that ALL taken tomograms (not only the tomograms for processing) exist in this directory
            collected_angles = self.get_collected_angle() 
            np.save('collected_angles.npy', collected_angles)

        # list saved phase data
        pattern = r"phase_stack.*\((\d+)\)-.*\((\d+)\)\.npy"
        files = os.listdir()
        saved_theta = []
        phase_file_list = []
        FOV = None
        for filename in files:
            if filename.endswith('.npy') and 'phase_stack' in filename:
                match = re.search(pattern, filename)
                if match:
                    # Extract the two numbers from the parentheses
                    theta_i, theta_e = int(match.group(1)), int(match.group(2))
                    saved_theta.append((theta_i, theta_e))
                    phase_file_list.append(filename)
                    if FOV is None:
                        data = np.load(filename)
                        FOV = (np.shape(data)[1],np.shape(data)[2])

        # looping over a full rotation and collect phase_data
        theta_ranges = ((0,360),)
        for delta_theta_ind in range(len(theta_ranges)):
            
            delta_theta = theta_ranges[delta_theta_ind]
            theta_i = np.abs(collected_angles - delta_theta[0]).argmin()
            theta_e = np.abs(collected_angles - delta_theta[1]).argmin()
            theta_ind_range = np.arange(theta_i,theta_e+1)
            phase_data = np.full((np.shape(theta_ind_range)[0],FOV[0],FOV[1]), np.nan, dtype=self.params.phase_data_type)

            # load the corresponding phase data
            theta_index = theta_i
            for file_ind in range(len(saved_theta)):

                if theta_index >= saved_theta[file_ind][0] and theta_index <= saved_theta[file_ind][1]:
                    data = np.load(phase_file_list[file_ind])
                    
                    for theta in np.arange(theta_index,theta_e+1):
                        if theta < saved_theta[file_ind][1]:
                            phase_data[theta,:,:] = data[theta-theta_index,:,:]

                    theta_index = theta
        
        return phase_data,collected_angles

    def phase_processing(self):
        
        def get_tomogram_data():
            
            print(" Getting tomogram data")
            # Step 1: Navigate to the directory
            directory = self.params.hologram_stack_dir
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directory '{directory}' does not exist.")
            
            # Step 2: List all .tiff files containing the word "Angle"
            files = [f for f in os.listdir(directory) if f.endswith('.tiff') and "Angle" in f]
            
            # Step 3: Updated regex pattern to extract angle values with exactly 3 decimal places
            angle_pattern = re.compile(r"Angle-([0-9]+\.[0-9]{3})\.tiff$")
            collected_angles = []
            FOV = None

            for file in files:
                if FOV is None:
                    Ylim, Xlim = np.shape(hologram_load(filename=file, directory=self.params.hologram_stack_dir)[0])
                    FOV = [Ylim, Xlim]

                match = angle_pattern.search(file)
                if match:
                    try:
                        # Convert the extracted value to float
                        angle = float(match.group(1))
                        collected_angles.append(angle)
                    except ValueError:
                        print(f"Warning: Could not convert angle in file '{file}' to float.")
                else:
                    print(f"No match for file: {file}")
            
            # Step 4: Convert to numpy array and sort in ascending order
            collected_angles = np.array(collected_angles)
            collected_angles.sort()
             
            self.FOV = np.array(FOV)
            self.collected_angles = collected_angles

            np.savez(os.path.join(self.params.processing_dir, "tomogram_data.npz"), FOV=self.FOV, collected_angles=self.collected_angles)
            
        def test_processing():

            print(' Phase processing is running in Test Mode!!!')
            self.params.image_segmentation['segmentation_check'] = True

            inputs = np.linspace(0,self.collected_angles.shape[0]-1,self.params.phase_processing['Unwrapping']['check_unwrapping']['test_prj']).astype(int)

            chunk_size = min(ncpus,len(inputs))
            Njobs = len(inputs) // chunk_size if len(inputs) % chunk_size == 0 else len(inputs) // chunk_size + 1
            
            fig_frames = []
            phase_data = []
            for job_index in range(0, Njobs):
                # Define the input range for this job
                start_idx = job_index * chunk_size
                end_idx = start_idx + chunk_size if job_index < Njobs - 1 else len(inputs)
                self.job_inputs = inputs[start_idx:end_idx]
            
                # Process the current chunk
                reference_area = []
                if ncpus == 1:
                    job_results = []
                    for ind in range(len(self.job_inputs)):
                        job_results.append(parallel_processing(ind))
                else:
                    if np.shape(self.job_inputs)[0] > 0:
                        job_results = Parallel(n_jobs=ncpus)(delayed(parallel_processing)(ind) for ind in range(np.shape(self.job_inputs)[0]))

                for job_ind in range(len(job_results)):
                    fig_frames.append(job_results[job_ind]['fig_output'])
                    phase_data.append(job_results[job_ind]['phase_data'])
                    reference_area.append(job_results[job_ind]['reference_area'])

                self.params.image_segmentation['otsu_thresholding']['dynamic_adjust']['reference_area'] = np.nanmean(np.array(reference_area))

            # Create output directory
            save_path = os.path.join(self.params.processing_dir, "test_mode")
            os.makedirs(save_path, exist_ok=True)

            base_names = ['segmentation', 'unwrapping']
            ext = ".mp4"
            counter = 0

            # Find available base filename using only the first video
            while True:
                if counter == 0:
                    filename = f"{base_names[0]}{ext}"
                else:
                    filename = f"{base_names[0]}_{counter:02d}{ext}"
                if not os.path.exists(os.path.join(save_path, filename)): 
                    break
                counter += 1

            # Save videos
            for video_ind in range(2):
                if counter == 0:
                    filename = base_names[video_ind]
                else:
                    filename = f"{base_names[video_ind]}_{counter:02d}"
                filename_full = os.path.join(save_path, f"{filename}{ext}")
                Nframes = len(fig_frames)
                height, width, _ = fig_frames[0][video_ind].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename_full, fourcc, fps=3, frameSize=(width, height))

                for frame_ind in range(Nframes):
                    frame = fig_frames[frame_ind][video_ind]
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)

                out.release()

            # Save parameters to a text file
            if counter == 0:
                param_path = os.path.join(save_path, f"params.txt")
            else:
                param_path = os.path.join(save_path, f"params_{counter:02d}.txt")
            with open(param_path, 'w') as f:
                f.write("Processing Parameters:\n")
                f.write("=" * 80 + "\n\n")
                for key, val in vars(self.params).items():
                    if not key.startswith("__"):
                        f.write(f"{key}:\n")
                        f.write(f"{val}\n\n")

            self.params.phase_processing['Unwrapping']['check_unwrapping']['perform'] = False
            self.params.image_segmentation['segmentation_check'] = False

            mean_phases = []
            std_phases = []
            correlations = []

            for idx, phase in enumerate(phase_data):
                mean_phases.append(np.mean(phase))
                std_phases.append(np.std(phase))
                if idx == 0:
                    correlations.append(1.0)  # Self-correlation for first frame
                else:
                    prev_phase = phase_data[idx - 1]
                    corr_coef = np.corrcoef(phase.flatten(), prev_phase.flatten())[0, 1]
                    correlations.append(corr_coef)

            angles = self.collected_angles[inputs]

            fig, axs = plt.subplots(3, 1, figsize=(10, 12))

            axs[0].plot(angles, mean_phases, marker='o')
            axs[0].set_title('Mean of Unwrapped Phase')
            axs[0].set_xlabel('Projection Angle (degrees or radians)')
            axs[0].set_ylabel('Mean Phase')

            axs[1].plot(angles, std_phases, marker='o')
            axs[1].set_title('Standard Deviation of Unwrapped Phase')
            axs[1].set_xlabel('Projection Angle (degrees or radians)')
            axs[1].set_ylabel('Std Dev Phase')

            axs[2].plot(angles, correlations, marker='o')
            axs[2].set_title('Correlation with Previous Projection')
            axs[2].set_xlabel('Projection Angle (degrees or radians)')
            axs[2].set_ylabel('Correlation Coefficient')

            plt.tight_layout()

            if counter == 0:
                metrics_path = os.path.join(save_path, 'unwrapping_reliability.png')
            else:
                metrics_path = os.path.join(save_path, f'unwrapping_reliability_{counter:02d}.png')
            plt.savefig(metrics_path)
            plt.close()

            print("\n   Please check the saved data in test_mode folder then rerun to continue with phase processing")

        def parallel_processing(ind):

            """
            Perform phase extraction, segmentation, and unwrapping for a single hologram frame.

            This function is intended to be used in a parallel processing context.
            It loads the hologram corresponding to a specific angle, computes the wrapped phase via 
            Fourier filtering, segments tissue and background regions, and performs phase unwrapping 
            with background correction.

            Parameters
            ----------
            ind : int
                Index of the job input corresponding to a specific angle.

            Returns
            -------
            phase_unwrapped : np.ndarray
                2D array of the final unwrapped phase. If an error occurs or the data is too noisy, 
                returns a NaN-filled array of the same shape as the field of view (FOV).
            """ 
            filename = f"Angle-{self.collected_angles[self.job_inputs[ind]]:.3f}.tiff"
            try:
                output = process_projection(filename)
            except Exception as e:
                print(f"\r Phase processing for {filename} failed. Error: {e}")

            return output
          
        def process_projection(filename,propagation_distance=None,unwrapping=True):

            def phase_unwrapping(wrapped_phase, medium_mask, laser_mask):
                """
                Perform phase unwrapping and background correction on a 2D wrapped phase image.

                This function unwarps the wrapped phase and removes background phase contributions 
                by fitting a 2D polynomial surface to the non-tissue (background) regions. 
                It uses the wrapped phase to estimate an initial background, then applies 
                phase unwrapping and optionally fits a second polynomial to refine the background 
                using the unwrapped phase.

                Parameters
                ----------
                wrapped_phase : np.ndarray
                    2D array representing the wrapped phase image (in radians).
                medium_mask : np.ndarray
                    Boolean mask where True corresponds to background (non-tissue) regions.
                tissue_mask : np.ndarray
                    Boolean mask where True corresponds to tissue regions.

                Returns
                -------
                corrected_unwrapped_phase : np.ndarray
                    Final background-corrected, unwrapped phase image.
                fig : matplotlib.figure.Figure or None
                    Optional figure showing the intermediate steps of background removal and unwrapping. 
                    Returned only if background check is enabled.
                """

                def iterative_background_fit(phase, mask, settings, setting_label):
                    """
                    Iterative robust background fitting with outlier removal and image resizing downsampling.

                    Parameters:
                        phase : np.ndarray
                            Phase image.
                        mask : np.ndarray
                            Boolean mask indicating open area.
                        method : str
                            'spline' or 'polynomial'.
                        degree : int
                            Polynomial degree (if method is 'polynomial').
                        spline_smoothing : float
                            Smoothing factor for spline.
                        threshold : float
                            Outlier threshold in units of MAD.
                        max_iter : int
                            Maximum number of iterations.
                        downsample_factor : float
                            Image resizing factor (0 < factor <= 1).
                        verbose : bool
                            Print debug information.

                    Returns:
                        background : np.ndarray
                            Final fitted background at original resolution.
                        refined_mask : np.ndarray
                            Final mask after outlier removal (original resolution).
                    """

                    method = settings['method'][setting_label]
                    poly_degree = settings['polynomial_deg'][setting_label]
                    spline_smoothing = settings['spline_smoothing'][setting_label]
                    threshold = settings['outlier_threshold']
                    max_iter = settings['max_iteration']
                    downsample_factor = settings['downsample_factor']

                    ny, nx = phase.shape

                    # Downsample image and mask
                    ny_small, nx_small = int(ny * downsample_factor), int(nx * downsample_factor)
                    phase_small = transform.resize(phase, (ny_small, nx_small), preserve_range=True, anti_aliasing=True)
                    mask_small = transform.resize(mask.astype(float), (ny_small, nx_small), preserve_range=True, anti_aliasing=False) > 0.5

                    y_small, x_small = np.mgrid[0:ny_small, 0:nx_small]
                    refined_mask = mask_small.copy()

                    for i in range(max_iter):
                        x_data = x_small[refined_mask].ravel()
                        y_data = y_small[refined_mask].ravel()
                        z_data = phase_small[refined_mask].ravel()

                        valid_idx = ~np.isnan(z_data) & ~np.isinf(z_data)
                        x_data, y_data, z_data = x_data[valid_idx], y_data[valid_idx], z_data[valid_idx]

                        if method == 'spline':
                            if spline_smoothing is not None:
                                spline = SmoothBivariateSpline(x_data, y_data, z_data, s=spline_smoothing)
                                background_small = spline(np.arange(nx_small), np.arange(ny_small), grid=True).T
                            else: 
                                return np.zeros_like(phase)

                        elif method == 'polynomial':
                            if poly_degree is not None:
                                terms = [(x_data**m) * (y_data**n) for m in range(poly_degree + 1) for n in range(poly_degree + 1 - m)]
                                A = np.vstack(terms).T
                                coeffs, *_ = np.linalg.lstsq(A, z_data, rcond=None)
                                y_full, x_full = np.mgrid[0:ny_small, 0:nx_small]
                                background_small = np.zeros_like(phase_small, dtype=float)
                                idx = 0
                                for m in range(poly_degree + 1):
                                    for n in range(poly_degree + 1 - m):
                                        background_small += coeffs[idx] * (x_full ** m) * (y_full ** n)
                                        idx += 1
                            else: 
                                return np.zeros_like(phase)
                            
                        else:
                            raise ValueError("Method must be 'spline' or 'polynomial'.")

                        residual = phase_small - background_small
                        residual_data = residual[refined_mask]

                        median_residual = np.nanmedian(residual_data)
                        mad_residual = np.nanmedian(np.abs(residual_data - median_residual)) + 1e-8

                        new_mask = refined_mask & (np.abs(residual - median_residual) < threshold * mad_residual)

                        refined_mask = new_mask

                    # Upsample background to original resolution
                    background = transform.resize(background_small, (ny, nx), preserve_range=True, anti_aliasing=True)

                    return background

                # Configuration
                background_fit = self.params.phase_processing['Unwrapping']['background_fit']
                bcgrnd_check = self.params.phase_processing['Unwrapping']['check_unwrapping']

                # Step 1: Estimate background from wrapped phase
                wrapped_BG = iterative_background_fit(wrapped_phase, medium_mask, background_fit, 'wrapped')

                # Step 2: Subtract estimated background
                corrected_wrapped_phase = wrapped_phase - wrapped_BG
                corrected_wrapped_phase[~laser_mask] = 0

                # Step 3: Perform phase unwrapping
                if self.params.phase_processing['Unwrapping']['method'] == 'Scikit':
                    unwrapped_phase = unwrap_phase(corrected_wrapped_phase)
                elif self.params.phase_processing['Unwrapping']['method'] == 'Weighted':
                    unwrapped_phase = unwrap_phase_weighted(corrected_wrapped_phase)
                unwrapped_phase[~laser_mask] = np.nan

                # Step 4: Optionally estimate background from unwrapped phase
                unwrapped_BG = iterative_background_fit(unwrapped_phase, medium_mask, background_fit, 'unwrapped')

                # Step 5: Final background correction
                corrected_unwrapped_phase = unwrapped_phase - unwrapped_BG
                corrected_unwrapped_phase = corrected_unwrapped_phase
                corrected_unwrapped_phase[~laser_mask] = 0

                # Optional: Visualization of processing steps
                fig = None
                if bcgrnd_check['perform']:
                    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
                    clim = ((-np.pi, np.pi), bcgrnd_check['unwrap_clim'])

                    images = [
                        (wrapped_phase, "Wrapped Phase", clim[0]),
                        (wrapped_BG, "Background Phase (Wrapped)", clim[0]),
                        (corrected_wrapped_phase, "Corrected Wrapped Phase", None),
                        (unwrapped_phase, "Unwrapped Phase", clim[1]),
                        (unwrapped_BG, "Background Phase (Unwrapped)",clim[1]),
                        (corrected_unwrapped_phase, "Final Corrected Unwrapped", clim[1]),
                    ]

                    for ax, (img, title, lim) in zip(axes.ravel(), images):
                        im = ax.imshow(img, cmap='Spectral', vmin=lim[0], vmax=lim[1]) if lim else ax.imshow(img, cmap='Spectral')
                        ax.set_title(title)
                        ax.axis('off')
                        fig.colorbar(im, ax=ax)
                    
                    plt.tight_layout()

                return corrected_unwrapped_phase, fig

            # Load hologram image
            startprocess = time.time()

            hologram = np.squeeze(
                hologram_load(
                    filename=filename,
                    directory=self.params.hologram_stack_dir,
                    dtype=phase_processing_dtype
                )[0]
            )

            # Step 1: Phase extraction via FFT filtering
            phase_wrapped, optional_output = FFT_filtering(
                hologram=hologram,
                pixelsize=self.params.pixelsizes,
                wavelength=self.params.wavelength,
                propagation_distance=propagation_distance,
                reference_hologram=reference_hologram,
                dtype = None,
                **self.params.FFT_filtering
            )
            del hologram
            gc.collect()

            if unwrapping:
                # Step 2: Tissue/background segmentation
                image_seg = self.params.phase_processing['Unwrapping']['segmentation']

                if image_seg == 'intensity': image = optional_output['intensity']
                elif image_seg == 'hf_phase': image = optional_output['hf_phase']
                elif image_seg == 'unwrapped_Scikit': image = unwrap_phase(phase_wrapped)
                elif image_seg == 'unwrapped_Weighted': image = unwrap_phase_weighted(phase_wrapped)
                elif image_seg == 'wrapped_phase': image = phase_wrapped

                tissue_mask, medium_mask, laser_mask, fig_seg, output_changed, dev = image_segmentation(
                    image= image,
                    pixelsize=self.params.pixelsizes,
                    **self.params.image_segmentation
                )
                del optional_output
                gc.collect()

                # Step 3: Phase unwrapping (if noise level is acceptable)
                verified = True
                # add checking if it is necessary

                if verified:
                    phase_unwrapped, fig_phase = phase_unwrapping(phase_wrapped, medium_mask, laser_mask)
                    phas_data = phase_unwrapped
                    del phase_wrapped, phase_unwrapped
                    gc.collect()

                    fig_output = []
                    if self.params.phase_processing['Unwrapping']['check_unwrapping']['perform']:
                        if fig_seg is not None:
                            fig_seg.tight_layout(rect=[0, 0, 1, 0.985])
                            fig_seg.suptitle(filename, y=0.995)
                            canvas = FigureCanvas(fig_seg)
                            canvas.draw()
                            fig_seg_frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
                            fig_seg_frame = fig_seg_frame.reshape(canvas.get_width_height()[::-1] + (4,))
                            fig_seg_frame = fig_seg_frame[:, :, :3]  # Drop alpha if needed
                            plt.close(fig_seg)
                            fig_seg = fig_seg_frame

                        if fig_phase is not None:
                            fig_phase.tight_layout(rect=[0, 0, 1, 0.985])
                            fig_phase.suptitle(filename, y=0.995)
                            canvas = FigureCanvas(fig_phase)
                            canvas.draw()
                            fig_phase_frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
                            fig_phase_frame = fig_phase_frame.reshape(canvas.get_width_height()[::-1] + (4,))
                            fig_phase_frame = fig_phase_frame[:, :, :3]  # Drop alpha if needed
                            plt.close(fig_phase)
                            fig_phase = fig_phase_frame

                        max_change = self.params.image_segmentation['otsu_thresholding']['dynamic_adjust']['max_change']
                        remasking = int(output_changed/max_change)

                        print(f'\r Phase unwrapping took {time.time() - startprocess:.2f} sec for {filename} with {remasking} times remasking and {dev:.1f}% change', 
                              end='', flush = remasking == 0 )

                    fig_output.append(fig_seg)
                    fig_output.append(fig_phase)

                else:
                    phas_data = np.full(self.FOV, np.nan, dtype=self.params.data_precision['phase_processing'])
                    fig_output = [None,None]
     
            else:
                phas_data = optional_output['hf_phase']
                fig_output = [None,None]
                tissue_mask = 0
                
            return {'phase_data':phas_data,'fig_output':fig_output,'reference_area':np.sum(tissue_mask)}

        def intensity_projection(filename,propagation_distance=None):
            hologram = np.squeeze(
                hologram_load(
                    filename=filename,
                    directory=self.params.hologram_stack_dir,
                    dtype=phase_processing_dtype
                )[0]
            )

            # Step 1: Phase extraction via FFT filtering
            phase_wrapped, optional_output = FFT_filtering(
                hologram=hologram,
                pixelsize=self.params.pixelsizes,
                wavelength=self.params.wavelength,
                propagation_distance=propagation_distance,
                reference_hologram=reference_hologram,
                dtype = phase_processing_dtype,
                **self.params.FFT_filtering
            )

            intensity = optional_output['intensity']
            del hologram, optional_output, phase_wrapped
            gc.collect()

            return intensity

        def out_of_focous_correction():
            
            def refocousing_parallel(ind):

                theta_ind = self.job_inputs[ind]
                theta_i = checking_angles[theta_ind]

                opt_range = self.params.phase_processing['propagation']['check_range']
                opt_range = np.round(opt_range, 10)

                # Prepare output filename
                filename = f"angleindex_{theta_ind}_theta_{theta_i}.tiff"
                tiff_stack_path = os.path.join(save_dir, filename)

                tiff_stack_raw = []

                # First pass: collect all raw phase maps
                for shift in opt_range:
                    print(f"\r    Refocusing optimization at {theta_i}° shift {shift} um .....", end='', flush=True)
                    phase = process_projection(f"Angle-{theta_i:.3f}.tiff", propagation_distance=shift * 1e-3).astype(np.float32)
                    tiff_stack_raw.append(phase)

                # Compute global min and max for normalization
                all_phases = np.array(tiff_stack_raw)
                global_min = np.nanmin(all_phases)
                global_max = np.nanmax(all_phases)

                # Second pass: normalize and convert to 8-bit images
                tiff_stack_norm = []
                for phase in tiff_stack_raw:
                    phase_norm = (phase - global_min) / (global_max - global_min + 1e-8)
                    phase_norm = np.clip(phase_norm, 0, 1)
                    phase_8bit = (phase_norm * 255).astype(np.uint8)
                    tiff_stack_norm.append(phase_8bit)

                # Save stack as 8-bit grayscale TIFF
                tiff.imwrite(
                    tiff_stack_path,
                    np.array(tiff_stack_norm),
                    photometric='minisblack',
                    compression=self.params.phase_processing['propagation']['compression']
                )
                del tiff_stack_raw, tiff_stack_norm

            print(" Correcting out-of-focous projections .....")

            number_checking_angles = self.params.phase_processing['propagation']['number_checking_angles']
            target_angles = np.linspace(0, 179, number_checking_angles)
            checking_angles = self.collected_angles[np.abs(self.collected_angles[:, None] - target_angles).argmin(axis=0)]

            shift = self.params.phase_processing['propagation']['shift']
            if shift is None:

                save_dir = os.path.join(self.params.processing_dir, "manual assessment",f"refocousing")
                os.makedirs(save_dir, exist_ok=True)
                self.params.phase_processing['Unwrapping']['check_unwrapping']['perform'] = False

                inputs = np.arange(number_checking_angles,dtype=int)
                chunk_size = min(ncpus,len(inputs))
                Njobs = len(inputs) // chunk_size if len(inputs) % chunk_size == 0 else len(inputs) // chunk_size + 1

                for job_index in range(0, Njobs):

                    # Define the input range for this job
                    start_idx = job_index * chunk_size
                    end_idx = start_idx + chunk_size if job_index < Njobs else len(inputs)-1
                    self.job_inputs = inputs[start_idx:end_idx]

                    # Process the current chunk
                    if ncpus == 1:
                        for ind in range(len(self.job_inputs)):
                            refocousing_parallel(ind)
                            gc.collect()
                    else:
                        if np.shape(self.job_inputs)[0] > 0:
                            Parallel(n_jobs=ncpus)(delayed(refocousing_parallel)(ind) for ind in range(np.shape(self.job_inputs)[0]))
                            gc.collect()

                print(f"\n   Please manually determine the propagation distances from the TIFF stack in:\n   {save_dir}\n   then rerun to continue phase measurement")

            else:

                theta_fit = np.array(checking_angles, dtype=float)       # Angles in degrees
                z_fit = np.copy(shift)                                    # Measured shift values in Z (pixels)
                
                # Fit the model to data
                popt, _ = curve_fit(cosine_shift_fun, theta_fit, z_fit)

                # Compute fitted Z shifts for all angles
                self.propagation_distance = cosine_shift_fun(self.collected_angles, *popt)

                # --- PLOTTING ---
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(theta_fit, z_fit, color='blue', label='Measured ΔZ')
                ax.plot(self.collected_angles, self.propagation_distance, color='red', label='Fitted ΔZ', linewidth=2)
                ax.set_xlabel('Angle (degrees)')
                ax.set_ylabel('ΔZ (pixels)')
                ax.set_title('Out-of-Focus Shift in Z (Refocusing Model)')
                ax.legend()
                ax.grid(True)

                # Save figure
                save_path = os.path.join(self.params.processing_dir, "phase figures")
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, "refocusing_optimization.png"))
                plt.close()

                np.save("propagation_dictance.npy",self.propagation_dictance)

        def save_sinograms():
            
            """
            Save sinogram slices chunked by Y-axis using Zarr format with adaptive chunk size fallback.
            """
            save_start = time.time()

            Yaxis_nchunks = self.params.phase_processing['Yaxis_nchunks']

            for yci in range(Yaxis_nchunks):
                chunk_length = int(np.ceil(Ylim / Yaxis_nchunks))
                Yi = int(yci * chunk_length)
                Ye = int(min((yci + 1) * chunk_length, Ylim))

                if self.params.phase_processing['save_on_cluster']:
                    zarr_path = os.path.join(self.script_dir, f"sinogram_stack_chunk({yci})-dim({Yi}-{Ye})")
                else:
                    zarr_path = os.path.join(self.params.processing_dir, f"sinogram_stack_chunk({yci})-dim({Yi}-{Ye})")
                    
                shape = (int(Ye - Yi), int(Xlim), int(Ntheta))

                if os.path.exists(zarr_path):
                    sinogram_zarr = zarr.open(zarr_path, mode='r+')
                        
                else:
                    if self.safe_chunk is None:

                        fallback_lengths = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
                        for chunk_len_ind in range(len(fallback_lengths)):
                            chunk_len = fallback_lengths[chunk_len_ind]
                            if chunk_len <= int(Ye - Yi):
                                try:
                                    sinogram_stack = zarr.open(
                                        zarr_path,
                                        mode='w',
                                        shape=shape,
                                        dtype=sinogram_dtype,
                                        chunks=(int(chunk_len), int(Xlim), int(self.theta_e - self.theta_i)),
                                        zarr_format=2
                                    )

                                    sinogram_data = np.transpose(self.phase_stack[:, Yi:Ye, :], (1, 2, 0))
                                    sinogram_stack[:, :, self.theta_i:self.theta_e] = sinogram_data

                                    self.safe_chunk = fallback_lengths[min(chunk_len_ind + 1, len(fallback_lengths)-1)]
                                    print(f"      Warning: {self.safe_chunk} bit is selected as the safe chunk length for saving sinogram")

                                    del sinogram_data, sinogram_stack
                                    gc.collect()
                                    break

                                except:
                                    continue

                    sinogram_zarr = zarr.open(
                        zarr_path,
                        mode='w',
                        shape=shape,
                        dtype=sinogram_dtype,
                        chunks=(int(self.safe_chunk), int(Xlim), int(self.theta_e - self.theta_i)),
                        zarr_format=2
                    )

                time_elapsed = time.time() - save_start
                print(f"\r      Sinogram Saving. (progress: {(yci/Yaxis_nchunks):.2f} % Elapsed time: {time_elapsed/60:.1f} min)", end='', flush=True)
                    
                sinogram_data = np.transpose(self.phase_stack[:, Yi:Ye, :], (1, 2, 0))
                sinogram_zarr[:, :, self.theta_i:self.theta_e] = sinogram_data

                del sinogram_data
                gc.collect()
                        
            print(f"\r      Saving Sinograms took: {(time.time() - quant_start)/60:.1f} min                                                    ")
            del self.phase_stack
            gc.collect()
                

            #         del sinogram_data
            #         gc.collect()

            #             sinogram_zarr = zarr.open(
            #                 zarr_path,
            #                 mode='w',
            #                 shape=shape,
            #                 dtype=sinogram_dtype,
            #                 chunks=chunks,
            #                 zarr_format=2
            #             )

            # del self.phase_stack
            # gc.collect()
            # time_elapsed = time.time() - save_start
            # print(f"\n Saving sinogram took: {time_elapsed/60:.1f} min")


            # for yci in range(Yaxis_nchunks):
            #     chunk_length = int(np.ceil(Ylim / Yaxis_nchunks))
            #     Yi = int(yci * chunk_length)
            #     Ye = int(min((yci + 1) * chunk_length, Ylim))

            #     zarr_path = os.path.join(self.params.processing_dir, f"sinogram_stack_chunk({yci})-dim({Yi}-{Ye})")

            #     shape = (int(Ye - Yi), int(Xlim), int(Ntheta))
            #     chunks = (int(Ye - Yi), int(Xlim), int(theta_e - theta_i))

            #     if os.path.exists(zarr_path):
            #         sinogram_stack = zarr.open(zarr_path, mode='r+')
            #     else:
            #         sinogram_stack = zarr.open(
            #             zarr_path,
            #             mode='w',
            #             shape=shape,
            #             dtype=sinogram_dtype,
            #             chunks=chunks,
            #             zarr_format=2 
            #         )

            #     # Transpose to (Y, X, θ)
            #     sinogram_chunk = np.transpose(sector_phase_data[:, Yi:Ye, :], (1, 2, 0))
            #     sinogram_stack[:, :, theta_i:theta_e] = sinogram_chunk

            # del sector_phase_data
            # gc.collect()

        def phase_verification():

            verify_start = time.time()

            min_corrs = self.params.phase_processing['Unwrapping']['verification']['min_corr']
            verify = self.params.phase_processing['Unwrapping']['verification']['perform']
            if not isinstance(min_corrs, (list, tuple)):
                min_corrs = list(min_corrs)

            if self.reference_projections is None:
                self.reference_projections = []
                self.phase_verification = np.ones((Ntheta,len(min_corrs)), dtype=bool)
                self.phase_correlations = np.ones((Ntheta,len(min_corrs)), dtype=np.float16)

            if verify:
                for prj_ind in range(self.theta_e-self.theta_i):
                    theta_ind = self.theta_i + prj_ind
                    phase = np.squeeze(self.phase_stack[prj_ind,:,:])
                    for tresh_ind in range(len(min_corrs)):
                        try:
                            phase_ref = self.reference_projections[tresh_ind]
                            corr_coef = np.corrcoef(phase.flatten(), phase_ref.flatten())[0, 1]
                            self.phase_correlations[theta_ind,tresh_ind] = corr_coef
                            if corr_coef > min_corrs[tresh_ind]:
                                self.phase_verification[theta_ind,tresh_ind] = True
                                self.reference_projections[tresh_ind] = phase
                            else:
                                self.phase_verification[theta_ind,tresh_ind] = False
                        except: 
                            self.reference_projections.append(phase)
                            self.phase_verification[theta_ind,tresh_ind] = False
                            self.phase_correlations[theta_ind,tresh_ind] = np.nan

                    progress = theta_ind / np.shape(self.phase_stack)[0] * 100
                    time_elapsed = time.time() - verify_start
                    print(f"\r      Verification Stage. (progress: {progress:.2f} % Elapsed time: {time_elapsed/60:.1f} min)", end='', flush=True)

                time_elapsed = time.time() - verify_start
                print(f"\r      Verification Stage took: {time_elapsed/60:.1f} min                                                    ")

        def dynamic_segmentation():
            
            print(' Preparing dynamis segmentation.')
            self.params.image_segmentation['otsu_thresholding']['dynamic_adjust']['reference_area'] = None
            hologram = np.squeeze(
                hologram_load(
                    filename=f"Angle-{self.collected_angles[0]:.3f}.tiff",
                    directory=self.params.hologram_stack_dir,
                    dtype=phase_processing_dtype
                )[0]
            )

            phase_wrapped, optional_output = FFT_filtering(
                hologram=hologram,
                pixelsize=self.params.pixelsizes,
                wavelength=self.params.wavelength,
                reference_hologram=reference_hologram,
                dtype = phase_processing_dtype,
                **self.params.FFT_filtering
            )
            del hologram
            gc.collect()

            image_seg = self.params.phase_processing['Unwrapping']['segmentation']

            if image_seg == 'intensity': image = optional_output['intensity']
            elif image_seg == 'hf_phase': image = optional_output['hf_phase']
            elif image_seg == 'unwrapped_Scikit': image = unwrap_phase(phase_wrapped)
            elif image_seg == 'unwrapped_Weighted': image = unwrap_phase_weighted(phase_wrapped)
            elif image_seg == 'wrapped_phase': image = phase_wrapped
            
            referance_area, mask = image_segmentation(
                image = image,
                pixelsize=self.params.pixelsizes,
                **self.params.image_segmentation
            )
            self.params.image_segmentation['otsu_thresholding']['dynamic_adjust']['reference_area'] = referance_area
            del referance_area

        print("\n Phase processing ----------------------------------------------------------")

    # preprocessing ------------------------------------------------------------------
        
        self.propagation_dictance = None
        self.misalignment = None
        self.finished_phase_processing = False

        Yaxis_nchunks = self.params.phase_processing['Yaxis_nchunks']
        sinogram_dtype = self.params.data_precision['saving_sinogram_data']
        phase_processing_dtype = self.params.data_precision['phase_processing']
        ncpus = self.params.phase_processing['n_cpus']
        if ncpus is None: ncpus = multiprocessing.cpu_count()
        if platform.system() == 'Linux' and ncpus > 1:
            matplotlib.use('Agg')

        # get tomogram data from the collected holograms
        try:
            data = np.load(os.path.join(self.params.processing_dir, "tomogram_data.npz"))
            self.FOV = data["FOV"]
            self.collected_angles = data["collected_angles"]
        except:
            get_tomogram_data()
        Ntheta = len(self.collected_angles)
        Ylim, Xlim = self.FOV

        # load reference experiment
        reference_hologram = np.squeeze(hologram_load(filename=self.params.phase_processing['reference_experiment'],
                                directory=self.params.hologram_stack_dir)[0])
        if self.params.image_segmentation['otsu_thresholding']['dynamic_adjust']['perform']:
            dynamic_segmentation()

    # processing ----------------------------------------------------------------------------------------

        if self.params.phase_processing['Unwrapping']['check_unwrapping']['perform']:
            # test phase processing (Optional) -----------------
            test_processing()
            
        else:

            # correcting out-of focous (optional) -----------------
            if self.params.phase_processing['propagation']['perform']:
                out_of_focous_correction()
            else:
                self.propagation_dictance = np.zeros((Ntheta,1))
                np.save("propagation_dictance.npy",self.propagation_dictance)

            # phase processing ------------------------------
            if self.propagation_dictance is not None:
                
                print(f" ℹ️  Info: Pase processing will be performed usin {phase_processing_dtype} precision.")
                print(f" ℹ️  Info: Parallel processing enabled across {ncpus} CPU core(s).")
                print(f" ℹ️  Info: Sinogram stack will be saved in {Yaxis_nchunks} files with {sinogram_dtype} precision.")
                
                # looping through the determined angular ranges process phase data and save sinograms
                processing_start_time = time.time()
                self.reference_projections = None
                self.safe_chunk = None

                # looping through the determined angular ranges process phase data and save sinograms
                theta_ranges = self.params.phase_processing['theta_range']

                for delta_theta_ind in range(len(theta_ranges)-1):

                    quant_start = time.time()
                    delta_theta = list(theta_ranges[delta_theta_ind:delta_theta_ind+2])
                    self.theta_i = np.abs(self.collected_angles - delta_theta[0]).argmin()
                    self.theta_e = np.abs(self.collected_angles - delta_theta[1]).argmin()
                    print(f"\r Phase procesing for the range {delta_theta[0]} - {delta_theta[1]} deg")

                    inputs = np.arange(self.theta_i,self.theta_e)

                    self.phase_stack = np.empty((len(inputs),Ylim, Xlim), dtype=phase_processing_dtype)
                    chunk_size = min(ncpus,len(inputs))
                    Njobs = len(inputs) // chunk_size if len(inputs) % chunk_size == 0 else len(inputs) // chunk_size + 1

                    for job_index in range(0, Njobs):

                        # Define the input range for this job
                        start_idx = job_index * chunk_size
                        end_idx = start_idx + chunk_size if job_index < Njobs - 1 else len(inputs)
                        self.job_inputs = inputs[start_idx:end_idx]

                        progress = start_idx / len(inputs) * 100
                        time_elapsed = time.time() - quant_start
                        print(f"\r      Quantification Stage. (progress: {progress:.2f} % Elapsed time: {time_elapsed/60:.1f} min)", end='', flush=True)

                        # Process the current chunk
                        reference_area = []
                        phase_data = []
                        if ncpus == 1:
                            job_results = []
                            for ind in range(len(self.job_inputs)):
                                job_results.append(parallel_processing(ind))  
                        else:
                            if np.shape(self.job_inputs)[0] > 0:
                                job_results = Parallel(n_jobs=ncpus)(delayed(parallel_processing)(ind) for ind in range(np.shape(self.job_inputs)[0]))

                        for job_ind in range(len(job_results)):
                            phase_data.append(job_results[job_ind]['phase_data'])
                            reference_area.append(job_results[job_ind]['reference_area'])

                        self.phase_stack[start_idx:end_idx,:,:] = np.array(phase_data)
                        self.params.image_segmentation['otsu_thresholding']['dynamic_adjust']['reference_area'] = np.nanmean(np.array(reference_area))
                        del job_results, phase_data, reference_area
                        gc.collect()

                        progress = end_idx / len(inputs) * 100
                        time_elapsed = time.time() - quant_start
                        print(f"\r      Quantification Stage. (progress: {progress:.2f} % Elapsed time: {time_elapsed/60:.1f} min)", end='', flush=True)

                    print(f"\r      Quantification Stage took: {(time.time() - quant_start)/60:.1f} min                                                    ")

                    # phase verification and discard the unreliable unwrapped phase
                    phase_verification()
                    
                    # phase verification and discard the unreliable unwrapped phase
                    save_sinograms()

                print(f"\n Phase procesing took: {(time.time() - processing_start_time)/3600:.2f} hours")
                np.savez(os.path.join(self.params.processing_dir, "phase_verification.npz"), phase_verification=self.phase_verification, phase_correlations=self.phase_correlations)
                self.finished_phase_processing = True
             
    def phase_recosntruction(self):

        def get_tomogram_data():
            
            print(" Getting tomogram data")
            # Step 1: Navigate to the directory
            directory = self.params.hologram_stack_dir
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directory '{directory}' does not exist.")
            
            # Step 2: List all .tiff files containing the word "Angle"
            files = [f for f in os.listdir(directory) if f.endswith('.tiff') and "Angle" in f]
            
            # Step 3: Updated regex pattern to extract angle values with exactly 3 decimal places
            angle_pattern = re.compile(r"Angle-([0-9]+\.[0-9]{3})\.tiff$")
            collected_angles = []
            FOV = None

            for file in files:
                if FOV is None:
                    Ylim, Xlim = np.shape(hologram_load(filename=file, directory=self.params.hologram_stack_dir)[0])
                    FOV = [Ylim, Xlim]

                match = angle_pattern.search(file)
                if match:
                    try:
                        # Convert the extracted value to float
                        angle = float(match.group(1))
                        collected_angles.append(angle)
                    except ValueError:
                        print(f"Warning: Could not convert angle in file '{file}' to float.")
                else:
                    print(f"No match for file: {file}")
            
            # Step 4: Convert to numpy array and sort in ascending order
            collected_angles = np.array(collected_angles)
            collected_angles.sort()
             
            self.FOV = np.array(FOV)
            self.collected_angles = collected_angles

            np.savez(os.path.join(self.params.processing_dir, "tomogram_data.npz"), FOV=self.FOV, collected_angles=self.collected_angles)

        def astra_FBP(sinogram,angles,filter_type):

            import astra
            
            Nangles, Xdimen = sinogram.shape
            assert len(angles) == Nangles, "Mismatch between number of angles and sinogram rows"

            proj_geom = astra.create_proj_geom('parallel', 1.0, Xdimen, angles)
            vol_geom = astra.create_vol_geom(Xdimen, Xdimen)
            
            sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
            recon_id = astra.data2d.create('-vol', vol_geom)

            cfg = astra.astra_dict('FBP_CUDA')
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = recon_id
            cfg['option'] = {'FilterType': filter_type}  # 'Ram-Lak', 'Shepp-Logan', 'Cosine', 'Hann', 'Hamming'

            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)

            phase_reconstructed = astra.data2d.get(recon_id)

            # --- Cleanup ---
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(sino_id)
            astra.data2d.delete(recon_id)

            return phase_reconstructed

        def threeD_reconstruction(sino,down_sampling,sino_filter):

            gpu = self.params.phase_recosntruction['speed_up']['gpu']
            sino = sino.astype(phase_reconst_dtype)
            approach = self.params.phase_recosntruction['transformation']['approach']
            settings = self.params.phase_recosntruction['transformation']['settings']

            Xdimension = sino.shape[0]
            delta_n = np.full((Xdimension, Xdimension), np.nan, dtype=phase_reconst_dtype)

            # Optional sinogram filtering
            if sino_filter['median_filter']['perform']:
                sino = median_filter(sino, size=tuple(sino_filter['median_filter']['sigma']))

            # discard unreliable colomns in the sinogram
            sino = sino[:,self.reliable_unwrapping_mask]
            reliable_angles = self.collected_angles[self.reliable_unwrapping_mask]

            # Determine Angle Range
            if settings['Rotation'] == 'Half':
                half_rotation_ind = np.argmin(np.abs(reliable_angles - 180))
                sino = sino[:,0:half_rotation_ind]
                reliable_angles = reliable_angles[0:half_rotation_ind]

            # Downsampling projections
            if down_sampling is not None:
                Num_projections = down_sampling
                theta_min, theta_max = np.nanmin(reliable_angles), np.nanmax(reliable_angles)
                Theta_projections = np.linspace(theta_min, theta_max, Num_projections)
                interp_func = interp1d(
                    reliable_angles,
                    sino,
                    kind='cubic',
                    axis=1,
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                sino = interp_func(Theta_projections)
            else:
                Theta_projections = reliable_angles

            if approach == 'FBP':
                if gpu:
                    phase_reconstructed = astra_FBP(
                        sinogram=sino.T,
                        angles=np.deg2rad(Theta_projections),
                        filter_type=settings['FBP_filter']
                    )
                else:
                    phase_reconstructed = iradon(
                        sino,
                        theta=Theta_projections,
                        filter_name=settings['FBP_filter'],
                        interpolation=settings['FBP_interpolation'],
                        output_size=Xdimension,
                        circle=settings['FBP_circle']
                    )

            elif approach == 'SART':
                phase_reconstructed = iradon_sart(sino, theta=Theta_projections,
                                                relaxation=settings['SART_relaxation'])
                for _ in range(settings['SART_iteration'] - 1):
                    phase_reconstructed = iradon_sart(sino, theta=Theta_projections,
                                                    image=phase_reconstructed)

            # Phase to refractive index change
            delta_n = phase_reconstructed * self.params.wavelength / (2 * np.pi * self.params.pixelsizes[0])

            return delta_n
        
        def parallel_processing(ind):

            sino_seq = np.squeeze(minisinogram[ind,:,:])

            down_sampling = self.params.phase_recosntruction['speed_up']['down_sampling']
            Yaxis_coor = Yi+self.job_inputs[ind]

            # rolling sinogram to correct camera offset
            camera_offset = int(self.offset[Yaxis_coor])
            shifted_sino = np.roll(sino_seq, shift=-camera_offset, axis=0)
            shifted_sino[np.isnan(shifted_sino)] = 0

            # cropping sinogram to speed up process
            skip = False
            if crop_sino:
                Yaxis_i,Yaxis_e = self.cropping_regions[1,:]
                if Yaxis_coor < Yaxis_i or Yaxis_coor > Yaxis_e:
                    skip = True              
                else:
                    yi,ye = self.cropping_regions[0,:]
                    crop_sift = int(np.mean([yi,ye]) - Xlim/2)
                    shifted_sino = np.roll(shifted_sino, shift=crop_sift, axis=0)

            delta_RI = np.full((Xdim, Xdim), np.nan, dtype=phase_reconst_dtype)
            if not skip:
                sino_filter = self.params.phase_recosntruction['sinogram_smoothen']
                try:
                    delta_RI = threeD_reconstruction(shifted_sino,down_sampling,sino_filter)
                except Exception as e: 
                    print(f"\r Phase Reconstruction for {Yaxis_coor} Y axis. Error: {e}")

            return delta_RI

        def calculate_camera_offset():

            def offset_optimization(sinogram,opt_range,Yind):
                
                def offset_parallel_processing(offset):

                    """
                    Reconstruct phase image from sinogram shifted by the given offset.
                    """
                    down_sampling = self.params.phase_recosntruction['camera_offset']['down_sampling']
                    shifted_sino = np.roll(sinogram, shift=-offset, axis=0)
                    shifted_sino[np.isnan(shifted_sino)] = 0

                    sino_filter = self.params.phase_recosntruction['sinogram_smoothen']
                    delta_n = threeD_reconstruction(shifted_sino,down_sampling,sino_filter)

                    return delta_n
                
                # Allocate crossection volume
                num_offset = len(opt_range)
                Xdim = sinogram.shape[0]
                chunk_size = min(ncpus, num_offset)
                Njobs = (num_offset + chunk_size - 1) // chunk_size

                delta_n_crossections = np.empty((num_offset, Xdim, Xdim), dtype=np.float64)
                
                for job_index in range(Njobs):
                    start_idx = job_index * chunk_size
                    end_idx = min(start_idx + chunk_size, num_offset)
                    self.job_inputs = opt_range[start_idx:end_idx]

                    # Process reconstruction either sequentially or in parallel
                    if ncpus == 1:
                        job_results = [offset_parallel_processing(ind) for ind in self.job_inputs]
                    else:
                        job_results = Parallel(n_jobs=ncpus)(
                            delayed(offset_parallel_processing)(ind) for ind in self.job_inputs)

                    delta_n_crossections[start_idx:end_idx, :, :] = np.array(job_results)

                # Prepare output tiff stack
                global_min = np.nanmin(delta_n_crossections)
                global_max = np.nanmax(delta_n_crossections)
                delta_n_crossections = (delta_n_crossections - global_min) / (global_max - global_min + 1e-8)
                delta_n_crossections = np.clip(delta_n_crossections, 0, 1)
                delta_n_crossections = (delta_n_crossections * 255).astype(np.uint8)

                start = int(opt_range[0])
                stop = int(opt_range[-1])
                step = int(opt_range[1] - opt_range[0])
                tiff_filename = os.path.join(save_path, f"cross-section-{Yind:.2f}_opt_range-{start}-{stop}_step-{step}.tiff")

                tiff.imwrite(
                    tiff_filename,
                    delta_n_crossections,
                    imagej=True,
                    compression=self.params.phase_recosntruction['camera_offset']['compression'], 
                    photometric='minisblack',
                    bigtiff=True,  # Ensures we avoid 4GB limit
                )

                del delta_n_crossections

            selected_crossections = self.params.phase_recosntruction['camera_offset']['selected_crossections']
            optimization_range = self.params.phase_recosntruction['camera_offset']['optimization_range']
            initial_offset = self.params.phase_recosntruction['camera_offset']['initial_offset']
            estimated_offset = self.params.phase_recosntruction['camera_offset']['estimate_offset'] 

            self.offset = None
            save_path = os.path.join(self.params.processing_dir, "manual assessment","camera offset")
            os.makedirs(save_path, exist_ok=True)

            if initial_offset is not None and isinstance(initial_offset, (list, tuple)):

                # Step 4: calculating camera offset for all cross sections along Y by fitting over the corresponding values of selected crossections
                if len(initial_offset) > 1:
                    coeffs = np.polyfit(selected_crossections, initial_offset, 1)  # linear fit (degree 1)
                    self.offset = np.polyval(coeffs, np.linspace(0, 1, Ydim)).astype(int)
                else:
                    const_val = initial_offset
                    self.offset = np.full(Ydim, const_val).astype(int)
                
                np.save("camera_offset.npy",self.offset)
                print(" camera Offset is determined for all crossections along Y axis")

            else:
                if estimated_offset is None:

                    # Step 1: estimating camera offset using manuall cropping of the singoram of the middle cross-section (Y=0.5)
                    print("   Estimating camera offset using manual cropping")
                    try:
                        middle_sinogram = np.load(os.path.join(save_path, 'middle_crossesction_sinogram.npy'))
                    except:
                    # Step 0: Loading sinogram for estimating camera offset   
                        print(f"   Loading sinogram for estimating camera offset")
                        try:
                            middle_sinogram,_,_ = load_sinogram(ypixel=int(0.5*Ylim))
                            np.save(os.path.join(save_path, 'middle_crossesction_sinogram.npy'),middle_sinogram)
                        except:
                            raise FileNotFoundError("  No sinogram data found!!")
                        
                    plt.figure(figsize=(8, 6))
                    plt.imshow(middle_sinogram, aspect='auto')
                    plt.title("Click on two points to determine the top and the bottom of sinogram")

                    # Manually select two points (user clicks on the sinogram)
                    points = plt.ginput(2)  # Returns a list of (x, y) coordinates
                    plt.close()

                    # Extract y-coordinates (row indices)
                    y1, y2 = points[0][1], points[1][1]

                    # Calculate the vertical offset
                    sino_center = int(np.mean([y1, y2]))
                    
                    # Plot again with selected lines
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(middle_sinogram, aspect='auto')
                    ax.axhline(y1, color='r', linestyle='--')
                    ax.axhline(y2, color='r', linestyle='--')
                    ax.axhline(sino_center, color='y', linestyle='--')
                    ax.set_title("Selected lines for offset estimation")

                    fig.savefig(os.path.join(save_path, f"Offset_estimated.png"))
                    plt.close()

                    camera_center = middle_sinogram.shape[0] / 2
                    estimated_offset = int(sino_center - camera_center)

                    print(f"   Estimated Offset from manual cropping: {estimated_offset}")
                    print("   Offset Processing is not done yet. Run again to optimize the estimated offset")

                else:

                    if initial_offset is None:
                    # Step 2: optimizing the estimated camera offset at the middle cross-section (Y=0.5)
                        print("   Optimizing the estimated camera offset")

                        # optimization for the first time at the middle crossection
                        try:
                            middle_sinogram = np.load(os.path.join(save_path, 'middle_crossesction_sinogram.npy'))
                        except:   
                            print(f"   Loading sinogram for optimizing camera offset")
                            try:
                                middle_sinogram,_,_ = load_sinogram(ypixel=int(0.5*Ylim))
                                np.save(os.path.join(save_path, 'middle_crossesction_sinogram.npy'),middle_sinogram)
                            except:
                                raise FileNotFoundError("  No sinogram data found!!")
                        offset_optimization(middle_sinogram,optimization_range['estimated'],0.5)
                        print("\n Offset Processing is not done yet. Run again to check the consistency over Y axis")

                    else:
                    # Step 3: checking the consistency of the optimized offset over Y axis
                        print("   Checking the consistency of the optimized offset over Y axis")
                        yind = 0
                        for yslice in selected_crossections:

                            ypixel = int(yslice*(Ylim-1))

                            step='Loading sinogram .....'
                            print(f"\r    Crossection: {yind}/{len(selected_crossections)} at {ypixel} Yaxis, Step:{step}",end='',flush=True)
                            sino_yi,_,_ = load_sinogram(ypixel=ypixel)

                            step='Optimizing ...............'
                            print(f"\r    Crossection: {yind}/{len(selected_crossections)} at {ypixel} Yaxis, Step:{step}",end='',flush=True)
                            offset_optimization(sino_yi,optimization_range['consistency'],yslice)

                            yind += 1

        def load_sinogram(ychunk=None,ypixel=None):

            chunk_length = int(np.ceil(Ylim / Yaxis_nchunks))

            if ychunk is not None:
                Yi = int(ychunk * chunk_length)
                Ye = int(min((ychunk + 1) * chunk_length, Ylim))
                if self.params.phase_recosntruction['speed_up']['load_from_cluster']:
                    chunk_path = os.path.join(self.script_dir, f"sinogram_stack_chunk({ychunk})-dim({Yi}-{Ye})")
                else:
                    chunk_path = os.path.join(self.params.processing_dir, f"sinogram_stack_chunk({ychunk})-dim({Yi}-{Ye})")
                zarr_array = zarr.open(chunk_path, mode='r')
                sinogram = zarr_array[:]

            elif ypixel is not None: 
                for yci in range(Yaxis_nchunks):
                    Yi = int(yci * chunk_length)
                    Ye = int(min((yci + 1) * chunk_length, Ylim))
                    if ypixel >= Yi and ypixel < Ye:
                        ychunk = yci
                        break
                
                if self.params.phase_recosntruction['speed_up']['load_from_cluster']:
                    chunk_path = os.path.join(self.script_dir, f"sinogram_stack_chunk({yci})-dim({Yi}-{Ye})")
                else:
                    chunk_path = os.path.join(self.params.processing_dir, f"sinogram_stack_chunk({yci})-dim({Yi}-{Ye})")
                zarr_array = zarr.open(chunk_path, mode='r')
                chunk_data  = zarr_array[:]
                sinogram = np.squeeze(chunk_data[ypixel-Yi,:,:])

            return sinogram, Yi, Ye

        def crop_sinogram():

            self.max_prj_sinograms = []
            try:
                self.max_prj_sinograms = np.load(os.path.join(self.params.processing_dir, "max_prj_sinograms.npy"))
            except:
                print(f"\r   Creating max projection sinograms....")
                for yci in range(Yaxis_nchunks):
                    print(f"\r   Loading sinograms for Yaxis chunk: {yci}", end='', flush=True)

                    sinogram_chunk,_,_ = load_sinogram(ychunk=yci)
                    self.max_prj_sinograms.append(np.nanmean(sinogram_chunk,axis=0))

                    del sinogram_chunk
                    gc.collect()

                self.max_prj_sinograms = np.array(self.max_prj_sinograms)
                self.max_prj_sinograms = np.nanmean(self.max_prj_sinograms,axis=0)
                shift = int(np.nanmean(self.offset))
                self.max_prj_sinograms = np.roll(self.max_prj_sinograms, shift=-shift, axis=0)

                np.save(os.path.join(self.params.processing_dir, "max_prj_sinograms.npy"),self.max_prj_sinograms.astype(np.float32))

            if crop_sino:
                try:
                    self.cropping_regions = np.load(os.path.join(self.params.processing_dir, "cropped_sinograms_regions.npy"))
                except:
                    print(f"\r   Cropping the sinograms for speeding up the process")

                    self.cropping_regions = []
                    directory = os.path.join(self.params.processing_dir, "manual assessment","phase reconstruction")
                    os.makedirs(directory, exist_ok=True)
                        
                    plt.figure(figsize=(8, 6))
                    plt.imshow(self.max_prj_sinograms, aspect='auto')
                    plt.title("Click on two points to determine the top and the bottom of the processing region")

                    # Manually select two points (user clicks on the sinogram)
                    points = plt.ginput(2)  # Returns a list of (x, y) coordinates
                    plt.close()

                    # Extract y-coordinates (row indices)
                    y1, y2 = int(points[0][1]), int(points[1][1])
                    self.cropping_regions.append([y1, y2])

                    # Plot again with selected lines
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(self.max_prj_sinograms, aspect='auto')
                    ax.axhline(y1, color='r', linestyle='--')
                    ax.axhline(y2, color='r', linestyle='--')
                    ax.set_title(f"cropped region for 3d reconstruction between {y1} and {y2} Ypixels")

                    fig.savefig(os.path.join(directory, f"cropped_maxprjsinogram({y1}-{y2}).tiff"))
                    plt.close()

                    print(f"   Cropped region for max projection sinogram: ({y1},{y2})")

                    filename = f"Angle-{self.collected_angles[0]:.3f}.tiff"
                    hologram = np.squeeze(
                        hologram_load(filename=filename, directory=self.params.hologram_stack_dir, dtype=phase_reconst_dtype)[0])
                    reference_hologram = np.squeeze(hologram_load(filename=self.params.phase_processing['reference_experiment'],
                                            directory=self.params.hologram_stack_dir)[0])
                    phase_wrapped, optional_output = FFT_filtering( hologram=hologram, pixelsize=self.params.pixelsizes,
                        wavelength=self.params.wavelength, reference_hologram=reference_hologram, dtype=phase_reconst_dtype,
                        **self.params.FFT_filtering)
                    
                    plt.figure(figsize=(8, 6))
                    plt.imshow(phase_wrapped, aspect='auto')
                    plt.title("Click on two points to determine the top and the bottom of the processing region")

                    # Manually select two points (user clicks on the sinogram)
                    points = plt.ginput(2)  # Returns a list of (x, y) coordinates
                    plt.close()

                    # Extract y-coordinates (row indices)
                    y1, y2 = int(points[0][1]), int(points[1][1])
                    self.cropping_regions.append([y1, y2])

                    # Plot again with selected lines
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(phase_wrapped, aspect='auto')
                    ax.axhline(y1, color='r', linestyle='--')
                    ax.axhline(y2, color='r', linestyle='--')
                    ax.set_title(f"cropped region for 3d reconstruction between {y1} and {y2} Ypixels")

                    fig.savefig(os.path.join(directory, f"cropped_maxprjsinogram({y1}-{y2}).tiff"))
                    plt.close()

                    print(f"   Cropped region along Y axis: ({y1},{y2})")

                    self.cropping_regions = np.array(self.cropping_regions )
                    np.save(os.path.join(self.params.processing_dir, "cropped_sinograms_regions.npy"),self.cropping_regions)

        def save_reconstructed_volume(VolumeData):
            """
            Save reconstructed 3D volume chunk to Zarr v3 format with adaptive chunk size fallback.
            """
            if self.params.phase_recosntruction['speed_up']['save_on_cluster']:
                zarr_path = os.path.join(self.script_dir,f"delta_n_chunk({yci})-dim({Yi}-{Ye})")
            else:
                zarr_path = os.path.join(self.params.processing_dir,f"delta_n_chunk({yci})-dim({Yi}-{Ye})")

            shape = np.shape(VolumeData)
            chunks = shape

            print(f"\r   Phase Reconstruction of chunk {yci} from {Yi+inputs[0]} to {Yi+inputs[-1]} Yaxis: Saving in {chunks[0]} sub-chunks .........................", end='', flush=True)
            try:
                # Try full-size chunk first
                if os.path.exists(zarr_path):
                    volume_stack = zarr.open(zarr_path, mode='r+')
                else:
                    volume_stack = zarr.open(
                        zarr_path,
                        mode='w',
                        shape=shape,
                        dtype=saved_deltan_dtype,
                        chunks=chunks,
                    )

                volume_stack[:] = VolumeData.astype(saved_deltan_dtype)

            except Exception as e:

                fallback_lengths = [256,128, 64, 32, 16, 8, 4, 2, 1]
                for chunk_len in fallback_lengths:
                    try:
                        safe_chunks = (min(chunk_len, shape[0]), shape[1], shape[2])
                        volume_stack = zarr.open(
                            zarr_path,
                            mode='w',
                            shape=shape,
                            dtype=saved_deltan_dtype,
                            chunks=safe_chunks,
                        )

                        print(f"\r   Phase Reconstruction of chunk {yci} from {Yi+inputs[0]} to {Yi+inputs[-1]} Yaxis: Saving in {chunk_len} sub-chunks .........................", end='', flush=True)
                        volume_stack[:] = VolumeData.astype(saved_deltan_dtype)
                        break
                    except:
                        pass

            del VolumeData
            gc.collect()

    # preprocessing ------------------------------------------------------------------

        processing_start_time = time.time()
        self.finished_phase_reconstruction = False

        phase_reconst_dtype = self.params.data_precision['phase_reconstruction']
        saved_deltan_dtype = self.params.data_precision['saved_deltan_data']
        Yaxis_nchunks = self.params.phase_processing['Yaxis_nchunks']
        Yaxis_chunk_skip = self.params.phase_recosntruction['Yaxis_chunk_skip']
        crop_sino = self.params.phase_recosntruction['speed_up']['cropping_sinogram']
        down_sampling = self.params.phase_recosntruction['speed_up']['down_sampling']
        sinogram_smoothen = self.params.phase_recosntruction['sinogram_smoothen']
        gpu = self.params.phase_recosntruction['speed_up']['gpu']
        min_corrs = self.params.phase_processing['Unwrapping']['verification']['min_corr']
        minimum_correlation = self.params.phase_recosntruction['min_correlation']

        # get tomogram data from the collected holograms
        try:
            data = np.load(os.path.join(self.params.processing_dir, "tomogram_data.npz"))
            self.FOV = data["FOV"]
            self.collected_angles = data["collected_angles"]
        except:
            get_tomogram_data()
        Ntheta = len(self.collected_angles)
        Ylim, Xlim = self.FOV
        Ydim, Xdim = self.FOV

        ncpus = self.params.phase_recosntruction['n_cpus']
        if ncpus is None:
            ncpus = multiprocessing.cpu_count()

        verification_data = np.load(os.path.join(self.params.processing_dir, "phase_verification.npz"))
        self.phase_verification = verification_data['phase_verification']
        if minimum_correlation is None:
            if not isinstance(min_corrs, (list, tuple)):
                min_corrs = list(min_corrs)
            print_string = " ℹ️  Info: \n"
            for mi_cor_ind in range(len(min_corrs)):
                reliable_mask = np.squeeze(self.phase_verification[:, mi_cor_ind])
                discard_proj = np.sum(~reliable_mask)  # Count of discarded projections
                total_proj = reliable_mask.size
                print_string += f"   - For min correlation {min_corrs[mi_cor_ind]:.2f}: {discard_proj} / {total_proj} projections discarded\n"
            print(print_string)
            minimum_correlation = float(input(f"Please select one of the min correlation values {min_corrs}: "))
        
        selected_index = np.abs(np.array(min_corrs) - minimum_correlation).argmin()
        # Extract reliable mask for that threshold
        self.reliable_unwrapping_mask = np.squeeze(self.phase_verification[:, selected_index])

        print("\n Calculating camera offset ----------------------------------------------------------")
        calculate_camera_offset()

        if self.offset is not None:
            print("\n 3D Reconstructing Phase ----------------------------------------------------------")

            if crop_sino:
                print(" ⚠️  Warning: The open (empty) area will be cropped from sinograms to speed up processing.")
            if len(Yaxis_chunk_skip) > 0:
                print(f" ⚠️  Warning: Phase reconstruction will skip for the {Yaxis_chunk_skip} Y-axis chunks.")
            if self.params.phase_recosntruction['speed_up']['down_sampling'] is not None:
                print(f" ⚠️  Warning: Sinograms will be downsampled from {Ntheta} to {down_sampling} projections.")
            if sinogram_smoothen['median_filter']['perform']:
                print(f" ⚠️  Warning: Sinograms will be smoothened using median filter.")
            print(f" ℹ️  Info: Parallel processing enabled across {ncpus} CPU core(s).")
            if gpu:
                print(f" ℹ️  Info: GPU processing is enabled.")
            print(f" ℹ️  Info: 3D Reconstruction will be performed using {phase_reconst_dtype} precision.")
            print(f" ℹ️  Info: The RI ditribution data will be stored in {saved_deltan_dtype} precision.")

            # manually crop sinogram for speeding up the 3d reconstruction
            crop_sinogram()
            if crop_sino:
                Xi,Xe = self.cropping_regions[0,:]
                Xdim = Xe - Xi

            print("   Looping over the Yaxis chunks ....")

            # reconstructing for every chunk in y axis
            for yci in range(Yaxis_nchunks):
                
                if yci in Yaxis_chunk_skip:
                    print(f"\r   Phase Reconstruction is skipped for the {yci} Y-axis chunks")
                    continue

                sinogram , Yi, Ye = load_sinogram(ychunk=yci)

                # reconstructing data
                Ylength = np.shape(sinogram)[0]
                VolumeData = np.empty((Ylength,Xdim,Xdim), dtype=phase_reconst_dtype)

                inputs = np.arange(0,Ylength)
                chunk_size = min(ncpus,len(inputs))
                Njobs = len(inputs) // chunk_size if len(inputs) % chunk_size == 0 else len(inputs) // chunk_size + 1

                for job_index in range(0, Njobs):

                    # Define the input range for this job
                    start_idx = job_index * chunk_size
                    end_idx = start_idx + chunk_size if job_index < Njobs - 1 else len(inputs) - 1
                    self.job_inputs = inputs[start_idx:end_idx]
                    
                    if crop_sino:
                        minisinogram = np.squeeze(sinogram[inputs[start_idx]:inputs[end_idx],Xi:Xe,:])
                    else: minisinogram = np.squeeze(sinogram[inputs[start_idx]:inputs[end_idx],:,:])
                    if len(np.shape(minisinogram)) == 2:
                        minisinogram = np.array([minisinogram])

                    progress = start_idx / len(inputs) * 100
                    time_elapsed = time.time() - processing_start_time
                    print(f"\r   Phase Reconstruction of chunk {yci} from {Yi+inputs[0]} to {Yi+inputs[-1]} Yaxis: progress: {progress:.1f} % Elapsed time: {time_elapsed/60:.1f} min", end='', flush=True)

                    # Process the current chunk
                    if ncpus == 1:
                        job_results = []
                        for ind in range(len(self.job_inputs)):
                            job_results.append(parallel_processing(ind))
                    else:
                        job_results = Parallel(n_jobs=ncpus)(delayed(parallel_processing)(ind) for ind in range(len(self.job_inputs)))

                    if np.shape(self.job_inputs)[0] > 0:
                        VolumeData[start_idx:end_idx,:,:] = np.array(job_results)

                    progress = end_idx / len(inputs) * 100
                    print(f"\r   Phase Reconstruction of chunk {yci} from {Yi+inputs[0]} to {Yi+inputs[-1]} Yaxis: progress: {progress:.1f} % Elapsed time: {time_elapsed/60:.1f} min", end='', flush=True)

                VolumeData = VolumeData.astype(saved_deltan_dtype)
                save_reconstructed_volume(VolumeData)

            self.finished_phase_reconstruction = True
            print(f"Phase Reconstruction took: {(time.time() - processing_start_time)/3600:.2f} hours")

    def tiff_stack_reconstructed(self):

        def manual_tissue_cropping():
            
            max_projection = np.load(os.path.join(self.params.processing_dir, "max_prj_deltan.npy"))

            fig, ax = plt.subplots()
            im = ax.imshow(max_projection, cmap='gray')
            ax.set_xlabel(r'$X (px)$')
            ax.set_ylabel(r'$Y (px)$')
            plt.colorbar(im, ax=ax, orientation='vertical')

            ROI = {'x1': None, 'y1': None, 'x2': None, 'y2': None}

            def on_select(eclick, erelease):
                # Store the coordinates of the rectangle
                ROI['x1'], ROI['y1'] = int(eclick.xdata), int(eclick.ydata)
                ROI['x2'], ROI['y2'] = int(erelease.xdata), int(erelease.ydata)
                plt.close()  # Close the plot once selection is done

            # Initialize RectangleSelector
            rect_selector = widgets.RectangleSelector(ax, on_select, useblit=True,
                                            button=[1],  # Enable left mouse button
                                            minspanx=5, minspany=5, spancoords='pixels',
                                            interactive=True,
                                            props=dict(edgecolor='red', linewidth=2, facecolor='none'))

            plt.title("Draw a rectangle to determine the ROI for exporting tiff files")
            plt.show()
            
            return ROI
        
        def load_chunk_data(ychunk):

            chunk_length = int(np.ceil(Ylim / Yaxis_nchunks))

            Yi = int(ychunk * chunk_length)
            Ye = int(min((ychunk + 1) * chunk_length, Ylim))
            if self.params.tiff_export['load_from_cluster']:
                chunk_path = os.path.join(self.script_dir, f"delta_n_chunk({ychunk})-dim({Yi}-{Ye})")
            else:
                chunk_path = os.path.join(self.params.processing_dir, f"delta_n_chunk({ychunk})-dim({Yi}-{Ye})")
            zarr_array = zarr.open(chunk_path, mode='r')
            data_chunk = zarr_array[:]

            return data_chunk, Yi, Ye

        """
        export 3d reconstructed data as tiffstack

        """
        print("\n Exporting 3D data as tiff stacks ----------------------------------------------------------")

        start_time = time.time()
        warnings.filterwarnings("ignore")
        saved_deltan_data = self.params.data_precision['saved_deltan_data']

        Yaxis_nchunks = self.params.phase_processing['Yaxis_nchunks']
        Yaxis_chunk_save = self.params.tiff_export['Yaxis_chunk_save']
        binning = self.params.tiff_export['binning']
        max_tiff_size = self.params.tiff_export['max_tiff_size'] * (1024**3)
        crop_data = self.params.tiff_export['crop_data']

        # get tomogram data from the collected holograms
        data = np.load(os.path.join(self.params.processing_dir, "tomogram_data.npz"))
        self.FOV = data["FOV"]
        self.collected_angles = data["collected_angles"]
        Ylim, Xlim = self.FOV

        if crop_data:

            if platform.system() == 'Linux':
                try:
                    max_projection = np.load(os.path.join(self.params.processing_dir, "max_prj_deltan.npy"))
                    # cropping field of saving
                    try:
                        roi_path = os.path.join(self.params.processing_dir, "ROI_export.npy")
                        self.ROI_export = np.load(roi_path, allow_pickle=True).item()
                    except Exception as e:
                        print("📌 Determine the cropping first on Windows and re-run.")
                        return
                except:
                    # Loading 3D data
                    print(f"  Creating Max projection slice for cropping ......")
                    if Yaxis_chunk_save is None:
                        Yaxis_chunk_save = np.arange(Yaxis_nchunks)
                    max_prj_deltan = []
                    for yci in Yaxis_chunk_save:
                        print(f"\r   Loading Yaxis chunk: {yci}\n", end='', flush=True)
                        data_chunk, Yi, Ye = load_chunk_data(yci)
                        max_prj_deltan.append(np.nanmean(data_chunk,axis=0))
                        del data_chunk
                        gc.collect()
                    max_prj_deltan = np.nanmean(np.array(max_prj_deltan),axis=0)
                    np.save(os.path.join(self.params.processing_dir, "max_prj_deltan.npy"),max_prj_deltan) 

                    print(f"  Please rerun on Windows to determine the cropping region ......")     
                    return  
            else:
                print("🛠️  Cropping ROI on Windows...")
                ROI_export = manual_tissue_cropping()  # <-- Make sure this function returns a dict
                np.save(os.path.join(self.params.processing_dir, "ROI_export.npy"), ROI_export, allow_pickle=True)
                print("✅ ROI saved. Now re-run this script on the cluster to apply cropping.")
                return

        # Loading 3D data
        print(f"  Loading volumetric data ......")
        if Yaxis_chunk_save is None:
            Yaxis_chunk_save = np.arange(Yaxis_nchunks)
        volume_data = None
        for yci in Yaxis_chunk_save:
            print(f"\r   Loading Yaxis chunk: {yci}\n", end='', flush=True)
            data_chunk, Yi, Ye = load_chunk_data(yci)
            if volume_data is None:
                Xlim = np.shape(data_chunk)[1]
                volume_data = np.full((Ylim, Xlim, Xlim), np.nan, dtype=saved_deltan_data)
            volume_data[Yi:Ye, :, :] = data_chunk
            del data_chunk
            gc.collect()
        if crop_data:
            volume_data = volume_data[:, 
                                    self.ROI_export['y1']:self.ROI_export['y2'], 
                                    self.ROI_export['x1']:self.ROI_export['x2']]
            
        # removing nan slices
        valid_slices_mask = ~np.all(np.isnan(volume_data), axis=(1, 2))
        volume_data = volume_data[valid_slices_mask]

        # Normalizing 
        global_min = np.nanmin(volume_data)
        global_max = np.nanmax(volume_data)
        volume_data = (volume_data - global_min) / (global_max - global_min + 1e-8)
        volume_data = np.clip(volume_data, 0, 1)
        volume_data[np.isnan(volume_data)]=1
        volume_data = (volume_data * 255).astype(np.uint8)

        # binning 
        def exporttiff(binningwin,data):
            # Normalizing 
            data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-8)
            data = np.clip(data, 0, 1)
            data = (data * 255).astype(np.uint8)

            # Save the final compressed TIFF
            print(' Saving TIFF image...')
            # ==== DETERMINE NUMBER OF TIFF FILES ====
            slice_size = data.shape[1] * data.shape[2] * 1  # Single slice size (uint8)
            max_slices_per_tiff = max_tiff_size // slice_size  # Compute slices per TIFF (~4GB per file)
            num_tiff_files = int(np.ceil(data.shape[0] / max_slices_per_tiff))  # Compute number of TIFF files

            for i in range(num_tiff_files):
                start_idx = i * max_slices_per_tiff
                end_idx = min((i + 1) * max_slices_per_tiff, data.shape[0])

                tiff_filename = f"Deltan_clim({global_min:.5f}-{global_max:.5f})_binning({binningwin})_{i:02d}.tiff"
                tiff.imwrite(
                    tiff_filename,
                    data[start_idx:end_idx, :, :],  # Save slices as a single multi-page TIFF
                    imagej=True,
                    compression=None,  # More efficient for large stacks
                    photometric='minisblack',
                    bigtiff=True,  # Ensures we avoid 4GB limit
                    metadata={'axes': 'ZYX'}  # ← THIS LINE ENSURES ImageJ reads it as a Z-stack
                )

                file_size_gb = os.path.getsize(tiff_filename) / (1024 ** 3)
                print(f" {tiff_filename} is successfully saved with the size {file_size_gb:.2f} GB")


        elapsed = (time.time() - start_time)/60
        start_time = time.time()
        print(f" Loading took: {elapsed:.2f} min. Final size: {volume_data.nbytes / (1024 ** 3):.2f} GB")
        exporttiff(binningwin=1,data=volume_data)

        if binning > 1:
            print(f"  Binning: {binning} by {binning} ......")
            shape = np.shape(volume_data)
            new_shape = [int(dim / binning) for dim in shape]
            volume_data = transform.resize(volume_data, new_shape, anti_aliasing=True, preserve_range=True)
            elapsed = (time.time() - start_time)/60
            print(f" Binning took: {elapsed:.2f} min. Final size: {volume_data.nbytes / (1024 ** 3):.2f} GB")
            exporttiff(binningwin=binning,data=volume_data)

        # # Normalizing 
        # volume_data = (volume_data - np.nanmin(volume_data)) / (np.nanmax(volume_data) - np.nanmin(volume_data) + 1e-8)
        # volume_data = np.clip(volume_data, 0, 1)
        # volume_data = (volume_data * 255).astype(np.uint8)

        # elapsed = (time.time() - start_time)/60
        # print(f" Loading & Compression took: {elapsed:.2f} min. Final size: {volume_data.nbytes / (1024 ** 3):.2f} GB")

        # # Save the final compressed TIFF
        # print(' Saving TIFF image...')
        # # ==== DETERMINE NUMBER OF TIFF FILES ====
        # slice_size = volume_data.shape[1] * volume_data.shape[2] * 1  # Single slice size (uint8)
        # max_slices_per_tiff = max_tiff_size // slice_size  # Compute slices per TIFF (~4GB per file)
        # num_tiff_files = int(np.ceil(volume_data.shape[0] / max_slices_per_tiff))  # Compute number of TIFF files

        # for i in range(num_tiff_files):
        #     start_idx = i * max_slices_per_tiff
        #     end_idx = min((i + 1) * max_slices_per_tiff, volume_data.shape[0])

        #     tiff_filename = f"Deltan_clim({global_min:.5f}-{global_max:.5f})_binning({binning})_{i:02d}.tiff"
        #     tiff.imwrite(
        #         tiff_filename,
        #         volume_data[start_idx:end_idx, :, :],  # Save slices as a single multi-page TIFF
        #         imagej=True,
        #         compression=None,  # More efficient for large stacks
        #         photometric='minisblack',
        #         bigtiff=True,  # Ensures we avoid 4GB limit
        #         metadata={'axes': 'ZYX'}  # ← THIS LINE ENSURES ImageJ reads it as a Z-stack
        #     )

        #     file_size_gb = os.path.getsize(tiff_filename) / (1024 ** 3)
        #     print(f" {tiff_filename} is successfully saved with the size {file_size_gb:.2f} GB")
 