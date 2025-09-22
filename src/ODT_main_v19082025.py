import os, re
import numpy as np
import tifffile as tiff
from skimage.transform import resize
from matplotlib import pyplot as plt
from skimage.transform import iradon
import ODT_v19082025 as ODT
from skimage.restoration import unwrap_phase
import time
  
     
class Parameters:
 
    """
    Configuration class for phase data processing and segmentation.
    Each class variable contains settings used across different stages.
    Data types and formats are explicitly described for clarity.
    """
 
    # Directory path to hologram stack for input (str or None)
    processing_dir = os.path.join("..", "data")

    # Data types used in various stages of the pipeline
    data_precision = {
        'phase_processing': np.float32,  # Use float64 only if extremely high precision is needed 
        'saving_sinogram_data': np.float32,  # Use float16 only to reduce disk usage (less recommended)                           
        'phase_reconstruction': np.float32, # Avoid float16 here to preserve resolution and avoid aliasing artifacts.
                                            # Use float64 only if reconstruction artifacts arise or if you're doing advanced regularizatio
        'saved_deltan_data': np.float32,  # Use float16 only to reduce disk usage (less recommended)

        'export_tiff': np.uint8,  # Use uint16 if more dynamic range is needed
    }

    # Pixel dimensions in cm (converted from micrometers), shape: (2,) (np.ndarray of float) 0.0055/10=0.00055
    pixelsizes = np.array([0.000438, 0.000438])
    wavelength = 0.000632
    stage_stepsize = 1638.4 

    # Parameters for collecting the tomograms
    collect_data = {
        'perform': False,
        'precheck':True,
        'alignment': {
            'perform': True,
            'zoom_step' : 0.1, 
            'move_step' : 200,
            'overlay_ratio' : {'theta':0.5,'theta+pi':0.5,'live':0.5},
            'gap_width' : 250,
        },
        'theta_range' : np.array((0,360)),
        'theta_inc' : 0.5,
        'hologram_stack' : (1,1), # number of repeated holograms for the initial and the rest projections
        'stage_timeout' : (200,2),  # initial projection and the rest tomograms
        'hologram_check':{
            'perform':True,
            'maxstd_dev':1.5,
            },
        'reference_phase':{
            'stack':5,
            'cool_down':0.5,
            'maxstd':2.0,
        },
        'checking_figsize' : (800,800),
        }
    
    # Parameters for performing FFT filtering to obtain phase information from the collected hologram
    FFT_filtering = {
        'center_padding': 100,  # Padding in pixels (int)
        'sample_filter_size': 250,  # Filter size in mm^-1 (int)
        'first_order_peak':None,   # Position of first-order peak in pixels (y, x)
        'low_frequency_filter':25,  # Optional: low-pass filter size in mm^-1 (float or None)
        'filter_type':'Linear',
        'smoothing_sigma': 2,  # Sigma for Gaussian smoothing (float)
    }

    # Parameters for performing tissue segmentation from phase image
    image_segmentation = {
        'scale_factor': 0.25,  # Downscale factor for faster computation (float)
        'Fourier_filter_pass':(0,100), # Fourier high-pass filter
        'blur_kernel': (25, 25),  # Gaussian blur kernel size (tuple of 2 int)
        'median_blur_size': 25,  # Median filter size (odd int)
        'border_remove': 100,  # Pixels to remove at image borders (int)
        'otsu_thresholding': {
            'classes': 3,  # Number of Otsu classes (int)
            'output_class': 1,  # Index of the selected class (int)
            'output_adjust': 1.0,
            'dynamic_adjust':{'perform':True, # it dynamicly adjust the 'output_adjust' parameter respecting to the segmented tissue area of the previous projection
                              'max_dif':30,  # (percentage) the maximum allowed difference between the surface area of the tissue of the current projection and the previous one
                              'step':0.0025,  # the increament of changing the 'output_adjust' parameter
                              'max_change':0.02},  
            'filter': 'Max', # 'Max', 'midmax', 'High' or 'Low' filter out higher or lower values 
        },
        'morph_remove_obj': {
            'size': 50,  # Minimum object size to keep (int)
            'connectivity': 5  # Pixel connectivity for morphological filtering (int)
        },
        'laser_profile': {
            'off_center': (-50, 0),  # Beam center offset in pixels (tuple of 2 int)
            'radius': 2275  # Radius of illumination area in pixels (int)
        },
        'morph_openning': {
            'kernel_size': (15, 15),  # Kernel size for morphological opening (tuple of 2 int)
            'iterations': 1,  # Number of iterations (int)
            'max_distance':15, # None or int
        },
        'morph_closing': {
            'kernel_size': (1000, 250),  # Kernel size for morphological closing (tuple of 2 int)
            'iterations': 3,  # Number of iterations (int)
            'max_distance':None, # None or int
        },
        'segmentation_check': False,  # Enable plotting for segmentation steps (bool)
    }

    # Parameters for performing phase extraction and correction
    phase_processing = {
        'perform': True,  # Enable or disable phase processing (bool)
        'theta_range': np.arange(0, 361, 45),  # Angular range in degrees to save sinograms (np.ndarray of int)
        'reference_experiment': "Reference_phase_after.tiff",  # File path to hologram of reference experiment (str)
        'Yaxis_nchunks': 12, # devide y axis into chunks to save sinograms (int)
        'n_cpus':16,  # Number of CPU cores to use (int or None)
        'save_on_cluster':False,
        'Unwrapping':{
            'method': 'Weighted', # methos for unwrapping : 'Scikit'  'Weighted'
            'segmentation': 'hf_phase', #'intensity' 'hf_phase 'unwrapped_Scikit' 'unwrapped_Weighted' 'wrapped_phase' 
            'background_fit':{
                'method': {'wrapped':'polynomial','unwrapped':'polynomial'},  # 'polynomial' 'spline'
                'polynomial_deg':{'wrapped':1,'unwrapped':1},  #  int or None (None for not performing fitting)
                'spline_smoothing':{'wrapped':1e3,'unwrapped':1e3}, #  int or None (None for not performing fitting)
                'outlier_threshold':2,
                'max_iteration':2,
                'downsample_factor':1.0,
            },
            'verification': {
                'perform':True,
                'min_corr': (0.95,0.9),
            } ,
            'check_unwrapping': {'perform':True, # run on test_mode
                                 'unwrap_clim':(-10,50), # Enable plotting for unwrapping (bool)
                                 'test_prj':128},   # number of projections to test phase processing (int),            
        }, 
        'propagation' : {
            'perform' : False,
            'verbose': True,
            'compression' : None, # None(no compression) 'zlib'(lossless) 'jpeg'(lossy) 'lzma' or 'zstd',
            'number_checking_angles' : 16,
            'check_range': np.arange(-20, 20, 0.5), #um
            'shift':None # it should be a tupple with the length of number_checking_angles
        },
        }

    # Parameters for performing phase reconstrucion
    phase_recosntruction = {
        'perform': False,
        'Yaxis_chunk_skip':(), # the number of slices to skip (int) [note: 0 means no slice will be skipped]
        'n_cpus': 8,  # None or int, if is set to None maximum cpu will be used
        'min_correlation':0.9,
        'speed_up':{
            'cropping_sinogram':False,
            'down_sampling':None,
            'gpu': True,
            'save_on_cluster':False,
            'load_from_cluster':False,
        },
        'sinogram_smoothen':{
            'median_filter':{'perform':False,'sigma':[1,3]},
        },
        'transformation':{
            'approach': 'FBP', # 'FBP' or 'SART'
            'settings': {
                'Rotation':'Full', # 'Full':0:360 and 'Half':0:180
                'FBP_filter' : 'Hann',   
                 # CPU:  None or 'ramp' 'shepp-logan' 'cosine' 'hamming' 'hann' (Dfault:'ramp')
                 # GPU: 'Ram-Lak', 'Shepp-Logan', 'Cosine', 'Hann', 'Hamming'
                'FBP_interpolation' : 'cubic',
                 # CPU only: linear  nearest and cubic
                'FBP_circle':True,
                'SART_iteration' : 0,
                'SART_relaxation' : 0.8,
            }},
        'camera_offset' : {
            'estimate_offset': 0,
                # None: the offset will be estimated using manual cropping None
                # int: the optimization will be performed
            'initial_offset': [-46,-46,-46,-46],
                    # None: the optimization will be done on the estimated offset for only the middle crossection (Y = 0.5)
                    # int: the optimization will be done on all the selected crossections to see if the optimized offset is consistent along Y
                    # tupple of int: The offset for all the crossections along the Y axis will be determined by fitting over the selected crossections
            'selected_crossections' : np.linspace(0.2, 0.8, 4), 
            'optimization_range': {     # Interwall around the estimated offset to find the accurate offset
                'estimated': np.arange(-50,50,5).astype(int), # if the offset is estimated method: 'segmentation' or 'manual_cropping'
                'consistency': np.arange(-50,101,1).astype(int),  # to see if the offset is the same for all crossections (misalignment of Y axis)
                'subpixel' : "Later", # get subpixel precision
            },
            'down_sampling':None,
            'compression' : None, # None(no compression) 'zlib'(lossless) 'jpeg'(lossy) 'lzma' or 'zstd',
            },
        }
    
    # Parameters for exporting tiff files
    tiff_export = {
        'perform':False,
        'Yaxis_chunk_save':None, # None: all chunks or a list of chunks
        'crop_data':True,
        'load_from_cluster':False,
        'binning': 2, # reducing the number of pixels to export smaller tiff stacks
        'max_tiff_size' : 4,  # Max TIFF size in GB
    }

# ODT Collecting data -------------------------------------------------------------
if Parameters.collect_data['perform']:
    ODT_data = ODT.ODT_Collecting(Parameters)
    ODT_data.sample_alignment()
    # ODT_data.collect_reference_phase('before')
    ODT_data.collect_all_projections()
    ODT_data.collect_reference_phase('after')

# ODT Processing ------------------------------------------------------------------
ODT_processed = ODT.ODT_processing(Parameters) 
if Parameters.phase_processing['perform']:
    ODT_processed.phase_processing()

if Parameters.phase_recosntruction['perform'] and ODT_processed.finished_phase_processing:
    ODT_processed.phase_recosntruction()

if ODT_processed.params.tiff_export['perform'] and ODT_processed.finished_phase_reconstruction:
    ODT_processed.tiff_stack_reconstructed()
    
# %%%%%%%%% To submit a job:
# module avail python 
# module load python/3.9.19
# python -m venv env
# source env/bin/activate
# pip install -r requirements.txt --upgrade
# nice -n 32 python ODT_main_v19082025.py
# %%%%%%%%% to kill a running job
# ps -edalf | grep vkhandan | awk '$2=="R"'
# kill -9 <job_number(4th column)>