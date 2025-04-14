""" Script to run for generating HPF telluric models.

    This script takes in a path to a list of HPF spectra and:
        - Fits the water vapor content in each observation
        - Generates a full telluric model
        - Outputs new FITS file with telluric model extension added

    Created by DMK on 12/12/2024.
"""

##### Imports

# Standard library imports
import argparse
import copy
import datetime
import glob
import os
import pickle
import re

# Third party imports
from astropy.io import fits
from astropy.time import Time
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import tellurics_utils

##### Setup

### Input argument(s)
parser = argparse.ArgumentParser(description='Generating telluric model')
parser.add_argument('data_path', type=str, help='Path to data files')
parser.add_argument('--grid_name', type=str, default='hpf_lblrtm_hitran_202402', help='Name of the model grid to use')
parser.add_argument('--instrument', type=str, default='HPF', help='Instrument used for observations')
parser.add_argument('--output_dir', type=str, default='telluric_output', help='Path to place output at')
parser.add_argument('--make_plots', type=bool, default=True, help='Path to place output at')
parser.add_argument('--skip_existing', type=bool, default=True, help='Skip frames that already have telluric outputs')
args = parser.parse_args()

### Input/output: paths and files

# List of input data
input_file_names = np.sort(glob.glob(os.path.join(args.data_path, '*.fits')))

# Set up the output directory
output_dir = os.path.join(args.data_path, args.output_dir)
os.makedirs(output_dir, exist_ok=True)

# Set up plot output directory if needed
if args.make_plots:
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

# Telluric grid path (will be flexible in future)
telluric_model_grid_path = os.path.join('models/', args.grid_name)

### Read in reference data (LSFs, blazes)

## Blaze

# HPF Blaze files -- pre and post May 2022 maintenance
blaze_hpf_pre = np.load('data/hpf_blaze/quartz_blaze_spline_fit_pre_maintenance_15pix.npy')
blaze_hpf_post = np.load('data/hpf_blaze/quartz_blaze_spline_fit_post_maintenance.npy')

## LSF parameters

# Fiber to use
fiber = 'Sci'

# Read in the fit result files -- PSF fit results and orthogonal polynomial basis
psf_fit_results = pickle.load(open(f'data/hpf_lsf/PSF_Fit_Results_fiber_{fiber}.pkl', 'rb'))
ortho_poly_fit_results = np.load(f'data/hpf_lsf/HPF_PSF_{fiber}_OrthoP.npy', allow_pickle=True)

# The hat width parameter for the HPF LSF
op_hat_width = 2

### Set up for the water vapor fitting

# Set wavelength regions to use for fitting precipitable water vapor
fit_ranges_pwv = [ [ 8140, 8158 ],
                   [ 8941, 8956 ],
                   [ 9842, 9855 ],
                   [ 11942, 11966 ],
                   [ 12160, 12175 ] 
                   ]

# Fit padding -- wavelength padding in angstroms for model generation pre convolution
fit_padding_wave = 1.0

# Kernel width -- half width of convolution kernel in wavelength (Angstrom)
kernel_half_width_wave = 0.5

### Now run the telluric fitting and model generation

for i_file, file_name in enumerate(tqdm.tqdm(input_file_names)):

    ### File prep

    # Check if the output file exists
    output_file_name = os.path.join(output_dir, os.path.basename(file_name))
    if os.path.exists(output_file_name) and args.skip_existing:
        continue

    # Read the input data file in
    file_in = fits.open(file_name)

    # Get the file token
    file_token = re.split('[-_]', os.path.basename(file_name))[1]

    # Subtract the sky (estimation of the factor required to accurately subtract sky for HPF)
    flux_skysub = file_in['SCI FLUX'].data - 0.93 * file_in['SKY FLUX'].data

    # Choose the blaze file to use -- based on the date of observation
    if Time(file_in[0].header['DATE-OBS'], format = 'isot') < Time('2022-06-01T00:00:00', format = 'isot'):
        blaze_use = blaze_hpf_pre
    else:
        blaze_use = blaze_hpf_post

    ### Fit the water vapor value!

    pwv_fit_results = tellurics_utils.fit_pwv_hpf(file_in[7].data, flux_skysub, blaze_use,
                                                  fit_padding=fit_padding_wave,
                                                  kernel_half_width=kernel_half_width_wave,
                                                  lsf_type='gauss_poly_model',
                                                  fit_ranges=fit_ranges_pwv,
                                                  grid_path=telluric_model_grid_path,
                                                  lsf_fit_results=psf_fit_results,
                                                  ortho_poly=ortho_poly_fit_results)
    
    ## Make a plot showing the PWV fit if make_plots argument is true
    if args.make_plots:
        # Multi-page PDF plot: each page is a different PWV fit region
        with PdfPages(os.path.join(output_dir, 'plots', f'telluric_fit_{file_token}.pdf')) as pdf:
            for fit_range in fit_ranges_pwv:

                # Pick out where in the fit results output are the data for this range
                range_loc = np.where( (pwv_fit_results[1] > fit_range[0]) 
                                     & (pwv_fit_results[1] < fit_range[1]) )[0]

                plt.figure( figsize = ( 12.5, 6 ) )

                plt.plot( pwv_fit_results[1][range_loc], pwv_fit_results[2][range_loc], '.-', c = '#323232', ms = 7, lw = 0.75, label = 'Data' )
                plt.plot( pwv_fit_results[1][range_loc], pwv_fit_results[3][range_loc], '-', c = '#1c6ccc', lw = 2, label = 'Best Fit Model' )
                plt.plot( pwv_fit_results[1][range_loc], pwv_fit_results[2][range_loc] / pwv_fit_results[3][range_loc], '.-', c = '#bf3465', ms = 5, label = 'Corrected Data')

                # Title -- with S/N of the corrected spectrum in title
                snr = np.nanmean(pwv_fit_results[2][range_loc] / pwv_fit_results[3][range_loc]) / np.nanstd(pwv_fit_results[2][range_loc] / pwv_fit_results[3][range_loc])
                plt.title( 'Fit Range: {} - {} A, S/N: {:.1f}, Fit PWV: {:.3f} mm'.format( *fit_range, snr, pwv_fit_results[0] ) )

                # Labels, legend, and save
                plt.xlabel( 'Wavelength (${\\rm\AA}$)' )
                plt.ylabel( 'Normalized Flux' )
                plt.legend()

                pdf.savefig( bbox_inches = 'tight', pad_inches = 0.05 )
                plt.close()

    ### Get the full telluric model and output!

    # Generate full telluric model at the best fit precipitable water vapor value for all orders
    # Models generated for multiple species and continuum
    full_telluric_model = tellurics_utils.generate_full_telluric_model(file_in[7].data, 
                                                                       fit_padding=fit_padding_wave, 
                                                                       kernel_half_width=kernel_half_width_wave, 
                                                                       lsf_type='gauss_poly_model',
                                                                       fit_pwv_value=pwv_fit_results[0],
                                                                       grid_path=telluric_model_grid_path, 
                                                                       lsf_fit_results=psf_fit_results,
                                                                       ortho_poly=ortho_poly_fit_results)

    ## Construct telluric extension for the output FITS file

    # Put together individual models into: line and continuum models
    line_model_data = full_telluric_model[:,:,0]
    continuum_model_data = full_telluric_model[:,:,1]

    # Make the HDU
    hdu_data = np.array([line_model_data, continuum_model_data])
    hdu_header = fits.Header({'EXTNAME': 'Telluric', 'PWV': np.round(pwv_fit_results[0], 5)})

    telluric_hdu = fits.hdu.ImageHDU(data=hdu_data, header=hdu_header)

    # Notes/comments
    telluric_hdu.header.comments['PWV'] = 'Best fit PWV value (mm)'

    date_created = datetime.datetime.now().strftime('%Y/%m/%d')
    telluric_hdu.header.add_history(f'Generated {date_created}. Preliminary version with {telluric_model_grid_path} model grid')

    # Create output file - exclude any potential extraneous extensions
    output_hdu_list = copy.copy(file_in)[:10]
    output_hdu_list.append(telluric_hdu)

    output_hdu_list.writeto(output_file_name, overwrite=True)
