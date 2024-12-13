""" Library of functions for fitting and generating telluric models for 
    EPRV spectrograph data.

    Currently this file is just set up for HPF data. This will be 
    generalized over the coming months to work on other instruments, 
    first NEID.

    Created by DMK on 12/13/2024
"""

##### Imports

# Standard library imports
import bisect
import glob
import os
import re

# Third party imports
import numpy as np
from scipy import interpolate, optimize, special, stats

##### Functions

### Utility functions
def continuum_fit_with_spline( x_values, y_values, x_knot_spacing, lower_sigma_reject, upper_sigma_reject, max_iter = 10 ):
    """ Function to fit a spline to a spectrum with sigma rejection. Different sigma rejection levels below and above the fit are allowed (i.e. get rid of absorption lines)
    It is assumed that no nans are present in the data provided.

    Parameters
    ----------
    x_values : array
        Array of x values to fit (independent variable, e.g. wavelength).
    y_values : array
        Array of y values to fit (dependent variable, e.g. flux).
    x_knot_spacing : float
        The knot spacing for the B spline.
    lower_sigma_reject : float
        The sigma level to reject points below the fit.
    upper_sigma_reject : float
        The sigma level to reject points above the fit.
    max_iter : int, optional
        The maximum number of sigma rejection iterations to perform. The default is 10.

    Returns
    -------
    spline_fit : tuple
        The tuple defining the best fit spline, as output by scipy.interpolate. Elements are (spline knot array, spline coefficient array, spline degree).
    """
    
    # Set the knots array. Keep in mind they must be interior knots, so start at the x value minimum + break space
    spline_knots = np.arange( x_values.min() + x_knot_spacing, x_values.max(), x_knot_spacing )
    
    # Set the values to fit array, will be modified in the rejection loop below
    x_values_to_fit = x_values.copy()
    y_values_to_fit = y_values.copy()

    # Loop for the maximum number of iterations, unless an iteration sooner results in no rejections
    for i_iter in range( max_iter ):
        
        # Redefine the spline knots -- issue can arise if x array changes in a certain way relative to original knot definitions
        spline_knots = np.arange( x_values_to_fit.min() + x_knot_spacing, x_values_to_fit.max(), x_knot_spacing )
        
        # Get the b spline representation of the data
        spline_fit = interpolate.splrep( x_values_to_fit, y_values_to_fit, k = 3, t = spline_knots )

        # Calculate the residuals bewteen the data values and the spline fit
        residuals = y_values_to_fit - interpolate.splev( x_values_to_fit, spline_fit )

        # Get the standard deviation of the residuals, using the MAD
        residuals_mad = stats.median_abs_deviation( residuals, scale = 'normal' )

        # Keep points within the lower/upper sigma level provided
        within_mad = np.where( ( residuals < np.nanmedian( residuals ) + upper_sigma_reject * residuals_mad ) & ( residuals > np.nanmedian( residuals ) - lower_sigma_reject * residuals_mad ) )[0]

        # If no points are rejected, break out of the loop!
        if within_mad.size == y_values_to_fit.size:
            break

        # Re-define the x and y values to fit -- get rid of MAD rejected points
        x_values_to_fit = x_values_to_fit[within_mad]
        y_values_to_fit = y_values_to_fit[within_mad]
        
    return spline_fit

def get_bin_edges_from_centers( bin_centers ):
    """ Converts an array of bin center locations into an array of its edges. Taken from NEID DRP

    Parameters
    ----------
    bin_centers : array_like
        Center locations of bins to get edge locations of

    Returns
    -------
    all_bin_edges : array_like
        Bin edges corresponding to bin_centers
    """
    
    # Calculate all non-bounding edges
    bin_edges = bin_centers[:-1] + np.diff(bin_centers) / 2

    # Calculate bounding edges
    bin_edge_first = bin_centers[0] - (bin_edges[0] - bin_centers[0])
    bin_edge_last  = bin_centers[-1] + (bin_centers[-1] - bin_edges[-1])
    
    # Combine
    all_bin_edges = np.concatenate( [ [ bin_edge_first ], bin_edges, [ bin_edge_last ] ] )

    return all_bin_edges

def rebin_spectrum( new_x, old_x, old_y, normalize = True ):
    """ Rebins a given spectrum to a new set of wavelengths. Taken from NEID DRP

    Parameters
    ----------
    new_x : array_like
        Wavelengths (as bin centers) to rebin to
    old_x : array_like
        Wavelengths (as bin centers) to rebin from
    old_y : array_like
        Bin counts to rebin from
    normalize : bool, optional
        Flag to control rescaling bin counts to match that of the original array. The default is True

    Returns
    -------
    new_y : array_like
        Rebinned fluxes corresponding to new_x
    """
    
    # Convert bin centers into edges
    old_edges_x = get_bin_edges_from_centers( old_x )
    new_edges_x = get_bin_edges_from_centers( new_x )

    # Apply normalization if requested
    normalization_factor = 1
    if normalize:
        old_widths = np.diff( old_edges_x )
        normalization_factor = old_widths

    # Determine cumulative sum at edges
    old_edges_y_sum = np.concatenate( [ [ 0 ], np.nancumsum( old_y * normalization_factor ) ] )
    new_edges_y_sum = np.interp( new_edges_x, old_edges_x, old_edges_y_sum )

    # Integrate each bin
    new_y = np.diff(new_edges_y_sum)

    # Apply normalization if requested
    if normalize:
        new_y /= np.diff(new_edges_x)

    return new_y

def scale_interval_m1top1( x, a, b, inverse_scale = False ):
    """ Scales input x in interval a to b to the range -1 to 1.
    Taken from NEID DRP

    Parameters
    ----------
    x : array_like
        Input array of values to scale.
    a : float
        The value to scale to -1.
    b : float
        The value to scale to 1.
    inverse_scale : bool, optional
        To inverse the process: scale from (-1,1) to (a,b). The default is False.

    Returns
    -------
    scaled_x : array_like
        The scaled version of the input array.
    """
    
    # If inverse: scale from (-1,1) back to (a,b)
    if inverse_scale:
        scaled_x = ( ( b - a ) * x + a + b ) / 2.0
    else:
        scaled_x = ( 2.0 * x - ( b + a ) ) / ( b - a )
    
    return scaled_x

def find_fitting_order_index(wave_data, fit_range):
    """Finds the index containing the order nearest the fitting range in wavelength for the given NeidData wavelength data

    Parameters
    ----------
    wave_data : array_like
        NeidData wavelength data

    fit_range : tuple
        A tuple describing the minimum and maximum wavelengths of the fitting range

    Returns
    -------
    fit_order_index : array_like
        The index of wave_data corresponding to the best order to fit
    """

    # Get the mean wavelength of each order
    order_means = np.nanmean(wave_data, axis = 1)

    # Find the order whose mean is closest to the mean of the given fitting range
    fit_order_index = np.nanargmin(np.abs(order_means - np.mean(fit_range)))

    return fit_order_index


### Functions for the HPF LSF model

def trapezium_gauss_kernel( x, hat_width, sigma, mean ):
    """
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    hat_width : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    mean : TYPE
        DESCRIPTION.

    Returns
    -------
    conv_values : TYPE
        DESCRIPTION.

    """
    
    # Set empty array to hold the profile values
    conv_values = np.zeros( x.shape )
    
    # Center the input pixel array
    x = x - mean[:,np.newaxis]
    
    ### Compressed slope terms -- i = 1 corresponds to the left-of-center sloped part and i = -1 corresponds to the right-of-center sloped part
    for i in [ 1, -1 ]:
        
        # Erf terms
        erf_arg_1 = ( i * hat_width[:,np.newaxis] / 2 + x + i * 1 ) / ( np.sqrt( 2 ) * sigma[:,np.newaxis] )
        erf_arg_2 = ( i * hat_width[:,np.newaxis] / 2 + x ) / ( np.sqrt( 2 ) * sigma[:,np.newaxis] )
    
        erf_term = np.sqrt( 2 * np.pi ) * ( hat_width[:,np.newaxis] + i * 2 * x + 2 ) * ( i * special.erf( erf_arg_1 ) - i * special.erf( erf_arg_2 ) )
        
        # Exponential terms
        exp_arg_1 = -( hat_width[:,np.newaxis] + i * 2 * x + 2 ) ** 2 / ( 8 * sigma[:,np.newaxis] ** 2 )
        exp_arg_2 = -( hat_width[:,np.newaxis] + i * 2 * x ) ** 2 / ( 8 * sigma[:,np.newaxis] ** 2 )

        exp_term = 4 * sigma[:,np.newaxis] * ( np.exp( exp_arg_1 ) - np.exp( exp_arg_2 ) )

        # Add to output values
        conv_values += 0.25 * sigma[:,np.newaxis] * ( erf_term + exp_term )
    
    ### Top Hat portion
    
    erf_arg_1 = ( hat_width[:,np.newaxis] / 2 + x ) / ( np.sqrt( 2 ) * sigma[:,np.newaxis] )
    erf_arg_2 = ( -hat_width[:,np.newaxis] / 2 + x ) / ( np.sqrt( 2 ) * sigma[:,np.newaxis] )

    term_3 = np.sqrt( np.pi / 2 ) * sigma[:,np.newaxis] * ( special.erf( erf_arg_1 ) - special.erf( erf_arg_2 ) )
    
    ### Combine and normalize
    
    conv_values += term_3
    
    conv_values /= np.nanmax( conv_values, axis = 1 )[:,np.newaxis]
    
    return conv_values

def hpf_model_lsf_kernel( x, wavelength_arr, hat_width_parameter, sigma_poly_coeffs, mean_arr, psf_fit_results_coeffs, ortho_polynomial, wavelength_spacing ):
    """
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    wavelength_arr : TYPE
        DESCRIPTION.
    hat_width_parameter : TYPE
        DESCRIPTION.
    sigma_poly_coeffs : TYPE
        DESCRIPTION.
    mean_arr : TYPE
        DESCRIPTION.
    psf_fit_results_coeffs : TYPE
        DESCRIPTION.
    ortho_polynomial : TYPE
        DESCRIPTION.
    wavelength_spacing : TYPE
        DESCRIPTION.

    Returns
    -------
    lsf_norm : TYPE
        DESCRIPTION.

    """

    ### Prep
    
    # Gradient of the wavelength
    wavelength_gradient = np.gradient( wavelength_arr )
    
    # wave_grad_at_pixels = np.interp( mean_arr, np.arange( 2048 ), wavelength_gradient )
    
    # Do a fit
    wavelength_gradient_fit = np.polyfit( scale_interval_m1top1( np.arange( 2048 ), 0, 2048 ), wavelength_gradient, 5 )
    wave_grad_at_pixels = np.polyval( wavelength_gradient_fit, scale_interval_m1top1( mean_arr, 0, 2048 ) )
    
    hat_width_arr = hat_width_parameter / wave_grad_at_pixels
    
    sigma_arr = np.polyval( sigma_poly_coeffs, mean_arr ) ** 2.0
    
    c_op = np.array( [ np.polyval( coeff_poly, mean_arr ) for coeff_poly in psf_fit_results_coeffs ] )
    
    # Center the input pixel arrays
    x_centered = x - mean_arr[:,np.newaxis]
    
    ### Setting the op_hat_width value
    op_hat_width = 2
    
    # Get the trapizium-convolved gaussian values
    trap_conv_gauss_values = trapezium_gauss_kernel( x, hat_width_arr, sigma_arr, mean_arr )
    
    # Evaluate the orthogonal polynomial
    evaled_poly = np.nansum( ( c_op[:,:,np.newaxis] * np.array( [ p( x_centered / ( hat_width_arr / op_hat_width )[:,np.newaxis] ) for p in ortho_polynomial ] ) ), axis = 0 )
    
    # Get the LSF from combining the trapezium-gaussian and the orthogonal polynomia
    lsf = evaled_poly * trap_conv_gauss_values
    
    # Normalize by the integral
    lsf_norm = lsf / np.trapz( lsf, dx = wavelength_spacing )[:,np.newaxis]
    
    return lsf_norm

def tophat_gauss_kernel( x, mean, fwhm, box_width ):
    """A function to return a top hat convolved Gaussian kernel provided the two width parameters. Note no normalization is provided
    
    Parameters
    ----------
    x : array_like (2-dim, shaped NxM)
        A 2D array of pixel locations to compute the kernel at. N is the length of the data array, M is the length of the kernel array
    
    mean : array_like (length N)
        The mean values of the kernel at each data location (aka the pixel location for the input data array)
        
    fwhm : array_like (length N)
        The gaussian FWHM describing the kernel at each input data point
        
    box_width : array_like (length N)
        The box width describing the kernel at each input data point
    
    Returns
    -------
    kernel_values : array_like (2-dim, shaped NxM)
        The values of the LSF convolution kernel for individual input data points. N is the data array length, M is the kernel size (odd).
    """
    
    # First convert the FWHM to a sigma
    sigma = fwhm / ( 2 * np.sqrt( 2 * np.log( 2 ) ) )
    
    # Generate the arguments for the two erf components. Note the newaxis usage for NDarray computation
    erf_arg_1 = ( ( 2 * mean + box_width )[:,np.newaxis] - 2 * x ) / ( 2 * np.sqrt( 2 ) * sigma[:,np.newaxis] )
    erf_arg_2 = ( (-2 * mean + box_width )[:,np.newaxis] + 2 * x ) / ( 2 * np.sqrt( 2 ) * sigma[:,np.newaxis] )
    
    # Generate the kernel values
    kernel_values = special.erf( erf_arg_1 ) + special.erf( erf_arg_2 )
    
    return kernel_values

def gauss_kernel( x, mean, sigma ):
    
    kernel_values = np.exp( - ( x - mean[:,np.newaxis] ) ** 2.0 / ( 2 * sigma[:,np.newaxis] ** 2.0 ) )
        
    return kernel_values

def get_variable_lsf_kernel_values( input_pixel_arr, kernel_pixel_arr, kernel_profile_information, kernel_parameters, wavelength_spacing ):
    """
    

    Parameters
    ----------
    input_pixel_arr : TYPE
        DESCRIPTION.
    kernel_pixel_arr : TYPE
        DESCRIPTION.
    kernel_profile_information : TYPE
        DESCRIPTION.
    kernel_parameters : TYPE
        DESCRIPTION.
    wavelength_spacing : TYPE
        DESCRIPTION.

    Returns
    -------
    lsf_kernel_values : TYPE
        DESCRIPTION.

    """
    
    # Scale the input pixel array to be between -1 and 1, the domian over which the kernel parameters are fit
    input_pixel_arr_m1top1 = scale_interval_m1top1( input_pixel_arr, *kernel_profile_information['DOMAIN'] )
    
    ### Pick out the right line profile parameterization from the input master file. Currently just top hat gaussian, but can add different types in the future
    if kernel_profile_information['PROFILE'] == 'TOP_HAT_GAUSSIAN': # Top Hat convolved Gaussian
        
        # Take the polynomial coefficients describing the kernel parameters vs. pixel for this order
        # Coefficient 1 is the Gaussian FWHM and Coefficient 2 is the top hat box width
        kernel_fwhm_poly     = kernel_parameters['COEFF_1']
        kernel_boxwidth_poly = kernel_parameters['COEFF_2']
        
        # Now apply the fits to the input pixel array (the pixel location of each model grid wavelength)
        kernel_fwhm_arr     = np.polyval( kernel_fwhm_poly, input_pixel_arr_m1top1 )
        kernel_boxwidth_arr = np.polyval( kernel_boxwidth_poly, input_pixel_arr_m1top1 )
        
        # Generate the actual kernel values
        lsf_kernel_values = tophat_gauss_kernel( kernel_pixel_arr, input_pixel_arr, kernel_fwhm_arr, kernel_boxwidth_arr )
        
    elif kernel_profile_information['PROFILE'] == 'GAUSSIAN':
        
        # Take the polynomial coefficients describing the kernel parameters vs. pixel for this order
        # Coefficient 1 is the Gaussian sigma
        kernel_sigma_poly = kernel_parameters['COEFF_1']
        
        # Now apply the fits to the input pixel array (the pixel location of each model grid wavelength)
        kernel_sigma_arr = np.polyval( kernel_sigma_poly, input_pixel_arr_m1top1 )
        
        # Generate the actual kernel values
        lsf_kernel_values = gauss_kernel( kernel_pixel_arr, input_pixel_arr, kernel_sigma_arr )
        
    # Normalize the kernel values so that their integral is 1 (for flux conservation)
    lsf_kernel_values /= np.trapz( lsf_kernel_values, dx = wavelength_spacing )[:,np.newaxis]
        
    return lsf_kernel_values

##### Functions related to the telluric model grid

def parse_model_grid_points_hpf( grid_path ):
    """ Function to search a directory of telluric model grid and return the zenith angle/PWV grid point values, along with the grid wavelength array.
    Adapted from the NEID DRP

    Parameters
    ----------
    grid_path : str
        The path to the telluric model grid directory.

    Returns
    -------
    zeniths : array
        An array of the zenith angle values included in the telluric model grid.
    waters : array
        An array of the precipitable water vapor columm values (mm) included in the telluric model grid.
    wavelengths : array
        The telluric model grid wavelength array.
    """
    
    # Generate a list of all model files. HPF only has a single zenith angle value, but this convention is retained from NEID
    files = glob.glob( os.path.join( grid_path, '*Z[0-9][0-9]PWV[0-9][0-9]*.npy' ) )

    # Initialize lists for each zenith angle/PWV grid point
    zeniths = []
    waters  = []

    # Parse each file name for zenith angle and water vapor
    for file in files:
        
        file_match = re.search( 'Z(\d{2})PWV(\d{2})', os.path.basename( file ) )
        
        if file_match:
            zenith = int( file_match[1] )
            water  = int( file_match[2] )
            
            zeniths.append( zenith )
            waters.append( water )

    # Sort zenith/PWV lists and only return unique points
    zeniths = np.sort( np.unique( zeniths ) )
    waters = np.sort( np.unique( waters ) )

    # Get potential telluric model grid wavelengths files -- and read in the first file name returned in the search
    wavelengths_search = glob.glob( os.path.join( grid_path, '*Wavelengths*.npy' ) )
    
    wavelengths = np.load( wavelengths_search[0] )

    return zeniths, waters, wavelengths

### Convolution functions

def lsf_convolve_per_pixel( input_wave_arr, input_flux_arr, convolution_kernel_values ):
    """Convolves the input spectrum with the provided LSF kernel value arrays, where the kernel is defined at each input pixel.
    Taken from NEID DRP.

    Parameters
    ----------
    input_wave_arr : array_like (1-dim, length N)
        The "x" values (wavelength) for the spectrum to be convolved. Used in normalizing the convolution by the x separation. This array needs to be equally spaced!

    input_flux_arr : array_like (1-dim, length N)
        The "y" values (flux) for the spectrum to be convolved.

    convolution_kernel_values : array_like (2-dim, shape N,M)
        The values of the convolution kernel (LSF) at each individual N pixels. The kernel array per pixel has M points, is odd length, and symmetric.

    Returns
    -------
    convolved_flux_arr : array_like (1-dim, length N)
        The convolved "y" values (flux) from the input spectrum and convolution kernel values
    """
    
    # Extract the kernel size
    kernel_size = convolution_kernel_values.shape[1]
    
    # Do the actual convolution multiplication: flux * kernel for each input spectrum pixel
    convolution_multiplication = input_flux_arr[:,np.newaxis] * convolution_kernel_values
    
    # Multiply by the input x array (wavelength) separation (getting the right convolution amplitude). Again, input wavelength array must be equally spaced
    convolution_multiplication *= np.diff( input_wave_arr )[0]
    
    ### *** Version that adds to the N length convolution output array, so no large 2D array being filled but still a loop (using scipy.sparse may be explored in the future)
    
    # Set up the 1D array that will hold the convolution output
    convolved_flux_arr = np.zeros( input_wave_arr.size )
    
    # Loop through and add in the convolution multiplication result where it needs to
    for i_point in range( input_wave_arr.size ):
        
        # Get the indices where the convolution mulitiplication result will be added into
        output_index_low  = max( 0, i_point - kernel_size // 2 )
        output_index_high = min( input_wave_arr.size, i_point + kernel_size // 2 + 1 )
        
        # Get the indices of the convolution multiplication result to add in
        adding_index_low  = max( 0, kernel_size // 2 - i_point )
        adding_index_high = min( kernel_size, input_wave_arr.size - i_point + kernel_size // 2 )

        # Add to convolution result
        convolved_flux_arr[output_index_low:output_index_high] += convolution_multiplication[i_point,adding_index_low:adding_index_high]
        
    return convolved_flux_arr


def fit_pwv_hpf( data_wavelengths, data_fluxes, blaze, fit_padding, kernel_half_width, lsf_type, fit_ranges, grid_path, lsf_fit_results = None, ortho_poly = None, fit_method = 'ratio' ):
     
    # Pull telluric model grid info -- the zenith angle/water column grid points and the full model wavelength array
    grid_zeniths, grid_waters, grid_wavelengths = parse_model_grid_points_hpf( grid_path )

    # Initializing the grid values for interpolation: first array is the set of PWV grid points and the second will hold the fit range data wavelength values
    fit_grid_points = [ grid_waters, [] ]

    # This will hold the zenith angle interpolated telluric model grid at each grid PWV value across all fit ranges, to be interpolated over
    fit_grid_values_all = [ [] for i in range( len( grid_waters ) ) ]

    # Empty arrays to hold the data wavelength and fluxes that will be fit, concatenating together all the fit ranges
    fluxes_to_fit      = np.array( [] )
    wavelengths_to_fit = np.array( [] )
        
    fit_ranges_used = []
    
    for i_fit_range, fit_range in enumerate( fit_ranges ):
        
        ### Set up -- fit order, data in the fit order, and the wavelength edges of the fit region

        # Get the spectral order index for this wavelength fit range and pull out the data wavelengths and fluxes for this order
        fit_order   = find_fitting_order_index( data_wavelengths, fit_range )
        wavelengths = data_wavelengths[fit_order]
        fluxes      = data_fluxes[fit_order]

        # order_lsf_fit_results = psf_fit_results[fit_order]

        # Normalize the data
        fluxes  = fluxes / blaze[fit_order]
        not_nan = np.where( np.isfinite( fluxes ) )[0]
        
        continuum_spl = continuum_fit_with_spline( wavelengths[not_nan], fluxes[not_nan], 12.5, 2, 10 )
        continuum = interpolate.splev( wavelengths, continuum_spl )

        fluxes /= continuum

        # The minimum and maximum wavelength for the fit range (this is inherited from the previous version of the code, but just using the input fit range might be fine. Will leave for now)
        wavelength_min = np.max( [ np.nanmin( wavelengths ), np.nanmin( grid_wavelengths ), fit_range[0] ] )
        wavelength_max = np.min( [ np.nanmax( wavelengths ), np.nanmax( grid_wavelengths ), fit_range[1] ] )

        ### Get indices of the data/model wavelength arrays that fall within the fit range, and set wavelength spacing for convolution

        # The indices of the model wavelength grid within the fit range
        telluric_fit_range_grid_loc = np.where( ( grid_wavelengths >= wavelength_min ) & ( grid_wavelengths <= wavelength_max ) )[0]

        # The wavelength spacing for convolution is based on the model in the order's wavelength range: the maximum of the spacing of the model here
        wavelength_spacing = np.nanmax( np.diff( grid_wavelengths[telluric_fit_range_grid_loc] ) )

        # The indices of the data wavelength array that fall within the fit range AND where the flux values are not nan (catch needed for fit routine)
        telluric_fit_range_data_loc = np.where( ( wavelengths >= wavelength_min ) & ( wavelengths <= wavelength_max ) & ( ~np.isnan( fluxes ) ) )[0]

        # A catch to skip if there are NO data points in the fit range with not-nan fluxes (e.g. if the blaze is bad and 0-valued)
        if telluric_fit_range_data_loc.size == 0:
            continue
        else: # If the fit range will be included -- add to the list out fit ranges used!
            fit_ranges_used.append( fit_range )

        # Append the data wavelength in this fit range to the for-fit grid point structure
        fit_grid_points[1].extend( wavelengths[telluric_fit_range_data_loc] )

        ### Per order rough wavelength solution mapping

        # A polynomial fit for wavelength -> pixel. This requires re-scaling the independent variable (wavelength) to avoid fitting issues
        # A quick bespoke function to perform the scaling for order wavelength running from -1 to 1
        order_wavelength_x_transform_fn = lambda x: scale_interval_m1top1( x, np.nanmin( wavelengths ), np.nanmax( wavelengths ) )

        # The polynomial fit going from the -1 to 1 scaled wavelength to pixel
        order_pixel_from_wavelength = np.polyfit( order_wavelength_x_transform_fn( wavelengths ), np.arange( len( wavelengths ) ), 9 )

        ### Next prepare for creating the model: generating the model wavelength grid for convolution, getting the kernel parameters, ...

        # The extent over which to generate the model (the order wavelength range + some padding)
        model_wavelength_extents = [ wavelength_min - fit_padding, wavelength_max + fit_padding ]

        # Generate the equally spaced wavelength grid for creating the LSF convolved model
        grid_model_wavelength_equal_space = np.arange( model_wavelength_extents[0], model_wavelength_extents[1] + wavelength_spacing, wavelength_spacing )

        # Also apply the mapping of wavelength -> pixel for this equally spaced grid
        grid_model_wavelength_equal_space_in_pixel = np.polyval( order_pixel_from_wavelength, order_wavelength_x_transform_fn( grid_model_wavelength_equal_space ) )

        ### Now to generate the convolution kernel values

        # Generate a common kernel (relative) wavelength array centered on zero, given the wavelength spacing and kernel extent
        # First, one half of the array is generated. Then 0 is sandwiched between that half array, and the negated/flipped version of that array to ensure symmetry
        common_kernel_relative_wavelength_arr = np.arange( wavelength_spacing, kernel_half_width + wavelength_spacing, wavelength_spacing )
        common_kernel_relative_wavelength_arr = np.hstack( [ -common_kernel_relative_wavelength_arr[::-1], [0], common_kernel_relative_wavelength_arr ] )

        # Next, add the common relative kernel wavelength array to each individual input wavelength point
        kernel_wavelength_arr = common_kernel_relative_wavelength_arr + grid_model_wavelength_equal_space[:,np.newaxis]

        # Convert the kernel wavelength array into pixel space, which is what the kernel is defined over
        kernel_pixel_arr = np.polyval( order_pixel_from_wavelength, order_wavelength_x_transform_fn( kernel_wavelength_arr ) )

        if lsf_type == 'gauss_poly_model':
            order_lsf_fit_results = lsf_fit_results[fit_order]
            # Now get the kernel values at each individual kernel pixel location
            kernel_values = hpf_model_lsf_kernel( kernel_pixel_arr, wavelengths, order_lsf_fit_results['hat_width_p0'], order_lsf_fit_results['gauss_sigmasqrt'], 
                                            grid_model_wavelength_equal_space_in_pixel, order_lsf_fit_results['op_coeff'], ortho_poly, wavelength_spacing )
        elif lsf_type == 'resolution_gauss':
            # Turn resolution into delta-wavelength
            lsf_pixel_resolution = ( wavelengths / 55000 / ( 2 * np.sqrt( 2 * np.log( 2 ) ) ) ) / np.gradient( wavelengths )
            lsf_resolution_fit = np.polyfit( scale_interval_m1top1( np.arange( 2048 ), 0, 2048 ), lsf_pixel_resolution, 5 )
            lsf_resolution_dict = { 'GLOBAL': { 'PROFILE': 'GAUSSIAN', 'DOMAIN': [ 0, 2048 ] }, fit_order: { 'COEFF_1': lsf_resolution_fit } }

            kernel_values = get_variable_lsf_kernel_values( grid_model_wavelength_equal_space_in_pixel, kernel_pixel_arr, lsf_resolution_dict['GLOBAL'], lsf_resolution_dict[fit_order], wavelength_spacing )

        ##### Water vapor fitting

        # Now loop through each of the grid point PWV values and interpolate the model onto the given zenith angle
        for i_water, water in enumerate( grid_waters ):

            # Interpolate the line absorption model at the grid water vapor value to the given zenith angle (zi = zenith interpolated)
            absorptions_zi = np.load( os.path.join( grid_path, 'hpf_TelluricModel_H2OLines_Z35PWV{:02d}_20240212_v001.npy'.format( water ) ) )

            # Rebin the model to the padded equally spaced wavelength grid
            absorptions_zi_rebin = rebin_spectrum( grid_model_wavelength_equal_space, grid_wavelengths, absorptions_zi )

            # Now perform the convolution! (zi_rebin is assumed here for sake of variable naming)
            absorptions_lsfconv = lsf_convolve_per_pixel( grid_model_wavelength_equal_space, absorptions_zi_rebin, kernel_values )

            # Rebin the LSF convolved telluric model to the data wavelength grid (gets rid of the convolution padding and such)
            absorptions_lsfconv_databin = rebin_spectrum( wavelengths[telluric_fit_range_data_loc], grid_model_wavelength_equal_space, absorptions_lsfconv )

            # Output the rebinned, convolved telluric model
            fit_grid_values_all[i_water].extend( absorptions_lsfconv_databin )

        wavelengths_to_fit = np.append( wavelengths_to_fit, wavelengths[telluric_fit_range_data_loc] )
        fluxes_to_fit      = np.append( fluxes_to_fit, fluxes[telluric_fit_range_data_loc] )

    if fit_method == 'ratio':
        
        def interpolate_telluric_model( x, water ):

            # Create interpolation array for interpn
            interpolation_points = np.array( [ np.full( len( x ), water ), x ] ).transpose()

            # Return interpolated array
            return interpolate.interpn( fit_grid_points, fit_grid_values_all, interpolation_points )

        def minimize_water_ratio( water, x, data ):

            interpolated_model = interpolate_telluric_model( x, water )

            return np.sqrt( np.sum( ( np.nanmedian( data / interpolated_model ) - data / interpolated_model ) ** 2.0 ) )

        res = optimize.minimize( minimize_water_ratio, [ np.median( grid_waters ) ], ( wavelengths_to_fit, fluxes_to_fit ), bounds = [ ( np.min( grid_waters ), np.max( grid_waters ) ) ] )

        water_fit = res.x[0]

        # Evaluate the model!
        best_fit_model = interpolate_telluric_model( wavelengths_to_fit, water_fit )
        
    elif fit_method == 'curve_fit':
        
        def interpolate_telluric_model( x, water, scale ):

            # Create interpolation array for interpn
            interpolation_points = np.array( [ np.full( len( x ), water ), x ] ).transpose()

            # Return interpolated array
            return interpolate.interpn( fit_grid_points, fit_grid_values_all, interpolation_points ) * scale

        # Perform the fit
        popt, popconv = optimize.curve_fit( interpolate_telluric_model, wavelengths_to_fit, fluxes_to_fit, bounds = ( [ np.min(grid_waters), 0 ], [ np.max(grid_waters), np.inf ] ) )
        water_fit = popt[0]
        scale_fit = popt[1]
        
        best_fit_model = interpolate_telluric_model( wavelengths_to_fit, water_fit, scale_fit ) / scale_fit

    return water_fit, wavelengths_to_fit, fluxes_to_fit, best_fit_model

def generate_full_telluric_model( data_wavelengths, fit_padding, kernel_half_width, lsf_type, fit_pwv_value, grid_path, lsf_fit_results = None, ortho_poly = None ):
     
    # Pull telluric model grid info -- the zenith angle/water column grid points and the full model wavelength array
    grid_zeniths, grid_waters, grid_wavelengths = parse_model_grid_points_hpf( grid_path )

    # Interpolate continuum and H2O model to the fit PWV value
    grid_absorptions_h2o = interpolate_model_grid_hpf( grid_path, 35.0, fit_pwv_value, 'lines' )
    grid_absorptions_cont = interpolate_model_grid_hpf( grid_path, 35.0, fit_pwv_value, 'continuum' )
    
    # Read in the other species
    grid_absorptions_o2 = np.load( os.path.join( grid_path, 'hpf_TelluricModel_O2Lines_Z35PWV05_20240212_v001.npy' ) )
    grid_absorptions_co2 = np.load( os.path.join( grid_path, 'hpf_TelluricModel_CO2Lines_Z35PWV05_20240212_v001.npy' ) )
    grid_absorptions_ch4 = np.load( os.path.join( grid_path, 'hpf_TelluricModel_CH4Lines_Z35PWV05_20240212_v001.npy' ) )
    grid_absorptions_co2_more = np.load( os.path.join( grid_path, 'hpf_TelluricModel_CO2Lines_420ppm_Z35PWV05_20240212_v001.npy' ) )

    # Empty array to hold the telluric model with: axis 0 = h2o, 1 = o2, 2 = co2, 3 = ch4, 4 = continuum
    telluric_data = np.empty( shape = ( data_wavelengths.shape[0], data_wavelengths.shape[1], 6 ) )
    
    for order, order_wavelengths in enumerate( data_wavelengths ):
        
        ### Set up -- fit order, data in the fit order, and the wavelength edges of the fit region

        # The minimum and maximum wavelength for the fit range (this is inherited from the previous version of the code, but just using the input fit range might be fine. Will leave for now)
        wavelength_min = np.nanmin( order_wavelengths )
        wavelength_max = np.nanmax( order_wavelengths )

        ### Get indices of the data/model wavelength arrays that fall within the fit range, and set wavelength spacing for convolution

        # The indices of the model wavelength grid within the fit range
        telluric_fit_range_grid_loc = np.where( ( grid_wavelengths >= wavelength_min ) & ( grid_wavelengths <= wavelength_max ) )[0]

        # The wavelength spacing for convolution is based on the model in the order's wavelength range: the maximum of the spacing of the model here
        wavelength_spacing = np.nanmax( np.diff( grid_wavelengths[telluric_fit_range_grid_loc] ) )

        ### Per order rough wavelength solution mapping

        # A polynomial fit for wavelength -> pixel. This requires re-scaling the independent variable (wavelength) to avoid fitting issues
        # A quick bespoke function to perform the scaling for order wavelength running from -1 to 1
        order_wavelength_x_transform_fn = lambda x: scale_interval_m1top1( x, wavelength_min, wavelength_max )

        # The polynomial fit going from the -1 to 1 scaled wavelength to pixel
        order_pixel_from_wavelength = np.polyfit( order_wavelength_x_transform_fn( order_wavelengths ), np.arange( len( order_wavelengths ) ), 9 )

        ### Next prepare for creating the model: generating the model wavelength grid for convolution, getting the kernel parameters, ...

        # The extent over which to generate the model (the order wavelength range + some padding)
        model_wavelength_extents = [ wavelength_min - fit_padding, wavelength_max + fit_padding ]

        # Generate the equally spaced wavelength grid for creating the LSF convolved model
        grid_model_wavelength_equal_space = np.arange( model_wavelength_extents[0], model_wavelength_extents[1] + wavelength_spacing, wavelength_spacing )

        # Also apply the mapping of wavelength -> pixel for this equally spaced grid
        grid_model_wavelength_equal_space_in_pixel = np.polyval( order_pixel_from_wavelength, order_wavelength_x_transform_fn( grid_model_wavelength_equal_space ) )

        ### Now to generate the convolution kernel values

        # Generate a common kernel (relative) wavelength array centered on zero, given the wavelength spacing and kernel extent
        # First, one half of the array is generated. Then 0 is sandwiched between that half array, and the negated/flipped version of that array to ensure symmetry
        common_kernel_relative_wavelength_arr = np.arange( wavelength_spacing, kernel_half_width + wavelength_spacing, wavelength_spacing )
        common_kernel_relative_wavelength_arr = np.hstack( [ -common_kernel_relative_wavelength_arr[::-1], [0], common_kernel_relative_wavelength_arr ] )

        # Next, add the common relative kernel wavelength array to each individual input wavelength point
        kernel_wavelength_arr = common_kernel_relative_wavelength_arr + grid_model_wavelength_equal_space[:,np.newaxis]

        # Convert the kernel wavelength array into pixel space, which is what the kernel is defined over
        kernel_pixel_arr = np.polyval( order_pixel_from_wavelength, order_wavelength_x_transform_fn( kernel_wavelength_arr ) )

        if lsf_type == 'gauss_poly_model':
            
            order_lsf_fit_results = lsf_fit_results[order]

            # Now get the kernel values at each individual kernel pixel location
            kernel_values = hpf_model_lsf_kernel( kernel_pixel_arr, order_wavelengths, order_lsf_fit_results['hat_width_p0'], order_lsf_fit_results['gauss_sigmasqrt'], 
                                            grid_model_wavelength_equal_space_in_pixel, order_lsf_fit_results['op_coeff'], ortho_poly, wavelength_spacing )
        elif lsf_type == 'resolution_gauss':
            # Turn resolution into delta-wavelength
            lsf_pixel_resolution = ( order_wavelengths / 55000 / ( 2 * np.sqrt( 2 * np.log( 2 ) ) ) ) / np.gradient( order_wavelengths )
            lsf_resolution_fit = np.polyfit( scale_interval_m1top1( np.arange( 2048 ), 0, 2048 ), lsf_pixel_resolution, 5 )
            lsf_resolution_dict = { 'GLOBAL': { 'PROFILE': 'GAUSSIAN', 'DOMAIN': [ 0, 2048 ] }, order: { 'COEFF_1': lsf_resolution_fit } }

            kernel_values = get_variable_lsf_kernel_values( grid_model_wavelength_equal_space_in_pixel, kernel_pixel_arr, lsf_resolution_dict['GLOBAL'], lsf_resolution_dict[order], wavelength_spacing )

        ## Convolve and output H2O model in this wavelength range
        absorptions_zi_rebin = rebin_spectrum( grid_model_wavelength_equal_space, grid_wavelengths, grid_absorptions_h2o )
        absorptions_lsfconv = lsf_convolve_per_pixel( grid_model_wavelength_equal_space, absorptions_zi_rebin, kernel_values )
        absorptions_lsfconv_databin = rebin_spectrum( order_wavelengths, grid_model_wavelength_equal_space, absorptions_lsfconv )

        telluric_data[order,:,0] = absorptions_lsfconv_databin
        
        ## Convolve and output O2 model in this wavelength range
        absorptions_zi_rebin = rebin_spectrum( grid_model_wavelength_equal_space, grid_wavelengths, grid_absorptions_o2 )
        absorptions_lsfconv = lsf_convolve_per_pixel( grid_model_wavelength_equal_space, absorptions_zi_rebin, kernel_values )
        absorptions_lsfconv_databin = rebin_spectrum( order_wavelengths, grid_model_wavelength_equal_space, absorptions_lsfconv )

        telluric_data[order,:,1] = absorptions_lsfconv_databin

        ## Convolve and output CO2 model in this wavelength range
        absorptions_zi_rebin = rebin_spectrum( grid_model_wavelength_equal_space, grid_wavelengths, grid_absorptions_co2 )
        absorptions_lsfconv = lsf_convolve_per_pixel( grid_model_wavelength_equal_space, absorptions_zi_rebin, kernel_values )
        absorptions_lsfconv_databin = rebin_spectrum( order_wavelengths, grid_model_wavelength_equal_space, absorptions_lsfconv )

        telluric_data[order,:,2] = absorptions_lsfconv_databin

        ## Convolve and output CH4 model in this wavelength range
        absorptions_zi_rebin = rebin_spectrum( grid_model_wavelength_equal_space, grid_wavelengths, grid_absorptions_ch4 )
        absorptions_lsfconv = lsf_convolve_per_pixel( grid_model_wavelength_equal_space, absorptions_zi_rebin, kernel_values )
        absorptions_lsfconv_databin = rebin_spectrum( order_wavelengths, grid_model_wavelength_equal_space, absorptions_lsfconv )

        telluric_data[order,:,3] = absorptions_lsfconv_databin

        ## Convolve and output continuum model in this wavelength range
        absorptions_zi_rebin = rebin_spectrum( grid_model_wavelength_equal_space, grid_wavelengths, grid_absorptions_cont )
        absorptions_lsfconv = lsf_convolve_per_pixel( grid_model_wavelength_equal_space, absorptions_zi_rebin, kernel_values )
        absorptions_lsfconv_databin = rebin_spectrum( order_wavelengths, grid_model_wavelength_equal_space, absorptions_lsfconv )

        telluric_data[order,:,4] = absorptions_lsfconv_databin

        ## Convolve and output CO2 model in this wavelength range
        absorptions_zi_rebin = rebin_spectrum( grid_model_wavelength_equal_space, grid_wavelengths, grid_absorptions_co2_more )
        absorptions_lsfconv = lsf_convolve_per_pixel( grid_model_wavelength_equal_space, absorptions_zi_rebin, kernel_values )
        absorptions_lsfconv_databin = rebin_spectrum( order_wavelengths, grid_model_wavelength_equal_space, absorptions_lsfconv )

        telluric_data[order,:,5] = absorptions_lsfconv_databin

    return telluric_data

def interpolate_model_grid_hpf( grid_path, zenith, water, grid_type ):
    
    # Parse the model grid
    zeniths, waters, wavelengths = parse_model_grid_points_hpf( grid_path )

    # Find model grid point(s) which the interpolation point lies between or at
    near_zeniths = nearest_points(zeniths, zenith)
    near_waters  = nearest_points(waters, water)
    
    # Construct a grid using the nearest data points
    grid_points = []
    if (len(near_zeniths) > 1):
        grid_points.append(near_zeniths)
    if (len(near_waters) > 1):
        grid_points.append(near_waters)
    grid_points.append(wavelengths[:])

    # Acquire grid values from disk
    grid_values_shape = []
    if (len(near_zeniths) > 1):
        grid_values_shape.append(len(near_zeniths))
    if (len(near_waters) > 1):
        grid_values_shape.append(len(near_waters))
    grid_values_shape.append(len(wavelengths))

    grid_values_shape = tuple(grid_values_shape)

    grid_values = np.empty(shape = grid_values_shape)
    for i, near_zenith in enumerate(near_zeniths):
        for j, near_water in enumerate(near_waters):
            # Load telluric transmission spectrum
            zenith_str = 'Z%02d' % near_zenith
            water_str = 'PWV%02d' % near_water
            
            if grid_type == 'lines': # The grid of line absorption models
                spectrum_search = glob.glob( os.path.join( grid_path, 'hpf_TelluricModel_H2O*{:}{:}*.npy'.format( zenith_str, water_str ) ) )
            elif grid_type == 'continuum': # The grid of continuum absorption models
                spectrum_search = glob.glob( os.path.join( grid_path, 'hpf_TelluricModel_Continuum_{:}{:}*.npy'.format( zenith_str, water_str ) ) )
                
            spectrum_search.sort()
            
            spectrum_path = spectrum_search[0]

            values = np.load(spectrum_path)

            # Place values in larger array
            indexed = grid_values[:]

            if (len(near_zeniths) > 1):
                indexed = indexed[i]

            if (len(near_waters) > 1):
                indexed = indexed[j]

            indexed[:] = values

    # Construct interpolation array
    interpolate_points = np.empty(shape = (len(wavelengths), len(grid_points)))

    index = 0
    if (len(near_zeniths) > 1):
        interpolate_points[:, index] = np.full(len(wavelengths), zenith)
        index += 1

    if (len(near_waters) > 1):
        interpolate_points[:, index] = np.full(len(wavelengths), water)
        index += 1

    interpolate_points[:, index] = wavelengths[:]

    # interpolate the grid
    absorptions = interpolate.interpn(grid_points, grid_values, interpolate_points)

    return absorptions

def nearest_points(values, value):
    """Returns a list of points within values bounding the given value

    Parameters
    ----------
    values : list
        List of (ascended sorted) values to search within

    value : float
        value to find bounds of

    Returns
    -------
    bounds : list
        The sorted list of values bounding the given value, or a list containing only the given value
    """
    # Find the index nearest and containing a value <= the given value
    value_i = bisect.bisect_left(values, value)

    # If the value exists in the list, return it
    if (value_i != bisect.bisect_right(values, value)):
        return [values[value_i]]

    # If the value is below the minimum, return the minimum
    if (value_i == 0):
        return [values[0]]

    # If the value is above the maximum, return the maximum
    if (value_i == len(values)):
        return [values[-1]]

    # Otherwise, return the list values bounding the given value
    return [values[value_i - 1], values[value_i]]

def convolve_full_spectrum( data_wavelengths, fit_padding, kernel_half_width, lsf_type, grid_wavelengths, grid_absorptions, lsf_fit_results = None, ortho_poly = None ):
    
    # Empty array to hold the telluric model
    telluric_data = np.empty( shape = data_wavelengths.shape )
    
    for order, order_wavelengths in enumerate( data_wavelengths ):
        
        ### Set up -- fit order, data in the fit order, and the wavelength edges of the fit region

        # The minimum and maximum wavelength for the fit range (this is inherited from the previous version of the code, but just using the input fit range might be fine. Will leave for now)
        wavelength_min = np.nanmin( order_wavelengths )
        wavelength_max = np.nanmax( order_wavelengths )

        ### Get indices of the data/model wavelength arrays that fall within the fit range, and set wavelength spacing for convolution

        # The indices of the model wavelength grid within the fit range
        telluric_fit_range_grid_loc = np.where( ( grid_wavelengths >= wavelength_min ) & ( grid_wavelengths <= wavelength_max ) )[0]

        # The wavelength spacing for convolution is based on the model in the order's wavelength range: the maximum of the spacing of the model here
        wavelength_spacing = np.nanmax( np.diff( grid_wavelengths[telluric_fit_range_grid_loc] ) )

        ### Per order rough wavelength solution mapping

        # A polynomial fit for wavelength -> pixel. This requires re-scaling the independent variable (wavelength) to avoid fitting issues
        # A quick bespoke function to perform the scaling for order wavelength running from -1 to 1
        order_wavelength_x_transform_fn = lambda x: scale_interval_m1top1( x, wavelength_min, wavelength_max )

        # The polynomial fit going from the -1 to 1 scaled wavelength to pixel
        order_pixel_from_wavelength = np.polyfit( order_wavelength_x_transform_fn( order_wavelengths ), np.arange( len( order_wavelengths ) ), 9 )

        ### Next prepare for creating the model: generating the model wavelength grid for convolution, getting the kernel parameters, ...

        # The extent over which to generate the model (the order wavelength range + some padding)
        model_wavelength_extents = [ wavelength_min - fit_padding, wavelength_max + fit_padding ]

        # Generate the equally spaced wavelength grid for creating the LSF convolved model
        grid_model_wavelength_equal_space = np.arange( model_wavelength_extents[0], model_wavelength_extents[1] + wavelength_spacing, wavelength_spacing )

        # Also apply the mapping of wavelength -> pixel for this equally spaced grid
        grid_model_wavelength_equal_space_in_pixel = np.polyval( order_pixel_from_wavelength, order_wavelength_x_transform_fn( grid_model_wavelength_equal_space ) )

        ### Now to generate the convolution kernel values

        # Generate a common kernel (relative) wavelength array centered on zero, given the wavelength spacing and kernel extent
        # First, one half of the array is generated. Then 0 is sandwiched between that half array, and the negated/flipped version of that array to ensure symmetry
        common_kernel_relative_wavelength_arr = np.arange( wavelength_spacing, kernel_half_width + wavelength_spacing, wavelength_spacing )
        common_kernel_relative_wavelength_arr = np.hstack( [ -common_kernel_relative_wavelength_arr[::-1], [0], common_kernel_relative_wavelength_arr ] )

        # Next, add the common relative kernel wavelength array to each individual input wavelength point
        kernel_wavelength_arr = common_kernel_relative_wavelength_arr + grid_model_wavelength_equal_space[:,np.newaxis]

        # Convert the kernel wavelength array into pixel space, which is what the kernel is defined over
        kernel_pixel_arr = np.polyval( order_pixel_from_wavelength, order_wavelength_x_transform_fn( kernel_wavelength_arr ) )

        if lsf_type == 'gauss_poly_model':
            
            order_lsf_fit_results = lsf_fit_results[order]

            # Now get the kernel values at each individual kernel pixel location
            kernel_values = hpf_model_lsf_kernel( kernel_pixel_arr, order_wavelengths, order_lsf_fit_results['hat_width_p0'], order_lsf_fit_results['gauss_sigmasqrt'], 
                                            grid_model_wavelength_equal_space_in_pixel, order_lsf_fit_results['op_coeff'], ortho_poly, wavelength_spacing )
        elif lsf_type == 'resolution_gauss':
            # Turn resolution into delta-wavelength
            lsf_pixel_resolution = ( order_wavelengths / 55000 / ( 2 * np.sqrt( 2 * np.log( 2 ) ) ) ) / np.gradient( order_wavelengths )
            lsf_resolution_fit = np.polyfit( scale_interval_m1top1( np.arange( 2048 ), 0, 2048 ), lsf_pixel_resolution, 5 )
            lsf_resolution_dict = { 'GLOBAL': { 'PROFILE': 'GAUSSIAN', 'DOMAIN': [ 0, 2048 ] }, order: { 'COEFF_1': lsf_resolution_fit } }

            kernel_values = get_variable_lsf_kernel_values( grid_model_wavelength_equal_space_in_pixel, kernel_pixel_arr, lsf_resolution_dict['GLOBAL'], lsf_resolution_dict[order], wavelength_spacing )

        ## Convolve and output H2O model in this wavelength range
        absorptions_zi_rebin = rebin_spectrum( grid_model_wavelength_equal_space, grid_wavelengths, grid_absorptions )
        absorptions_lsfconv = lsf_convolve_per_pixel( grid_model_wavelength_equal_space, absorptions_zi_rebin, kernel_values )
        absorptions_lsfconv_databin = rebin_spectrum( order_wavelengths, grid_model_wavelength_equal_space, absorptions_lsfconv )

        telluric_data[order] = absorptions_lsfconv_databin

    return telluric_data

