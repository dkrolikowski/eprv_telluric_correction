# eprv_telluric_correction
 Package to fit and generate telluric models for data from EPRV spectrographs, namely HPF and NEID

## Introduction

This repository includes code and telluric model grids to generate telluric models for HPF observations. In the future, this will be generalized to include NEID (and potentially other instruments).

In brief, this code convolves the model grid with a variable, non-Gaussian line spread function defined for HPF using laser frequency comb observations. A precipitable water vapor value is fit for each input observed spectrum, and a new FITS file is written with the full telluric model generated using that best fit precipitable water vapor value. The bespoke line spread function is the crucial aspect of this correction to generate models that accurately reflect an observed spectrum.

_For HPF, the species currently include in the model are H<sub>2</sub>O, O<sub>2</sub>, CO<sub>2</sub>, and CH<sub>4</sub> -- the four molecules with signifcant absorption in the HPF bandpass._

## Instructions

This code will be properly packaged in the future, but for now run it as follows:

1. Clone this repository to your machine: this includes the code and ancillary data needed to generate the models (model grid, LSF information).
2. Set up your data directory: all HPF FITS files you want to generate models for (regardless of pre- or post- maintenance) should sit in the same directory.
3. ``cd`` into this code repostory and run with: ``python run_hpf_tellurics.py /path/to/HPF/FITS/files``

By default the output will be placed in a folder called ``telluric_output`` within your data directory. You can change this output directory using the ``--output_path`` argument when running the script. The outputs include:
1. New FITS files (with the same name as the input files) with an extension called ``TELLURIC`` added that includes:
   
   \- Header keyword ``PWV`` with the best fit precipitable water vapor value in mm.
   
   \- Header ``HISTORY`` that includes the date the model was generated and the model grid used.
   
   \- Data of shape (2, 28, 2048) where the first (28, 2048) array is the line absorption model for all species and the second is the continuum absorption model.
2. A sub-directory called ``plots`` with a PDF for each input file showing the best fit telluric model in the wavelength regions used for fitting the precipitable water vapor value. These plots are handy to take a quick-look at the correction quality.

## Caveats

- This code is currently only written to be used for HPF data reduced with the instrument pipeline. Flexibility will be added in the future for:
   - Other instruments (NEID will come first)
   - HPF data reduced with the GOLDILOCKS pipeline
- The telluric correction is high quality, but the models are still imperfect:
   - The line lists, particularly for water, have known issues.
   - There is no airmass dependence incorporated. This needs to be added, although the effect is not large because HPF observations span a relatively narrow range of airmasses (due the HET's fixed elevation).
   - CO<sub>2</sub> concentration is fixed to 420 ppm. It has a small, but noticeable, increase from 2018 to present. 420 ppm is roughly the average over this timespan.
