""" Script to run for generating HPF telluric models.

    This script takes in a path to a list of HPF spectra and:
        - Fits the water vapor content in each observation
        - Generates a full telluric model
        - Outputs new FITS file with telluric model extension added

    Created by DMK on 12/12/2024.
"""

##### Imports

# Standard library imports
import copy
import os
import re

# Third party imports
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import tqdm

