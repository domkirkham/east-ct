import numpy as np
from photons import *
from ct_calibrate import *
from ct_scan import *


def hu(p, material_data, reconstruction, scale):
    """ convert CT reconstruction output to Hounsfield Units
    calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into Hounsfield
    Units, using the material coefficients, photon energy p and scale given."""

    # use water to calibrate

    # Get dimensions and work out detection for just water of the side
    # length (has to be the same as in ct_scan.m)
    n_samples = reconstruction.shape[0]

    # Width of the scanned area is twice the width of the phantom:
    max_width =  n_samples * scale  # Width of the scanned area

    # The calibration attenuation is same at every point -> just use a single scan of water phantom

    water_residual = ct_detect(p, material_data.coeff('Water'), depth=max_width)



    # put this through the same calibration process as the normal CT data
    air_scan = ct_detect(p, material_data.coeff("Air"), max_width)

    path_atten = -np.log(water_residual / air_scan)

    water_atten = path_atten / max_width


    print(f'Water attenuation = {water_atten}')


    # use result to convert to hounsfield units
    hounsfield = ((reconstruction/water_atten) - 1.0) * 1000


    # limit minimum to -1024, which is normal for CT data.
    hounsfield = np.clip(hounsfield, a_min=-1024, a_max=None)

    return hounsfield
