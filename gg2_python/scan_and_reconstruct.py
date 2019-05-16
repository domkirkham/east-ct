from ct_scan import ct_scan
from ct_calibrate import ct_calibrate
from ramp_filter import ramp_filter
from back_project import back_project
from hu import *


def scan_and_reconstruct(photons, material, phantom, scale, angles, mas=10000, alpha=0.001):
    """ Simulation of the CT scanning process
        reconstruction = scan_and_reconstruct(photons, material, phantom, scale, angles, mas, alpha)
        takes the phantom data in phantom (samples x samples), scans it using the
        source photons and material information given, as well as the scale (in cm),
        number of angles, time-current product in mas, and raised-cosine power
        alpha for filtering. The output reconstruction is the same size as phantom."""

    # convert source (photons per (mas, cm^2)) to photons

    # create sinogram from phantom data, with received detector values
    sinogram = ct_scan(photons, material, phantom, scale, angles)

    # convert detector values into calibrated attenuation values
    calib_sinogram = ct_calibrate(photons, material, sinogram, scale)

    # Ram-Lak
    calib_filtered_sinogram = ramp_filter(calib_sinogram, scale)

    # Back-projection
    reconstruction = back_project(calib_filtered_sinogram)

    # convert to Hounsfield Units

    return reconstruction
