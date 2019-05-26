from ct_scan import *
from ct_calibrate import *
from ct_lib import *
from ramp_filter import *
from back_project import *
from hu import *


def scan_and_reconstruct(photon_source, material_data, phantom, scale, angles, mas=10000, alpha=0.001,
                         interpolation_order=1, model_noise=False,
                         scatter_noise_level=0.0, fixed_noise_level=100.0):
    """ Simulation of the CT scanning process
        reconstruction = scan_and_reconstruct(photons, material, phantom, scale, angles, mas, alpha)
        takes the phantom data in phantom (samples x samples), scans it using the
        source photons and material information given, as well as the scale (in cm),
        number of angles, time-current product in mas, and raised-cosine power
        alpha for filtering. The output reconstruction is the same size as phantom."""

    # Convert source (photons per (mas, cm^2)) to photons
    photon_source = photon_source * mas * beam_cross_section(scale)

    # create sinogram from phantom data, with received detector values
    sinogram = ct_scan(photon_source, material_data, phantom, scale, angles, mas=mas,
                       model_noise=model_noise,
                       interpolation_order=interpolation_order, fixed_noise_level=fixed_noise_level,
                       scatter_noise_level=scatter_noise_level)

    # convert detector values into calibrated attenuation values
    total_attenuation = ct_calibrate(photon_source, material_data, sinogram, scale)

    # Ram-Lak
    filtered_sinogram = ramp_filter(total_attenuation, scale, alpha=alpha)

    # Back-projection
    backprojection = back_project(filtered_sinogram)

    # todo: convert to Hounsfield Units

    return backprojection


# def scan_and_reconstruct_iterator(photon_source, material_data, phantom, scale, angles, mas=10000, alpha=0.001):
#     """ Simulation of the CT scanning process
#         reconstruction = scan_and_reconstruct(photons, material, phantom, scale, angles, mas, alpha)
#         takes the phantom data in phantom (samples x samples), scans it using the
#         source photons and material information given, as well as the scale (in cm),
#         number of angles, time-current product in mas, and raised-cosine power
#         alpha for filtering. The output reconstruction is the same size as phantom."""
#
#     # todo: convert source (photons per (mas, cm^2)) to photons
#
#     # create sinogram from phantom data, with received detector values
#     sinogram = ct_scan(photon_source, material_data, phantom, scale, angles, mas=mas, interpolation_order=1,
#                        noise_level=noise_level)
#
#     # convert detector values into calibrated attenuation values
#     total_attenuation = ct_calibrate(photon_source, material_data, sinogram, scale)
#
#     # Ram-Lak
#     filtered_sinogram = ramp_filter(total_attenuation, scale, alpha=alpha)
#
#     # Back-projection
#     for backprojection in back_project_iterator(filtered_sinogram):
#         # todo: convert to Hounsfield Units
#
#         yield backprojection


def beam_cross_section(scale):
    return np.pi * scale**2 / 4
