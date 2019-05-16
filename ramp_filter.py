import math
import numpy as np


def ramp_filter(sinogram, scale: float, alpha: float = 0.001):
    """ Ram-Lak filter with raised-cosine for CT reconstruction

    fs = ramp_filter(sinogram, scale) filters the input in sinogram (angles x samples)
    using a Ram-Lak filter.

    fs = ramp_filter(sinogram, scale, alpha) can be used to modify the Ram-Lak filter by a
    cosine raised to the power given by alpha."""

    # get input dimensions
    angles = sinogram.shape[0]
    n = sinogram.shape[1]

    # Set up filter to be at least twice as long as input
    m = np.floor(np.log(n) / np.log(2) + 2)  # Find the smallest 2^m twice as large as input
    n_samples = int(2 ** m)

    # Apply filter to all angles
    print('Ramp filtering...')

    # Take FT of sinogram
    sinogram_ft = np.fft.fft(sinogram, n=n_samples, axis=1)
    raised_cosine = get_ramlak(scale, n_samples, alpha=alpha)

    filtered_sinogram = np.fft.ifft(sinogram_ft * raised_cosine[None, :], axis=1)[:, :n]

    return np.real(filtered_sinogram)


def get_ramlak(scale, n_samples: int, alpha: float):
    """Get the raised cosine filtered RamLak for given scale and with given number of samples."""
    assert n_samples % 2 == 0
    assert isinstance(n_samples, int)

    # The sampling frequency
    omega_0 = 2*np.pi / scale
    # The interval between sampled values in freq. domain
    delta_omega = omega_0 / n_samples

    ramlak = np.zeros([n_samples], dtype=np.float32)
    halfpoint = int(n_samples / 2)

    # Fill the first half of the ramlak array with correct values
    omega = np.linspace(0, omega_0 / 2, halfpoint, endpoint=False)
    ramlak[:halfpoint] = (omega / (2 * np.pi)) * np.cos((omega / omega_0) * (np.pi / 2))**alpha
    # Fix the value at zero
    ramlak[0] = (delta_omega / (8 * np.pi))

    # Fill the second half of the ramlak array with the mirrored values from the first half
    ramlak[halfpoint:] = np.flip(ramlak[:halfpoint])

    return ramlak
