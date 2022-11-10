import numpy as np

'''
PoissonNoise treats (обрабатывает) each pixel value as the expectation value for the number of incident photons in the pixel,
and implements a Poisson process drawing a realization of the observed number of photons in each pixel.
'''
def apply_noise(img, photon_count):
    opt = dict(dtype=np.float32)
    img = np.exp(-img, **opt)
    # Add poisson noise and ☺retain scale by dividing by photon_count
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html
    img = np.random.poisson(img * photon_count)
    img[img == 0] = 1
    img = img / photon_count
    # Redo log transform and scale img to range [0, img_max] +- some noise.
    img = -np.log(img, **opt)
    return img

# transmittance - пропускаемость
# absorption - поглощение, абсорбация
# +
def transmittance(sinogram):
    return np.mean(np.exp(-sinogram)[sinogram > 0])
# +
def absorption(sinogram):
    return 1 - transmittance(sinogram)
