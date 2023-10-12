import numpy as np


def calc_sky_flat(images, iterations=3, nsigma=5.0):
    """Calculate the sky flat for a set of images.

    Parameters
    ----------
    images : list of 2D numpy.ndarrays or single 3D numpy.ndarray
        The images to calculate the sky flat for.
    iterations : int
        The number of iterations to use for the sigma clipping procedure.
        Default is 3. If the procedure converges it may do fewer iterations.
    nsigma : float
        The number of sigma to use for the sigma clipping procedure.
        Values further from this many standard deviations are removed.
        Default is 5.0.
    Returns
    -------
    sky_flat : numpy.ndarray
        The sky flat image. The value in each pixel represents how much light was
        lost between the sky and the detector (including quantum efficiency, and digitization).
        Divide an image by the flat to correct for pixel-to-pixel sensitivity variations
        and camera vignetting.
    """

    if isinstance(images, np.ndarray) and images.ndim == 3:
        pass
    elif isinstance(images, list) and all(isinstance(im, np.ndarray) for im in images):
        images = np.array(images)
    else:
        raise TypeError("images must be a list of 2D numpy arrays or a 3D numpy array")

    # normalize all images to the same mean sky level
    mean_sky = np.nanmedian(images, axis=(1, 2), keepdims=True)
    im = images.copy() / mean_sky

    # first iteration:
    sky_flat = np.nanmedian(im, axis=0)
    noise = np.nanstd(im, axis=0)

    # how many nan values?
    nans = np.isnan(im).sum()

    for i in range(iterations):
        # remove pixels that are more than nsigma from the median
        clipped = np.abs(im - sky_flat) > nsigma * noise
        im[clipped] = np.nan

        # recalculate the sky flat and noise
        sky_flat = np.nanmean(im, axis=0)  # we only use median on the first iteration!
        noise = np.nanstd(im, axis=0)

        new_nans = np.isnan(im).sum()
        print(new_nans)
        if new_nans == nans:
            break
        else:
            nans = new_nans

    return sky_flat