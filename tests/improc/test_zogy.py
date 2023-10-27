import pytest

import numpy as np
import scipy

import matplotlib.pyplot as plt


from improc.simulator import Simulator
from improc.zogy import zogy_subtract, downsample


def get_images_etc(sim, seeing=None):
    """Get a pair of images of the same sky, along with some additional info like background, PSF, variance map.
    The simulator will generate a new sky between the first and second image.
    The remaining factors, like sensor, camera and star field will remain the same.

    Parameters
    ----------
    sim: Simulator
        The simulator to use to generate the images.
    seeing: tuple with two floats
        Overrides the simulator's seeing_mean for
        the first (ref) and second (new) images.
    Returns
    -------
    im1, im2: numpy.ndarray
        The two images of the same sky.
    var1, var2: numpy.ndarray
        The variance maps for the two images.
    psf1, psf2: numpy.ndarray
        The PSFs for the two images.
    bkg1, bkg2: float
        The background levels for the two images.
    truth1, truth2: SimTruth
        The truth values objects for the two images.
    """
    if seeing is not None:
        sim.pars.seeing_mean = seeing[0]
    sim.make_image(new_sky=True)

    im1 = sim.apply_bias_correction(sim.image)
    var1 = sim.noise_var_map
    psf1 = downsample(sim.psf, sim.truth.oversampling)
    bkg1 = sim.truth.background_mean
    truth1 = sim.truth
    truth1.psf = sim.psf

    if seeing is not None:
        sim.pars.seeing_mean = seeing[1]
    sim.make_image(new_sky=True)

    im2 = sim.apply_bias_correction(sim.image)
    var2 = sim.noise_var_map
    psf2 = downsample(sim.psf, sim.truth.oversampling)
    bkg2 = sim.truth.background_mean
    truth2 = sim.truth
    truth2.psf = sim.psf

    return im1, im2, var1, var2, psf1, psf2, bkg1, bkg2, truth1, truth2


def test_easy_subtraction_no_stars():
    imsize = 256

    low_threshold = 4.31  # this is the maximum value we expect to get from a 256x256 image with unit noise
    assert abs(scipy.special.erfc(low_threshold / np.sqrt(2)) * imsize ** 2 - 1) < 0.1
    threshold = 6.0  # this should be high enough to avoid false positives at the 1/1000 level
    assert scipy.special.erfc(threshold / np.sqrt(2)) * imsize ** 2 < 1e-3

    # this simulator creates images with the same b/g and seeing, so the stars will easily be subtracted
    sim = Simulator(
        image_size_x=imsize,  # not too big, but allow some space for stars
        background_std=0,  # keep background constant between images
        background_mean=1,  # keep the background to a simple value
        background_minimum=0.5,
        read_noise=0,  # make it easier to input the background noise (w/o source noise)
        seeing_mean=2.0,  # make the seeing a little more pronounced
        seeing_std=0,  # keep seeing constant between images
        seeing_minimum=0.5,  # assume seeing never gets much better than this
        transmission_std=0,  # keep transmission constant between images
        pixel_qe_std=0,  # keep pixel QE constant between images
        gain_std=0,  # keep pixel gain constant between images
        star_number=0,
    )
    im1, im2, var1, var2, psf1, psf2, bkg1, bkg2, t1, t2 = get_images_etc(sim, seeing=(2.5, 2.5))
    im1 -= bkg1
    im2 -= bkg2

    print(np.max(psf1), np.max(psf2))
    zogy_diff, zogy_psf, zogy_score, zogy_score_corr, alpha, alpha_err = zogy_subtract(
        # im1, im2, psf1, psf2, np.sqrt(bkg1), np.sqrt(bkg2), 1/np.sqrt(np.sum(psf1**2)), 1/np.sqrt(np.sum(psf2**2)),
        im1, im2, psf1, psf2, np.sqrt(bkg1), np.sqrt(bkg2), 10, 10,

    )

    print(
        f'std(diff)= {np.std(zogy_diff):.3f}, '
        f'std(score)= {np.std(zogy_score):.3f}, '
        f'std(score_corr)= {np.std(zogy_score_corr):.3f}'
    )

    print('done')

def test_easy_subtraction_no_sources():
    imsize = 256

    low_threshold = 4.31  # this is the maximum value we expect to get from a 256x256 image with unit noise
    assert abs(scipy.special.erfc(low_threshold / np.sqrt(2)) * imsize ** 2 - 1) < 0.1
    threshold = 6.0  # this should be high enough to avoid false positives at the 1/1000 level
    assert scipy.special.erfc(threshold / np.sqrt(2)) * imsize ** 2 < 1e-3

    # this simulator creates images with the same b/g and seeing, so the stars will easily be subtracted
    sim = Simulator(
        image_size_x=imsize,  # not too big, but allow some space for stars
        background_std=0,  # keep background constant between images
        read_noise=0,  # make it easier to input the background noise (w/o source noise)
        seeing_mean=2.0,  # make the seeing a little more pronounced
        seeing_std=0,  # keep seeing constant between images
        seeing_minimum=1.0,  # assume seeing never gets much better than this
        transmission_std=0,  # keep transmission constant between images
        pixel_qe_std=0,  # keep pixel QE constant between images
        gain_std=0,  # keep pixel gain constant between images
        saturation_limit=1e9,  # make it impossible to saturate
        star_number=0,
    )

    # im1, im2, var1, var2, psf1, psf2, bkg1, bkg2, t1, t2 = get_images_etc(sim)
    # im1 -= bkg1
    # im2 -= bkg2
    #
    # diff = im2 - im1
    # diff /= np.sqrt(var1 + var2)  # adjust the image difference by the noise in both images
    # assert np.max(abs(diff)) < threshold  # we should not have anything get to the high threshold
    # assert abs( np.max(abs(diff)) - low_threshold) < 1.5  # at least one value should be close to the low threshold
    #
    # # check that peaks in the matched filter image also obey the same statistics (i.e., we don't find anything)
    # matched1 = scipy.signal.convolve(diff, psf1, mode='same') / np.sqrt(np.sum(psf1 ** 2))
    # assert np.max(abs(matched1)) < threshold  # we should not have anything get to the high threshold
    # assert abs( np.max(abs(matched1)) - low_threshold ) < 1.5  # at least one value should be close to the low threshold

    # sim.pars.seeing_std = 0.8  # add some variability of the seeing
    naive_failed = False
    zogy_succeed = True

    for i in range(1):  # try different rolls of the seeing pairs
        im1, im2, var1, var2, psf1, psf2, bkg1, bkg2, t1, t2 = get_images_etc(sim, seeing=(1.5, 2.5))
        im1 -= bkg1
        im2 -= bkg2

        diff = im2 - im1
        diff /= np.sqrt(var1 + var2)  # adjust the image difference by the noise in both images
        if np.max(abs(diff)) > threshold:  # some stars must have been badly subtracted
            naive_failed = True  # at least once did this method fail

        matched1 = scipy.signal.convolve(diff, psf1, mode='same') / np.sqrt(np.sum(psf1 ** 2))
        if np.max(abs(matched1)) > threshold:  # filtering with ref image's PSF has found outliers
            naive_failed = True  # at least once did this method fail

        matched2 = scipy.signal.convolve(diff, psf2, mode='same') / np.sqrt(np.sum(psf2 ** 2))
        if np.max(abs(matched2)) > threshold:  # filtering with new image's PSF has found outliers
            naive_failed = True  # at least once did this method fail

        # now try ZOGY
        zogy_diff, zogy_psf, zogy_score, zogy_score_corr, alpha, alpha_err = zogy_subtract(
            im1, im2, psf1, psf2, np.sqrt(bkg1), np.sqrt(bkg2), 20.0, 20.0,
        )

        if np.max(abs(zogy_score_corr)) > threshold:  # some stars have been badly subtracted?
            zogy_succeed = False  # at least once did this method fail

    assert naive_failed
    assert zogy_succeed
