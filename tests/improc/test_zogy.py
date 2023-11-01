import pytest

import numpy as np
import scipy

import matplotlib.pyplot as plt


from improc.simulator import Simulator
from improc.zogy import zogy_subtract

imsize = 256

low_threshold = 4.31  # this is the maximum value we expect to get from a 256x256 image with unit noise
assert abs(scipy.special.erfc(low_threshold / np.sqrt(2)) * imsize ** 2 - 1) < 0.1
threshold = 6.0  # this should be high enough to avoid false positives at the 1/1000 level
assert scipy.special.erfc(threshold / np.sqrt(2)) * imsize ** 2 < 1e-3


def test_subtraction_no_stars():
    # this simulator creates images with the same b/g and seeing, so the stars will easily be subtracted
    sim = Simulator(
        image_size_x=imsize,  # not too big, but allow some space for stars
        background_std=0,  # keep background constant between images
        background_mean=1,  # keep the background to a simple value
        background_minimum=0.5,
        read_noise=0,  # make it easier to input the background noise (w/o source noise)
        dark_current=0,  # make it easier to input the background noise (w/o source noise)
        seeing_mean=2.0,  # make the seeing a little more pronounced
        seeing_std=0,  # keep seeing constant between images
        seeing_minimum=0.5,  # assume seeing never gets much better than this
        transmission_std=0,  # keep transmission constant between images
        pixel_qe_std=0,  # keep pixel QE constant between images
        gain_std=0,  # keep pixel gain constant between images
        star_number=0,
    )

    sim.pars.seeing_mean = 2.5
    sim.make_image(new_sky=True)
    im1 = sim.apply_bias_correction(sim.image)
    im1 -= sim.truth.background_instance
    truth1 = sim.truth

    sim.pars.seeing_mean = 2.5
    sim.make_image(new_sky=True)
    im2 = sim.apply_bias_correction(sim.image)
    im2 -= sim.truth.background_instance
    truth2 = sim.truth

    # what is the best value for the "flux-based zero point"?
    # the flux of a star needs to be this much to provide S/N=1
    # since the PSF is unit normalized, we only need to figure out how much noise in a measurement:
    # the sum over PSF squared times the noise variance gives the noise in a measurement
    F1 = 1 / np.sqrt(np.sum(truth1.psf_downsampled ** 2) * truth1.background_instance)
    F2 = 1 / np.sqrt(np.sum(truth2.psf_downsampled ** 2) * truth2.background_instance)

    zogy_diff, zogy_psf, zogy_score, zogy_score_corr, alpha, alpha_err = zogy_subtract(
        im1,
        im2,
        truth1.psf_downsampled,
        truth2.psf_downsampled,
        np.sqrt(truth1.total_bkg_var),
        np.sqrt(truth2.total_bkg_var),
        F1,
        F2,
    )

    assert abs(np.std(zogy_diff) - 1) < 0.1  # the noise should be unit variance
    assert np.max(abs(zogy_diff)) < threshold  # we should not have anything get to the high threshold
    assert abs( np.max(abs(zogy_diff)) - low_threshold ) < 1.5  # the peak should be close to the low threshold

    # currently this doesn't work, I need to figure out the correct normalization for F_r and F_n
    # assert abs(np.std(zogy_score) - 1) < 0.1  # the noise should be unit variance
    # assert np.max(abs(zogy_score)) < threshold  # we should not have anything get to the high threshold
    # assert abs(np.max(abs(zogy_score)) - low_threshold) < 1.5  # some value should be close to the low threshold

    assert abs(np.std(zogy_score_corr) - 1) < 0.1  # the noise should be unit variance
    assert np.max(abs(zogy_score_corr)) < threshold  # we should not have anything get to the high threshold
    assert abs( np.max(abs(zogy_score_corr)) - low_threshold ) < 1.5  # the peak should be close to the low threshold


def test_subtraction_no_new_sources():
    sim = Simulator(
        image_size_x=imsize,  # not too big, but allow some space for stars
        background_std=0,  # keep background constant between images
        read_noise=0,  # make it easier to input the background noise (w/o source noise)
        seeing_mean=2.0,  # make the seeing a little more pronounced
        seeing_std=0,  # keep seeing constant between images
        seeing_minimum=0.5,  # assume seeing never gets much better than this
        transmission_std=0,  # keep transmission constant between images
        pixel_qe_std=0,  # keep pixel QE constant between images
        gain_std=0,  # keep pixel gain constant between images
        saturation_limit=1e9,  # make it impossible to saturate
        star_number=1000,
    )

    seeing = np.arange(1.0, 3.0, 0.1)
    # seeing = [2.9]
    naive_successes = 0
    zogy_successes = 0
    zogy_failures = 0

    for which in ('R', 'N'):
        for i, s in enumerate(seeing):
            sim.pars.seeing_mean = s if which == 'R' else 1.5
            sim.make_image(new_sky=True)
            truth1 = sim.truth
            im1 = sim.apply_bias_correction(sim.image)
            im1 -= truth1.background_instance
            psf1 = truth1.psf_downsampled
            bkg1 = truth1.total_bkg_var

            sim.pars.seeing_mean = s if which == 'N' else 1.5
            sim.make_image(new_sky=True)
            truth2 = sim.truth
            im2 = sim.apply_bias_correction(sim.image)
            im2 -= truth2.background_instance
            psf2 = truth2.psf_downsampled
            bkg2 = truth2.total_bkg_var

            # need to figure out better values for this
            F1 = 1.0
            F2 = 1.0

            diff = im2 - im1
            diff /= np.sqrt(bkg1 + bkg2)  # adjust the image difference by the noise in both images

            # check that peaks in the matched filter image also obey the same statistics (i.e., we don't find anything)
            matched1 = scipy.signal.convolve(diff, psf1, mode='same') / np.sqrt(np.sum(psf1 ** 2))
            matched2 = scipy.signal.convolve(diff, psf2, mode='same') / np.sqrt(np.sum(psf2 ** 2))

            if (
                np.max(abs(diff)) <= threshold and
                np.max(abs(matched1)) <= threshold and
                np.max(abs(matched2)) <= threshold
            ):
                naive_successes += 1

            # now try ZOGY
            zogy_diff, zogy_psf, zogy_score, zogy_score_corr, alpha, alpha_err = zogy_subtract(
                im1, im2, psf1, psf2, np.sqrt(bkg1), np.sqrt(bkg2), F1, F2,
            )

            # must ignore the edges where sometimes stars are off one image but on the other (if PSF is wide)
            edge = int(np.ceil(max(s, 1.5) * 2))
            if np.max(abs(zogy_score_corr[edge:-edge, edge:-edge])) <= threshold:
                zogy_successes += 1
            else:
                zogy_failures += 1

    assert naive_successes == 0
    assert zogy_failures == 0


def test_subtraction_new_source_snr():
