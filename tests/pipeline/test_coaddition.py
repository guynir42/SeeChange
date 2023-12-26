
import numpy as np

from improc.simulator import Simulator
from improc.tools import sigma_clipping


def test_zogy_simulation(coadder):
    num_images = 10
    sim = Simulator(
        image_size_x=256,  # make smaller images to make the test faster
        star_number=100,  # the smaller images require a smaller number of stars to avoid crowding
        seeing_mean=3.0,
        seeing_std=1.0,
        gain_std=0,  # leave the gain at 1.0
        read_noise=1,
        optic_psf_pars={'sigma': 0.5}, # make the optical PSF much smaller than the seeing
    )
    images = []
    weights = []
    flags = []
    truths = []
    psfs = []
    fwhms = []
    zps = []
    bkg_means = []
    bkg_stds = []
    for i in range(num_images):
        sim.make_image(new_sky=True, new_stars=False)
        images.append(sim.apply_bias_correction(sim.image))
        weights.append(np.ones_like(sim.image))
        flags.append(np.zeros_like(sim.image, dtype=np.int16))
        flags[-1][100, 100] = 1  # just to see what happens to a flagged pixel
        truths.append(sim.truth)
        psfs.append(sim.truth.psf_downsampled)
        fwhms.append(sim.truth.atmos_psf_fwhm)
        zps.append(sim.truth.transmission_instance)
        bkg_means.append(sim.truth.background_instance)
        bkg_stds.append(np.sqrt(sim.truth.background_instance + sim.truth.read_noise ** 2))

    outim, outwt, outfl, outpsf, score = coadder._coadd_zogy(
        images,
        weights,
        flags,
        psfs,
        fwhms,
        zps,
        bkg_means,
        bkg_stds,
    )

    assert outim.shape == (256, 256)
    assert outwt.shape == (256, 256)
    assert outfl.shape == (256, 256)

    assert np.sum(outfl) > 1  # there should be more than one flagged pixel (PSF splash)
    assert np.sum(outfl) < 100  # there should be fewer than 100 flagged pixels

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(images[0], vmin=0, vmax=100)
    ax[0, 0].set_title('original 0')
    ax[0, 1].imshow(images[1], vmin=0, vmax=100)
    ax[0, 1].set_title('original 1')

    ax[1, 0].imshow(outim)
    ax[1, 0].set_title('coadded')

    ax[1, 1].imshow(score)
    ax[1, 1].set_title('score')

    plt.show(block=True)


def test_zogy_vs_naive(ptf_aligned_images, coadder):
    assert all([im.psf is not None for im in ptf_aligned_images])
    assert all([im.zp is not None for im in ptf_aligned_images])

    naive_im, naive_wt, naive_fl = coadder._coadd_naive(ptf_aligned_images)

    zogy_im, zogy_wt, zogy_fl, zogy_psf, zogy_score = coadder._coadd_zogy(ptf_aligned_images)

    assert naive_im.shape == zogy_im.shape

    # ZOGY must dilate the bad pixels to acount for PSF match-filtering:
    assert np.sum(naive_fl == 1) < np.sum(zogy_fl == 1)  # more bad pixels
    assert np.sum(naive_fl == 0) > np.sum(zogy_fl == 0)  # less good pixels

    mu, sigma = sigma_clipping(zogy_im)
    print(mu, sigma)
