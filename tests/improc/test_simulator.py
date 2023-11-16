import os
import pytest
import time

import numpy as np

import matplotlib.pyplot as plt

from models.base import CODE_ROOT
from improc.simulator import Simulator, SimGalaxies
from improc.sky_flat import sigma_clipping

# uncomment this to run the plotting tests interactively
os.environ['INTERACTIVE'] = '1'


@pytest.mark.skipif( os.getenv('INTERACTIVE') is None, reason='Set INTERACTIVE to run this test' )
def test_make_star_field(blocking_plots):
    s = Simulator( image_size_x=256, star_number=1000, galaxy_number=0)
    s.make_image()
    vmin = np.percentile(s.image, 1)
    vmax = np.percentile(s.image, 99)

    plt.imshow(s.image, vmin=vmin, vmax=vmax)
    plt.show(block=blocking_plots)

    filename = os.path.join(CODE_ROOT, 'tests/plots/simulator_star_field')
    plt.savefig(filename+'.png')
    plt.savefig(filename+'.pdf')


@pytest.mark.skipif( os.getenv('INTERACTIVE') is None, reason='Set INTERACTIVE to run this test' )
def test_make_galaxy_field(blocking_plots):
    s = Simulator( image_size_x=256, star_number=0, galaxy_number=1000, galaxy_min_width=1, galaxy_min_flux=1000 )
    t0 = time.time()
    s.make_image()
    print(s.truth.oversampling)
    print(f'Generating image took {time.time() - t0:.1f} seconds')

    mu, sig = sigma_clipping(s.image.astype(float))

    plt.imshow(s.image, vmin=mu-sig, vmax=mu+10*sig)

    plt.show(block=blocking_plots)

    filename = os.path.join(CODE_ROOT, 'tests/plots/simulator_galaxy_field')
    plt.savefig(filename+'.png')
    plt.savefig(filename+'.pdf')


@pytest.mark.parametrize("exp_scale", [5.0, 500.0])
def test_making_galaxy_in_image(exp_scale, blocking_plots):
    center_x = 33.123
    center_y = 76.456
    imsize = (256, 256)
    sersic_scale = 5.0
    exp_flux = 1000
    sersic_flux = 0
    rotation = 33.0
    cos_i = 0.5
    cutoff_radius = exp_scale * 5.0

    im1 = SimGalaxies.make_galaxy_image(
        imsize=imsize,
        center_x=center_x,
        center_y=center_y,
        exp_scale=exp_scale,
        sersic_scale=sersic_scale,
        exp_flux=exp_flux,
        sersic_flux=sersic_flux,
        cos_i=cos_i,
        rotation=rotation,
        cutoff_radius=cutoff_radius,
    )

    im2 = np.zeros(imsize)
    SimGalaxies.add_galaxy_to_image(
        image=im2,
        center_x=center_x,
        center_y=center_y,
        exp_scale=exp_scale,
        sersic_scale=sersic_scale,
        exp_flux=exp_flux,
        sersic_flux=sersic_flux,
        cos_i=cos_i,
        rotation=rotation,
        cutoff_radius=cutoff_radius,
    )

    diff = im1 - im2

    if blocking_plots:
        fig, ax = plt.subplots(1, 3)

        ax[0].imshow(im1)
        ax[0].set_title('make_galaxy_image')

        ax[1].imshow(im2)
        ax[1].set_title('add_galaxy_to_image')

        h = ax[2].imshow(np.log10(abs(diff)))
        ax[2].set_title('log10 of difference')
        plt.colorbar(h, ax=ax[2], orientation='vertical')

        plt.show(block=True)

    peak1 = np.unravel_index(im1.argmax(), im1.shape)
    peak2 = np.unravel_index(im2.argmax(), im2.shape)

    assert peak1 == peak2
    assert np.max(diff) < 1e-5
