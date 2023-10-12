import os

import numpy as np

import matplotlib.pyplot as plt

from improc.simulator import Simulator
from improc.sky_flat import calc_sky_flat


def test_simple_sky_flat():
    clear_cache = False  # cache the images from the simulator
    filename = "tests/improc/cache/test_images_flats.npy"
    if os.path.isfile(filename) and not clear_cache:
        images = np.load(filename)
    else:
        sim = Simulator()

        # sim.pars.background_std = 5.0
        images = []
        for i in range(10):
            sim.make_image(new_sky=True, new_stars=True)
            # plt.imshow(sim.image)
            # plt.title(f"Image {i}")
            # plt.show(block=True)
            images.append(sim.apply_bias_correction(sim.image))

        images = np.array(images)
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        np.save(filename, images)

    sky_flat = calc_sky_flat(images, iterations=1)
    plt.imshow(sky_flat)
    plt.title("Sky Flat")
    plt.show(block=True)

