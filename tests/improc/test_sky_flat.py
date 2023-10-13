import os
import time
import pytest

import numpy as np

import matplotlib.pyplot as plt

from models.base import CODE_ROOT

from improc.simulator import Simulator
from improc.sky_flat import calc_sky_flat


@pytest.mark.parametrize("N", [10, 100, 1000])
def test_simple_sky_flat(N):
    clear_cache = False  # cache the images from the simulator
    filename = os.path.join(CODE_ROOT, f"tests/improc/cache/flat_test_images_{N}.npy")
    sim = Simulator()

    if os.path.isfile(filename) and not clear_cache:
        sim.make_image()
        images = np.load(filename)
    else:
        images = []
        for i in range(N):
            sim.make_image(new_sky=True, new_stars=True)
            # plt.imshow(sim.image)
            # plt.title(f"Image {i}")
            # plt.show(block=True)
            images.append(sim.apply_bias_correction(sim.image))

        images = np.array(images)
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        np.save(filename, images)

    t0 = time.time()
    sky_flat = calc_sky_flat(images, nsigma=2.0, iterations=10)
    print(f'calc_sky_flat took {time.time() - t0:.1f} seconds')

    # plt.imshow(sim.sensor.pixel_qe_map)
    # plt.title('sensor QE')
    # plt.show(block=True)

    # plt.imshow(sim.camera.vignette_map)
    # plt.title('camera vignette')
    # plt.show(block=True)

    # plt.imshow(sky_flat)
    # plt.title("Sky Flat")
    # plt.show(block=True)

    plt.plot(sky_flat[100, :], label="sky flat")
    plt.plot(sim.sensor.pixel_qe_map[100, :], label="sensor QE")
    plt.plot(sim.camera.vignette_map[100, :], label="camera vignette")
    plt.legend()
    plt.show(block=False)

    delta = (sky_flat - sim.camera.vignette_map) ** 2
    print(delta[:10, :10])
    print(np.sqrt(np.sum(delta)))

