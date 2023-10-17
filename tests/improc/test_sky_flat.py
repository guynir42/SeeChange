import os
import time
import pytest

import numpy as np

import matplotlib.pyplot as plt

from models.base import CODE_ROOT

from improc.simulator import Simulator
from improc.sky_flat import calc_sky_flat


# @pytest.mark.parametrize("num_images", [10, 100, 1000])
@pytest.mark.parametrize("num_images", [100])
def test_simple_sky_flat(num_images):
    clear_cache = True  # cache the images from the simulator
    filename = os.path.join(CODE_ROOT, f"tests/improc/cache/flat_test_images_{num_images}.npz")
    sim = Simulator(image_size_x=256, vignette_inner_radius=1500, pixel_qe_std=0.00, star_number=0)

    if os.path.isfile(filename) and not clear_cache:
        file_obj = np.load(filename, allow_pickle=True)
        images = file_obj['images']
        sim.truth = file_obj['truth'][()]
    else:
        t0 = time.time()
        images = []
        for i in range(num_images):
            sim.make_image(new_sky=True, new_stars=True)
            # plt.imshow(sim.image)
            # plt.title(f"Image {i}")
            # plt.show(block=True)
            images.append(sim.apply_bias_correction(sim.image))

        images = np.array(images)
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        np.savez(filename, images=images, truth=sim.truth)
        print(f"Generating {num_images} images took {time.time() - t0:.1f} seconds")

    t0 = time.time()
    sky_flat = calc_sky_flat(images, nsigma=8.0, iterations=1)
    print(f'calc_sky_flat took {time.time() - t0:.1f} seconds')

    plt.plot(sky_flat[10, :], label="sky flat")
    # plt.plot(sim.sensor.pixel_qe_map[100, :], label="sensor QE")
    plt.plot(sim.truth.vignette_map[10, :], label="camera vignette")
    plt.plot(sim.truth.vignette_map[10, :] * sim.truth.pixel_qe_map[10, :], label="expected flat")

    plt.legend()
    plt.show(block=True)

    delta = (sky_flat - sim.truth.vignette_map * sim.truth.pixel_qe_map)
    # print(delta[:3, :3])
    # print(np.sqrt(np.sum(delta ** 2)))
    print(f'mean(delta)= {np.nanmean(delta)}, std(delta)= {np.nanstd(delta)}')

    # delta[abs(delta) > 0.3] = np.nan
    # plt.hist(delta.flatten(), bins=100)
    # plt.show(block=True)

