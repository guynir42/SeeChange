
import numpy as np


def test_zogy_vs_naive(ptf_aligned_images, coadder):
    assert all([im.psf is not None for im in ptf_aligned_images])
    assert all([im.zp is not None for im in ptf_aligned_images])

    naive_im, naive_fl, naive_wt = coadder._coadd_naive(ptf_aligned_images)

    zogy_im, zogy_fl, zogy_wt, zogy_psf, zogy_score = coadder._coadd_zogy(ptf_aligned_images)