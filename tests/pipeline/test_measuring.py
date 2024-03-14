import numpy as np


def test_measuring(measurer, decam_cutouts):
    ds = measurer.run(decam_cutouts)

    assert len(ds.measurements) == len(ds.cutouts)

    # TODO: should we move these tests to models/test_measurements.py ?

    # grab one example measurements object
    m = ds.measurements[0]
    new_im = m.cutouts.sources.image.new_image
    assert np.array_equal(m.aper_radii, new_im.zp.aper_cor_radii)
    assert np.array_equal(
        new_im.zp.aper_cor_radii,
        new_im.psf.fwhm_pixels * np.array(new_im.instrument_object.standard_apertures()),
    )

    original_flux = m.flux_apertures[m.best_aperture]

    # set the flux temporarily to something positive
    m.flux_apertures[m.best_aperture] = 1000
    assert m.magnitude == -2.5 * np.log10(1000) + new_im.zp.zp + new_im.zp.aper_cors[m.best_aperture]

    # set the flux temporarily to something negative
    m.flux_apertures[m.best_aperture] = -1000
    assert np.isnan(m.magnitude)

    # set the flux and zero point to some randomly chosen values and test the distribution of the magnitude:
    fiducial_zp = new_im.zp.zp
    original_zp_err = new_im.zp.dzp
    fiducial_zp_err = 0.1  # more reasonable ZP error value
    fiducial_flux = 1000
    fiducial_flux_err = 50
    m.flux_apertures_err[m.best_aperture] = fiducial_flux_err
    new_im.zp.dzp = fiducial_zp_err

    iterations = 1000
    mags = np.zeros(iterations)
    for i in range(iterations):
        m.flux_apertures[m.best_aperture] = np.random.normal(fiducial_flux, fiducial_flux_err)
        new_im.zp.zp = np.random.normal(fiducial_zp, fiducial_zp_err)
        mags[i] = m.magnitude

    m.flux_apertures[m.best_aperture] = fiducial_flux

    # the measured magnitudes should be normally distributed
    assert np.abs(np.std(mags) - m.magnitude_err) < 0.01
    assert np.abs(np.mean(mags) - m.magnitude) < m.magnitude_err * 3

    # make sure to return things to their original state
    m.flux_apertures[m.best_aperture] = original_flux
    new_im.zp.dzp = original_zp_err

    # TODO: add test for limiting magnitude (issue #143)

    # test that we cannot save the same measurements object twice

    print(ds)