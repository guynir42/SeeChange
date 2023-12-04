import os
import warnings

import pytest

from models.base import _logger
from models.provenance import Provenance
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint

from pipeline.data_store import DataStore
from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from pipeline.subtraction import Subtractor
from pipeline.cutting import Cutter
from pipeline.measurement import Measurer


@pytest.fixture
def preprocessor(config_test):
    prep = Preprocessor(**config_test.value('preprocessing'))
    prep.pars._enforce_no_new_attrs = False
    prep.pars.test_parameter = prep.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    prep.pars._enforce_no_new_attrs = True

    return prep


@pytest.fixture
def extractor(config_test):
    extr = Detector(**config_test.value('extraction'))
    extr.pars._enforce_no_new_attrs = False
    extr.pars.test_parameter = extr.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    extr.pars._enforce_no_new_attrs = True

    return extr


@pytest.fixture
def astrometor(config_test):
    astrom = AstroCalibrator(**config_test.value('astro_cal'))
    astrom.pars._enforce_no_new_attrs = False
    astrom.pars.test_parameter = astrom.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    astrom.pars._enforce_no_new_attrs = True

    return astrom


@pytest.fixture
def photometor(config_test):
    photom = PhotCalibrator(**config_test.value('photo_cal'))
    photom.pars._enforce_no_new_attrs = False
    photom.pars.test_parameter = photom.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    photom.pars._enforce_no_new_attrs = True

    return photom


@pytest.fixture
def subtractor(config_test):
    sub = Subtractor(**config_test.value('subtraction'))
    sub.pars._enforce_no_new_attrs = False
    sub.pars.test_parameter = sub.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    sub.pars._enforce_no_new_attrs = True

    return sub


@pytest.fixture
def detector(config_test):
    det = Detector(**config_test.value('detection'))
    det.pars._enforce_no_new_attrs = False
    det.pars.test_parameter = det.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    det.pars._enforce_no_new_attrs = False

    return det


@pytest.fixture
def cutter(config_test):
    cut = Cutter(**config_test.value('cutting'))
    cut.pars._enforce_no_new_attrs = False
    cut.pars.test_parameter = cut.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    cut.pars._enforce_no_new_attrs = False

    return cut


@pytest.fixture
def measurer(config_test):
    meas = Measurer(**config_test.value('measurement'))
    meas.pars._enforce_no_new_attrs = False
    meas.pars.test_parameter = meas.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    meas.pars._enforce_no_new_attrs = False

    return meas


@pytest.fixture
def datastore_factory(
        data_dir,
        preprocessor,
        extractor,
        astrometor,
        photometor,
        detector,
        cutter,
        measurer,
):
    """Provide a function that returns a datastore with all the products based on the given exposure and section ID.

    To use this data store in a test where new data is to be generated,
    simply change the pipeline object's "test_parameter" value to a unique
    new value, so the provenance will not match and the data will be regenerated.

    EXAMPLE
    -------
    extractor.pars.test_parameter = uuid.uuid().hex
    extractor.run(datastore)
    assert extractor.has_recalculated is True
    """
    def make_datastore(exposure, section_id, cache_dir=None, cache_base_name=None):
        code_version = exposure.provenance.code_version
        ds = DataStore(exposure, section_id)  # make a new datastore

        ############ preprocessing to create image ############

        if cache_dir is not None and cache_base_name is not None:
            # check if preprocessed image is in cache
            cache_name = cache_base_name + '.image.fits.json'
            cache_path = os.path.join(cache_dir, cache_name)
            if os.path.isfile(cache_path):
                _logger.debug('loading image from cache. ')
                ds.image = Image.copy_from_cache(cache_dir, cache_name)
                ds.image.provenance = Provenance(
                    code_version=code_version,
                    process='preprocessing',
                    upstreams=[ds.exposure.provenance],
                    parameters=preprocessor.pars.to_dict(),
                )

        if ds.image is None:  # make the preprocessed image
            _logger.debug('making preprocessed image. ')
            ds = preprocessor.run(ds)
            ds.image.save()
            output_path = ds.image.copy_to_cache(cache_dir)
            if cache_dir is not None and cache_base_name is not None and output_path != cache_path:
                warnings.warn(f'cache path {cache_path} does not match output path {output_path}')

        ############# extraction to create sources #############
        if cache_dir is not None and cache_base_name is not None:
            cache_name = cache_base_name + '.sources.fits.json'
            cache_path = os.path.join(cache_dir, cache_name)
            if os.path.isfile(cache_path):
                _logger.debug('loading source list from cache. ')
                ds.sources = SourceList.copy_from_cache(cache_dir, cache_name)
                ds.sources.image = ds.image
                ds.sources.provenance = Provenance(
                    code_version=code_version,
                    process='extraction',
                    upstreams=[ds.image.provenance],
                    parameters=extractor.pars.to_dict(),
                )
            cache_name = cache_base_name + '.psf.json'
            cache_path = os.path.join(cache_dir, cache_name)
            if os.path.isfile(cache_path):
                _logger.debug('loading PSF from cache. ')
                ds.psf = PSF.copy_from_cache(cache_dir, cache_name)
                ds.psf.image = ds.image
                ds.psf.provenance = Provenance(
                    code_version=code_version,
                    process='extraction',
                    upstreams=[ds.image.provenance],
                    parameters=extractor.pars.to_dict(),
                )

        if ds.sources is None or ds.psf is None:  # make the source list from the regular image
            _logger.debug('extracting sources. ')
            ds = extractor.run(ds)
            ds.sources.save()
            ds.sources.copy_to_cache(cache_dir)
            ds.psf.save(overwrite=True)
            output_path = ds.psf.copy_to_cache(cache_dir)
            if cache_dir is not None and cache_base_name is not None and output_path != cache_path:
                warnings.warn(f'cache path {cache_path} does not match output path {output_path}')

        ############## astro_cal to create wcs ################

        if cache_dir is not None and cache_base_name is not None:
            cache_name = cache_base_name + '.wcs.json'
            cache_path = os.path.join(cache_dir, cache_name)
            if os.path.isfile(cache_path):
                _logger.debug('loading WCS from cache. ')
                ds.wcs = WorldCoordinates.copy_from_cache(cache_dir, cache_name)
                ds.wcs.sources = ds.sources
                ds.wcs.provenance = Provenance(
                    code_version=code_version,
                    process='astro_cal',
                    upstreams=[ds.sources.provenance],
                    parameters=astrometor.pars.to_dict(),
                )

        if ds.wcs is None:  # make the WCS
            _logger.debug('Running astrometric calibration')
            ds = astrometor.run(ds)
            if cache_dir is not None and cache_base_name is not None:
                output_path = ds.wcs.copy_to_cache(cache_dir, cache_name)  # must provide a name because this one isn't a FileOnDiskMixin
                if output_path != cache_path:
                    warnings.warn(f'cache path {cache_path} does not match output path {output_path}')

        ########### photo_cal to create zero point ############

        if cache_dir is not None and cache_base_name is not None:
            cache_name = cache_base_name + '.zp.json'
            cache_path = os.path.join(cache_dir, cache_name)
            if os.path.isfile(cache_path):
                _logger.debug('loading zero point from cache. ')
                ds.zp = ZeroPoint.copy_from_cache(cache_dir, cache_name)
                ds.zp.sources = ds.sources
                ds.zp.provenance = Provenance(
                    code_version=code_version,
                    process='photo_cal',
                    upstreams=[ds.sources.provenance],
                    parameters=photometor.pars.to_dict(),
                )

        if ds.zp is None:  # make the zero point
            _logger.debug('Running photometric calibration')
            ds = photometor.run(ds)
            if cache_dir is not None and cache_base_name is not None:
                output_path = ds.zp.copy_to_cache(cache_dir, cache_name)
                if output_path != cache_path:
                    warnings.warn(f'cache path {cache_path} does not match output path {output_path}')

        # TODO: add the same cache/load and processing for the rest of the pipeline

        return ds

    return make_datastore