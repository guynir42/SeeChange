import pytest

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
    prep.pars.test_parameter = 'test_value'  # replace with uuid to make a new provenance
    prep.pars._enforce_no_new_attrs = True

    return prep


@pytest.fixture
def extractor(config_test):
    extr = Detector(**config_test.value('extraction'))
    extr.pars._enforce_no_new_attrs = False
    extr.pars.test_parameter = 'test_value'  # replace with uuid to make a new provenance
    extr.pars._enforce_no_new_attrs = True

    return extr


@pytest.fixture
def astrometor(config_test):
    astrom = AstroCalibrator(**config_test.value('astro_cal'))
    astrom.pars._enforce_no_new_attrs = False
    astrom.pars.test_parameter = 'test_value'  # replace with uuid to make a new provenance
    astrom.pars._enforce_no_new_attrs = True

    return astrom


@pytest.fixture
def photometor(config_test):
    photom = PhotCalibrator(**config_test.value('phot_cal'))
    photom.pars._enforce_no_new_attrs = False
    photom.pars.test_parameter = 'test_value'  # replace with uuid to make a new provenance
    photom.pars._enforce_no_new_attrs = True

    return photom


@pytest.fixture
def subtractor(config_test):
    sub = Subtractor(**config_test.value('subtraction'))
    sub.pars._enforce_no_new_attrs = False
    sub.pars.test_parameter = 'test_value'  # replace with uuid to make a new provenance
    sub.pars._enforce_no_new_attrs = True

    return sub


@pytest.fixture
def detector(config_test):
    det = Detector(**config_test.value('detection'))
    det.pars._enforce_no_new_attrs = False
    det.pars.test_parameter = 'test_value'
    det.pars._enforce_no_new_attrs = False

    return det


@pytest.fixture
def cutter(config_test):
    cut = Cutter(**config_test.value('cutout'))
    cut.pars._enforce_no_new_attrs = False
    cut.pars.test_parameter = 'test_value'
    cut.pars._enforce_no_new_attrs = False

    return cut


@pytest.fixture
def measurer(config_test):
    meas = Measurer(**config_test.value('measurement'))
    meas.pars._enforce_no_new_attrs = False
    meas.pars.test_parameter = 'test_value'
    meas.pars._enforce_no_new_attrs = False

    return meas