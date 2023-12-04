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