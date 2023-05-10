import pytest
import re

import sqlalchemy as sa
import numpy as np
from datetime import datetime

from astropy.time import Time

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from models.base import SmartSession
from models.exposure import Exposure, SectionData

from models.instrument import Instrument, DECam, DemoInstrument


def rnd_str(n):
    return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n))


def test_exposure_no_null_values():

    # cannot create an exposure without a filename!
    with pytest.raises(ValueError, match='Must give a filename'):
        e = Exposure()

    required = {
        'mjd': 58392.1,
        'exp_time': 30,
        'filter': 'r',
        'ra': np.random.uniform(0, 360),
        'dec': np.random.uniform(-90, 90),
        'instrument_id': 1,
        'project': 'foo',
        'target': 'bar',
    }

    added = {}

    # use non-capturing groups to extract the column name from the error message
    expr = r'(?:null value in column )(".*")(?: of relation "exposures" violates not-null constraint)'

    try:
        exposure_id = None  # make sure to delete the exposure if it is added to DB
        with SmartSession() as session:
            for i in range(len(required) - 1):
                e = Exposure(f"Demo_test_{rnd_str(5)}.fits", **added)
                with pytest.raises(IntegrityError) as exc:
                    session.add(e)
                    session.commit()
                    exposure_id = e.id
                session.rollback()

                if "check constraint" in str(exc.value):
                    # the constraint on the filter is either filter or filter array must be not-null
                    colname = 'filter'
                else:

                    match_obj = re.search(expr, str(exc.value))
                    assert match_obj is not None

                    # find which column raised the error
                    colname = match_obj.group(1).replace('"', '')

                # add missing column name:
                added.update({colname: required[colname]})

        e = Exposure(f"Demo_test_{rnd_str(5)}.fits", **added)
        session.add(e)
        session.commit()
        exposure_id = e.id
        assert exposure_id is not None

    finally:
        # cleanup
        with SmartSession() as session:
            exposure = None
            if exposure_id is not None:
                exposure = session.scalars(sa.select(Exposure).where(Exposure.id == exposure_id)).first()
            if exposure is not None:
                session.delete(exposure)
                session.commit()


def test_exposure_must_have_instrument():
    try:
        exposure_id = None  # make sure to delete the exposure if it is added to DB
        with SmartSession() as session:
            e = Exposure(
                f"Demo_test_{rnd_str(5)}.fits",
                mjd=58392.1,
                exp_time=30,
                filter='r',
                ra=123,
                dec=-23,
                project='foo',
                target='bar',
            )
            e.instrument_id = None
            with pytest.raises(IntegrityError) as exc:
                session.add(e)
                session.commit()
                exposure_id = e.id
            session.rollback()
            assert 'column "instrument_id" of relation "exposures" violates not-null constraint' in str(exc.value)

    finally:
        # cleanup
        with SmartSession() as session:
            exposure = None
            if exposure_id is not None:
                exposure = session.scalars(sa.select(Exposure).where(Exposure.id == exposure_id)).first()
            if exposure is not None:
                session.delete(exposure)
                session.commit()


def test_provide_explicit_instrument_id():
    try:
        exposure_id = None  # make sure to delete the exposure if it is added to DB
        with SmartSession() as session:
            # get the ID of the DECam instrument:
            inst = session.scalars(sa.select(Instrument)).all()
            inst_id = [i.id for i in inst if i.name == 'DECam'][0]

            e = Exposure(
                f"{rnd_str(15)}.fits",
                mjd=58392.1,
                exp_time=30,
                filter='r',
                ra=123,
                dec=-23,
                project='foo',
                target='bar',
                instrument_id=inst_id,
            )
            session.add(e)
            session.commit()
            exposure_id = e.id
            assert exposure_id is not None
            assert e.instrument_name == 'DECam'

    finally:
        # cleanup
        with SmartSession() as session:
            exposure = None
            if exposure_id is not None:
                exposure = session.scalars(sa.select(Exposure).where(Exposure.id == exposure_id)).first()
            if exposure is not None:
                session.delete(exposure)
                session.commit()


def test_exposure_guess_demo_instrument():
    e = Exposure(f"Demo_test_{rnd_str(5)}.fits", exp_time=30, mjd=58392.0, filter="F160W", ra=123, dec=-23,
                 project='foo', target='bar')

    # make sure instruments are loaded into DB:
    with SmartSession() as session:
        inst = session.scalars(sa.select(Instrument)).all()
        demo_id = [i.id for i in inst if i.name == 'DemoInstrument'][0]

    e.guess_instrument()

    assert e.instrument_id == demo_id


def test_exposure_guess_decam_instrument():

    t = datetime.now()
    mjd = Time(t).mjd
    time_str = t.strftime("%Y%m%d_%H%M%S")
    e = Exposure(f"c4d_{time_str}_ori.fits", exp_time=30, mjd=mjd, filter="r", ra=123, dec=-23,
                 project='foo', target='bar')

    # make sure instruments are loaded into DB:
    with SmartSession() as session:
        inst = session.scalars(sa.select(Instrument)).all()
        decam_id = [i.id for i in inst if i.name == 'DECam'][0]

    e.guess_instrument()

    assert e.instrument_id == decam_id


def test_coordinates():
    e = Exposure('foo.fits', ra=None, dec=None)
    assert e.ecllat is None
    assert e.ecllon is None
    assert e.gallat is None
    assert e.gallon is None

    with pytest.raises(ValueError, match='Exposure must have RA and Dec set'):
        e.calculate_coordinates()

    e = Exposure('foo.fits', ra=123.4, dec=None)
    assert e.ecllat is None
    assert e.ecllon is None
    assert e.gallat is None
    assert e.gallon is None

    e = Exposure('foo.fits', ra=123.4, dec=56.78)
    assert abs(e.ecllat - 35.846) < 0.01
    assert abs(e.ecllon - 111.838) < 0.01
    assert abs(e.gallat - 33.542) < 0.01
    assert abs(e.gallon - 160.922) < 0.01


def test_load_exposure_demo_instrument_data(exposure):
    # this is a new exposure, created as a fixture (not from DB):
    assert exposure.from_db == 0

    # instrument is not yet loaded, as exposure is not yet added to DB:
    assert exposure.instrument is None

    exposure.fetch_instrument()  # internally open/close a session
    assert exposure.instrument is not None

    # the data is a SectionData object that lazy loads from file
    assert isinstance(exposure.data, SectionData)
    assert exposure.data.filename == exposure.filename
    assert exposure.data.instrument == exposure.instrument

    # loading the first (and only!) section data gives a random array
    array = exposure.data[0]
    assert isinstance(array, np.ndarray)

    # this array is random integers, but it is not re-generated each time:
    assert np.array_equal(array, exposure.data[0])

    inst = exposure.instrument
    assert array.shape == (inst.sections[0].size_y, inst.sections[0].size_x)

    # check that we can clear the cache and "re-load" data:
    exposure.data.clear_cache()
    assert not np.array_equal(array, exposure.data[0])


# TODO: here's a list of tests that need to be added when more functionality is added:
#  - test loading data from a FITS file (e.g., using the DECam instrument)
#  - test reading a header from FITS (e.g., using the DECam instrument)
#  - test that header keys with different formatting (e.g., 'EXPTIME' vs 'exptime') are recognized
