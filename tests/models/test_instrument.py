import pytest
import time
import re
import uuid
import datetime

import numpy as np

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from models.base import SmartSession
from models.instrument import SensorSection, Instrument, DemoInstrument, DECam


def test_instruments_on_db():
    with SmartSession() as session:
        assert session.scalars(sa.select(Instrument).where(Instrument.name == 'DECam')).first() is not None
        assert session.scalars(sa.select(Instrument).where(Instrument.name == 'DemoInstrument')).first() is not None


def test_modified_instrument():
    try:
        with SmartSession() as session:
            # get the original instrument
            inst_db = session.scalars(sa.select(Instrument).where(Instrument.name == 'DemoInstrument')).first()
            inst_id = inst_db.id
            assert inst_db is not None
            assert isinstance(inst_db, DemoInstrument)

            # modify the aperture size:
            new_aperture = np.random.uniform(1, 2)
            inst_new = DemoInstrument(aperture=new_aperture)

            # new and old instruments are different!
            assert inst_new != inst_db

            # make them the same:
            inst_db.update(inst_new)

            assert inst_new == inst_db
            assert inst_db.aperture == new_aperture
            assert inst_db.focal_ratio == inst_new.focal_ratio
            assert inst_db.id != inst_new.id
            assert inst_db.from_db is True
            assert inst_new.from_db is False
            session.add(inst_db)
            session.commit()

        with SmartSession() as session:
            # check that the database has been updated
            inst_db = session.scalars(sa.select(Instrument).where(Instrument.name == 'DemoInstrument')).first()

            assert inst_db.id == inst_id

            assert inst_new == inst_db
            assert inst_db.aperture == new_aperture
            assert inst_db.focal_ratio == inst_new.focal_ratio
            assert inst_db.id != inst_new.id
            assert inst_db.from_db is True
            assert inst_new.from_db is False

            # modify the list of allow filters
            new_filters = [np.random.choice(list('abcdefghijk')) for _ in range(5)]
            inst_new = DemoInstrument(allowed_filters=new_filters, aperture=new_aperture)

            # new and old instruments are different!
            assert inst_new != inst_db

            # make them the same:
            inst_db.update(inst_new)

            assert inst_new == inst_db
            assert inst_db.aperture == new_aperture
            assert inst_db.allowed_filters == new_filters
            assert inst_db.focal_ratio == inst_new.focal_ratio
            assert inst_db.id != inst_new.id
            assert inst_db.from_db is True
            assert inst_new.from_db is False
            session.add(inst_db)
            session.commit()

        with SmartSession() as session:
            inst_db = session.scalars(sa.select(Instrument).where(Instrument.name == 'DemoInstrument')).first()

            assert inst_db.id == inst_id

            assert inst_new == inst_db
            assert inst_db.aperture == new_aperture
            assert inst_db.allowed_filters == new_filters
            assert inst_db.focal_ratio == inst_new.focal_ratio
            assert inst_db.id != inst_new.id
            assert inst_db.from_db is True
            assert inst_new.from_db is False

    finally:  # return to original state
        with SmartSession() as session:
            DemoInstrument._verify_instrument_on_db(session=session)
            inst_db = session.scalars(sa.select(Instrument).where(Instrument.name == 'DemoInstrument')).first()
            assert inst_db.aperture != new_aperture
            assert inst_db.allowed_filters != new_filters


def test_name_telescope_uniqueness():
    with SmartSession() as session:
        session.add(DemoInstrument())
        with pytest.raises(IntegrityError) as e:
            session.commit()

        assert 'duplicate key value violates unique constraint' in str(e.value)


def test_delete_instrument():
    with SmartSession() as session:
        instruments = session.scalars(sa.select(Instrument)).all()
        assert 'DemoInstrument' in [inst.name for inst in instruments]
        assert 'DECam' in [inst.name for inst in instruments]
        inst = [i for i in instruments if i.name == 'DemoInstrument'][0]
        section_id = inst.sections[0].id  # should be only one section

        assert section_id is not None

        session.delete(inst)
        session.commit()

        section = session.scalars(sa.select(SensorSection).where(SensorSection.id == section_id)).first()
        assert section is None


def test_non_null_constraints():
    class TestInstrument(Instrument):
        def _make_sections(self):
            return []

    inst = TestInstrument()

    try:  # cleanup at the end
        with SmartSession() as session:
            # string attributes
            for att in ['name', 'telescope']:
                with pytest.raises(IntegrityError) as e:
                    session.add(inst)
                    session.commit()
                assert re.search('null value in column ".*" violates not-null constraint', str(e.value))
                session.rollback()
                if att == 'name':
                    setattr(inst, att, 'DemoInstrument')
                else:
                    setattr(inst, att, uuid.uuid4().hex)

            # float attributes
            for att in [
                'aperture',
                'focal_ratio',
                'pixel_scale',
                'square_degree_fov',
                'read_time',
                'read_noise',
                'dark_current',
                'gain',
                'saturation_limit',
                'non_linearity_limit',

            ]:
                with pytest.raises(IntegrityError) as e:
                    session.add(inst)
                    session.commit()
                assert re.search('null value in column ".*" violates not-null constraint', str(e.value))
                session.rollback()
                setattr(inst, att, np.random.uniform(0, 1))

            # string array attributes
            for att in ['allowed_filters']:
                with pytest.raises(IntegrityError) as e:
                    session.add(inst)
                    session.commit()
                assert re.search('null value in column ".*" violates not-null constraint', str(e.value))
                session.rollback()
                setattr(inst, att, [np.random.choice(list('grizy')) for _ in range(3)])

            # should now be ok to add
            session.add(inst)
            session.commit()

    finally:
        # get rid of this instrument

        with SmartSession() as session:
            session.execute(sa.delete(Instrument).where(Instrument.name == 'DemoInstrument'))
            session.commit()
