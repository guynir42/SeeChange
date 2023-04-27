import pytest
import uuid

import sqlalchemy as sa

from models.base import SmartSession
from models.instrument import SensorSection, Instrument, DemoInstrument, DECam


def test_instruments_on_db():
    with SmartSession() as session:
        assert session.scalars(sa.select(Instrument).where(Instrument.name == 'DECam')).first() is not None
        assert session.scalars(sa.select(Instrument).where(Instrument.name == 'DemoInstrument')).first() is not None
