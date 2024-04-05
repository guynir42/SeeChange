import re
import sqlalchemy as sa

from models.base import SmartSession
from models.objects import Object


def test_object_creation():
    obj = Object(ra=1.0, dec=2.0, is_test=True)
    with SmartSession() as session:
        session.add(obj)
        session.commit()
        assert obj.id is not None
        assert obj.name is not None
        assert re.match(r'\w+\d{4}\w+', obj.name)

    with SmartSession() as session:
        obj2 = session.scalars(sa.select(Object).where(Object.id == obj.id)).first()
        assert obj2.ra == 1.0
        assert obj2.dec == 2.0
        assert obj2.name is not None
        assert re.match(r'\w+\d{4}\w+', obj2.name)


def test_objects_from_measurements(sim_lightcurves):
    pass
