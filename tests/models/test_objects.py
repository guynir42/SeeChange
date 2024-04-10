import pytest
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
    for lc in sim_lightcurves:
        expected_flux = []
        expected_error = []
        measured_flux = []

        for m in lc:
            measured_flux.append(m.flux_apertures[3] - m.background * m.area_apertures[3])
            expected_flux.append(m.sources.data['flux'][m.cutouts.index_in_sources])
            expected_error.append(m.sources.data['flux_err'][m.cutouts.index_in_sources])

        assert len(expected_flux) == len(measured_flux)
        for i in range(len(measured_flux)):
            assert measured_flux[i] == pytest.approx(expected_flux[i], abs=expected_error[i] * 3)
