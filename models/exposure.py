from collections import defaultdict

import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import JSONB

from models.base import SeeChangeBase, SpatiallyIndexed


class CCD_Data:
    """
    A helper class that lazy loads the CCD data from the database.
    When requesting one of the CCD IDs it will fetch that data from
    disk and store it in memory.
    To clear the memory cache, call the clear_cache() method.
    """
    def __init__(self, filename, inst):
        self.filename = filename
        self.instrument = inst
        self._data = defaultdict(lambda: None)

    def __getitem__(self, ccd_id):
        if self._data[ccd_id] is None:
            self._data[ccd_id] = self.instrument.load(self.filename, ccd_id)
        return self._data[ccd_id]

    def __setitem__(self, ccd_id, value):
        self._data[ccd_id] = value

    def clear_cache(self):
        self._data = defaultdict(lambda: None)


class Exposure(SeeChangeBase, SpatiallyIndexed):
    __tablename__ = "exposures"

    header = sa.Column(JSONB, nullable=False, doc="Header of the raw exposure. ")

    mjd = sa.Column(
        sa.Double,
        nullable=False,
        index=True,
        doc="Modified Julian date of the exposure. (MJD=JD-2400000.5)"
    )

    exp_time = sa.Column(sa.Float, nullable=False, index=True, doc="Exposure time in seconds")

    filter = sa.Column(sa.Text, nullable=False, index=True, doc="Filter name")

    instrument = sa.Column(sa.Text, nullable=False, index=True, doc="Instrument name")

    telescope = sa.Column(sa.Text, nullable=False, index=True, doc="Telescope name")

    num_ccds = sa.Column(sa.Integer, nullable=False, index=True, doc="Number of CCDs / sections in the full field. ")

    filename = sa.Column(sa.Text, nullable=False, index=True, unique=True, doc="Filename for raw exposure. ")

    project = sa.Column(sa.Text, index=True, nullable=False, doc='Name of the project, (could also be a proposal ID). ')

    target = sa.Column(sa.Text, index=True, nullable=False, doc='Name of the target object or field id. ')

    gallat = sa.Column(sa.Double, index=True, doc="Galactic latitude of the target. ")

    gallon = sa.Column(sa.Double, index=False, doc="Galactic longitude of the target. ")

    ecllat = sa.Column(sa.Double, index=True, doc="Ecliptic latitude of the target. ")

    ecllon = sa.Column(sa.Double, index=False, doc="Ecliptic longitude of the target. ")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data = None

    @orm.reconstructor
    def init_on_load(self):
        self._data = None

    def __repr__(self):
        return f"Exposure({self.id}, {self.exp_time}s, {self.filter}, {self.instrument}, {self.telescope})"

    def __str__(self):
        return self.__repr__()

    def save(self):
        pass  # TODO: implement this!

    def load(self, ccd_ids=None):
        if ccd_ids is None:
            ccd_ids = list(range(self.num_ccds))

        if not isinstance(ccd_ids, list):
            ccd_ids = [ccd_ids]

        if not all([isinstance(ccd_id, int) for ccd_id in ccd_ids]):
            raise ValueError("ccd_ids must be a list of integers. ")

        if self.filename is not None:
            pass  # TODO: implement this!
        else:
            # TODO: get this back to the original ValueError
            # raise ValueError("Cannot load data from database without a filename! ")

            # I've added this fake image generation for testing purposes.
            for i in ccd_ids:
                self._data[i] = np.random.normrnd(0, 1, (512, 1024))

    @property
    def data(self):
        if self._data is None:
            self._data = CCD_Data(self.filename, self.get_instrument_object())
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, CCD_Data):
            raise ValueError(f"data must be a CCD_Data object. Got {type(value)} instead. ")
        self._data = value

    def get_instrument_object(self):
        import models.instruments

        return getattr(models.instruments, self.instrument)()
