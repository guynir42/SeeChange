
import inspect
from collections import defaultdict

import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import JSONB

from astropy.coordinates import SkyCoord

from models.base import Base, SpatiallyIndexed
import models.instrument

from pipeline.utils import normalize_header_key

class CCD_Data:
    """
    A helper class that lazy loads the CCD data from the database.
    When requesting one of the CCD IDs it will fetch that data from
    disk and store it in memory.
    To clear the memory cache, call the clear_cache() method.
    """
    def __init__(self, filename, inst):
        self.filename = filename
        self.instrument_object = inst
        self._data = defaultdict(lambda: None)

    def __getitem__(self, ccd_id):
        if self._data[ccd_id] is None:
            self._data[ccd_id] = self.instrument_object.load(self.filename, ccd_id)
        return self._data[ccd_id]

    def __setitem__(self, ccd_id, value):
        self._data[ccd_id] = value

    def clear_cache(self):
        self._data = defaultdict(lambda: None)


class Exposure(Base, SpatiallyIndexed):

    __tablename__ = "exposures"

    header = sa.Column(JSONB, nullable=False, default={}, doc="Header of the raw exposure. ")

    filename = sa.Column(sa.Text, nullable=False, index=True, unique=True, doc="Filename for raw exposure. ")

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

    num_ccds = sa.Column(sa.Integer, nullable=False, index=False, doc="Number of CCDs / sections in the full field. ")

    width = sa.Column(sa.Integer, nullable=False, index=False, doc="Width of each CCD in pixels. ")

    height = sa.Column(sa.Integer, nullable=False, index=False, doc="Height of each CCD in pixels. ")

    project = sa.Column(sa.Text, index=True, nullable=False, doc='Name of the project, (could also be a proposal ID). ')

    target = sa.Column(sa.Text, index=True, nullable=False, doc='Name of the target object or field id. ')

    gallat = sa.Column(sa.Double, index=True, doc="Galactic latitude of the target. ")

    gallon = sa.Column(sa.Double, index=False, doc="Galactic longitude of the target. ")

    ecllat = sa.Column(sa.Double, index=True, doc="Ecliptic latitude of the target. ")

    ecllon = sa.Column(sa.Double, index=False, doc="Ecliptic longitude of the target. ")

    def __init__(self, *args, **kwargs):
        """
        Initialize the exposure object.
        Can give the filename of the exposure
        as the single positional argument.

        Otherwise, give any arguments that are
        columns of the Exposure table.

        If the filename is given, it will parse
        the instrument name from the filename.
        The header will be read out from the FITS file.
        """
        super().__init__(**kwargs)  # put keywords into columns

        self._data = None
        self.instrument_object = None

        if len(args) == 1 and isinstance(args[0], str):
            self.filename = args[0]

        if self.instrument is None:
            self.parse_instrument_name()

        # find additional column values from the header
        self.read_header()

        self.num_ccds = self.get_instrument_object().get_num_ccds()

        # override the header values with any given keyword arguments?
        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)

        if self.ra is not None and self.dec is not None:
            self.calculate_coordinates()

    @orm.reconstructor
    def init_on_load(self):
        self._data = None
        self.instrument_object = None

    def parse_instrument_name(self):
        if self.filename is None:
            return

        for name, class_ in inspect.getmembers(models.instrument, inspect.isclass):
            if name.lower() in self.filename.lower():
                self.instrument = name
                self.instrument_object = class_()
                break

    def __repr__(self):
        return f"Exposure(id: {self.id}, exp: {self.exp_time}s, filt: {self.filter}, from: {self.instrument}/{self.telescope})"

    def __str__(self):
        return self.__repr__()

    def save(self):
        pass  # TODO: implement this! do we need this?

    def load(self, ccd_ids=None):
        if ccd_ids is None:
            ccd_ids = list(range(self.num_ccds))

        if not isinstance(ccd_ids, list):
            ccd_ids = [ccd_ids]

        if not all([isinstance(ccd_id, int) for ccd_id in ccd_ids]):
            raise ValueError("ccd_ids must be a list of integers. ")

        if self.filename is not None:
            for i in ccd_ids:
                self.data[i]  # use the CCD_Data __getitem__ method to load the data
        else:
            raise ValueError("Cannot load data from database without a filename! ")

    @property
    def data(self):
        if self._data is None:
            inst = self.get_instrument_object()
            if inst is None:
                if self.instrument is None:
                    raise ValueError("Cannot have an instrument with no name! ")
                else:
                    raise ValueError(f"Could not find instrument with name: {self.instrument}! ")
            self._data = CCD_Data(self.filename, inst)
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, CCD_Data):
            raise ValueError(f"data must be a CCD_Data object. Got {type(value)} instead. ")
        self._data = value

    def get_instrument_object(self):
        if self.instrument_object is not None:
            if self.instrument is None:
                self.instrument_object = None

            else:
                self.instrument_object = getattr(models.instruments, self.instrument)()

        return self.instrument_object

    def calculate_coordinates(self):
        if self.ra is None or self.dec is None:
            raise ValueError("Exposure must have RA and Dec set before calculating coordinates! ")

        coords = SkyCoord(self.ra, self.dec, unit="deg", frame="icrs")
        self.gallat = coords.galactic.b.deg
        self.gallon = coords.galactic.l.deg
        self.ecllat = coords.barycentrictrueecliptic.lat.deg
        self.ecllon = coords.barycentrictrueecliptic.lon.deg

    def read_header(self):
        inst = self.get_instrument_object()

        # read header info, put it in the header JOSNB column
        if inst is not None:
            self.header = inst.read_header(self.filename)

    def parse_header_keys(self):
        """
        Parse the relevant columns: mjd, project, target,
        num_ccds, width, height, exp_time, filter, telescope
        from self.header and into the column attributes.
        """

        for k, v in self.header:
            norm_k = normalize_header_key(k)
            if norm_k.upper() in ['MJD-OBS', 'MJD']:
                self.mjd = v
            elif norm_k.upper() in ['PROPOSID', 'PROPOSAL', 'PROJECT']:
                self.project = v
            elif norm_k.upper() in ['OBJECT', 'TARGET', 'FIELD', 'FIELDID']:
                self.target = v
            elif norm_k.upper() in ['EXPTIME', 'EXPOSURE']:
                self.exp_time = v
            elif norm_k.upper() in ['FILTER', 'FILT']:
                self.filter = v
            elif norm_k.upper() in ['TELESCOP', 'TELESCOPE']:
                self.telescope = v
            elif norm_k.upper() in ['INSTRUME', 'INSTRUMENT']:
                self.instrument = v
            elif norm_k.upper() in ['NAXIS1']:
                self.width = v
            elif norm_k.upper() in ['NAXIS2']:
                self.height = v
            # TODO: how to parse the number of CCDs?


if __name__ == '__main__':
    e = Exposure("Demo_test.fits")
    print(e)
    # print(e.data[0])
