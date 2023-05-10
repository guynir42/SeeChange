import re
import inspect
from collections import defaultdict

import numpy as np

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.schema import CheckConstraint
from sqlalchemy.ext.associationproxy import association_proxy

from astropy.coordinates import SkyCoord

from models.base import Base, SpatiallyIndexed, SmartSession
from models.instrument import Instrument, INSTRUMENT_FILENAME_REGEX

from pipeline.utils import normalize_header_key


class SectionData:
    """
    A helper class that lazy loads the section data from the database.
    When requesting one of the section IDs it will fetch that data from
    disk and store it in memory.
    To clear the memory cache, call the clear_cache() method.
    """
    def __init__(self, filename, inst):
        self.filename = filename
        self.instrument = inst
        self._data = defaultdict(lambda: None)

    def __getitem__(self, section_id):
        if self._data[section_id] is None:
            self._data[section_id] = self.instrument.load_section(self.filename, section_id)
        return self._data[section_id]

    def __setitem__(self, section_id, value):
        self._data[section_id] = value

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

    filter = sa.Column(sa.Text, nullable=True, index=True, doc="Filter name")

    filter_array = sa.Column(sa.ARRAY(sa.Text), nullable=True, index=True, )

    __table_args__ = (
        CheckConstraint(
            sqltext='NOT(filter IS NULL AND filter_array IS NULL)',
            name='exposures_filter_or_array_check'
        ),
    )

    instrument_id = sa.Column(sa.Integer, sa.ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False)

    instrument = sa.orm.relationship(
        Instrument,
        back_populates='exposures',
        doc='Instrument used to take the exposure'
    )

    instrument_name = association_proxy('instrument', 'name')

    telescope_name = association_proxy('instrument', 'telescope')

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
        super(Base, self).__init__(**kwargs)  # put keywords into columns

        self._data = None

        if len(args) == 1 and isinstance(args[0], str):
            self.filename = args[0]

        if self.filename is None:
            # TODO: is there any use case where we dump the data into an exposure and then make a file?
            raise ValueError("Must give a filename to initialize an Exposure object. ")

        if self.instrument_id is None:
            self.guess_instrument()

        self.read_header()

        if self.ra is not None and self.dec is not None:
            self.calculate_coordinates()

    @sa.orm.reconstructor
    def init_on_load(self):
        super(Base, self).init_on_load()
        self._data = None

    def guess_instrument(self):
        if self.filename is None:
            raise ValueError("Cannot guess instrument without a filename! ")

        instrument_list = []
        for k, v in INSTRUMENT_FILENAME_REGEX.items():
            if re.search(k, self.filename):
                instrument_list.append(v)

        if len(instrument_list) == 0:
            # TODO: maybe add a fallback of looking into the file header?
            # raise ValueError(f"Could not guess instrument from filename: {self.filename}. ")
            pass  # leave empty is the right thing? should probably go to a fallback method
        elif len(instrument_list) == 1:
            self.instrument_id = instrument_list[0]
        else:
            raise ValueError(f"Found multiple instruments in filename: {self.filename}. ")

        # TODO: add fallback method that runs all instruments
        #  (or only those on the short list) and checks if they can load the file

    def __repr__(self):
        return (
            f"Exposure(id: {self.id}, "
            f"exp: {self.exp_time}s, "
            f"filt: {self.filter}, "
            f"from: {self.instrument_name}/{self.telescope_name})"
        )

    def __str__(self):
        return self.__repr__()

    def fetch_instrument(self, session=None):
        """
        Fetch the instrument object from the database.

        Parameters
        ----------
        session: sqlalchemy.orm.session.Session
            Session to use to query the database.
            If not given (or None), will open a session
            and close it at the end of the call.

        Returns
        -------
        Instrument
            Instrument object from the database.
        """
        with SmartSession(session) as session:
            self.instrument = session.scalars(sa.select(Instrument).where(Instrument.id == self.instrument_id)).first()
            return self.instrument  # careful, this instrument is attached to session that could close right away

    def save(self):
        pass  # TODO: implement this! do we need this?

    def load(self, section_ids=None):
        if section_ids is None:
            section_ids = self.instrument.get_section_ids()

        if not isinstance(section_ids, list):
            section_ids = [section_ids]

        if not all([isinstance(sec_id, int) for sec_id in section_ids]):
            raise ValueError("section_ids must be a list of integers. ")

        if self.filename is not None:
            for i in section_ids:
                self.data[i]  # use the SectionData __getitem__ method to load the data
        else:
            raise ValueError("Cannot load data from database without a filename! ")

    @property
    def data(self):
        if self._data is None:
            if self.instrument is None:
                raise ValueError("Cannot load data without an instrument! ")
            self._data = SectionData(self.filename, self.instrument)
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, SectionData):
            raise ValueError(f"data must be a SectionData object. Got {type(value)} instead. ")
        self._data = value

    def calculate_coordinates(self):
        if self.ra is None or self.dec is None:
            raise ValueError("Exposure must have RA and Dec set before calculating coordinates! ")

        coords = SkyCoord(self.ra, self.dec, unit="deg", frame="icrs")
        self.gallat = coords.galactic.b.deg
        self.gallon = coords.galactic.l.deg
        self.ecllat = coords.barycentrictrueecliptic.lat.deg
        self.ecllon = coords.barycentrictrueecliptic.lon.deg

    def read_header(self):
        # read header info, put it in the header JOSNB column
        if self.instrument is not None:
            self.header = self.instrument.read_header(self.filename)

    def parse_header_keys(self):
        """
        Parse the relevant columns: mjd, project, target,
        num_ccds, width, height, exp_time, filter, telescope
        from self.header and into the column attributes.
        """

        for k, v in self.header:
            norm_k = normalize_header_key(k)
            if norm_k in ['MJD-OBS', 'MJD']:
                self.mjd = v
            elif norm_k in ['PROPOSID', 'PROPOSAL', 'PROJECT']:
                self.project = v
            elif norm_k in ['OBJECT', 'TARGET', 'FIELD', 'FIELDID']:
                self.target = v
            elif norm_k in ['EXPTIME', 'EXPOSURE']:
                self.exp_time = v
            elif norm_k in ['FILTER', 'FILT']:
                self.filter = v
            elif norm_k in ['TELESCOP', 'TELESCOPE']:
                self.telescope = v
            elif norm_k in ['INSTRUME', 'INSTRUMENT']:
                self.instrument = v


if __name__ == '__main__':

    from models.base import Session
    import models.instrument

    import numpy as np
    rnd_str = lambda n: ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n))

    e = Exposure(f"Demo_test_{rnd_str(5)}.fits", exp_time=30, mjd=58392.0, filter="F160W", ra=123, dec=-23, project='foo', target='bar')

    session = Session()

    inst = session.scalars(sa.select(Instrument)).all()

    e.guess_instrument()

    session.add(e)
    session.commit()
