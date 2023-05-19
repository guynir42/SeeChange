import re
from collections import defaultdict

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.schema import CheckConstraint
from sqlalchemy.orm.session import object_session

from astropy.coordinates import SkyCoord

from models.base import Base, SeeChangeBase, FileOnDiskMixin, SpatiallyIndexed, SmartSession
from models.instrument import Instrument, guess_instrument, get_instrument_instance


# columns key names that must be loaded from the header for each Exposure
EXPOSURE_COLUMN_NAMES = [
    'ra',
    'dec',
    'mjd',
    'project',
    'target',
    'exp_time',
    'filter',
    'filter_array',
    'telescope',
    'instrument'
]

# these are header keywords that are not stored as columns of the Exposure table,
# but are still useful to keep around inside the "header" JSONB column.
EXPOSURE_HEADER_KEYS = []  # TODO: add more here


class SectionData:
    """
    A helper class that lazy loads the section data from the database.
    When requesting one of the section IDs it will fetch that data from
    disk and store it in memory.
    To clear the memory cache, call the clear_cache() method.
    """
    def __init__(self, filename, instrument):
        """
        Must initialize this object with a filename
        (or list of filenames) and an instrument object.
        These two things will control how data is loaded
        from the disk.

        Parameters
        ----------
        filename: str or list of str
            The filename of the exposure to load.
            If each section is in a different file, then
            this should be a list of filenames.
        instrument: Instrument
            The instrument object that describes the
            sections and how to load them from disk.

        """
        self.filename = filename
        self.instrument = instrument
        self._data = defaultdict(lambda: None)

    def __getitem__(self, section_id):
        if self._data[section_id] is None:
            self._data[section_id] = self.instrument.load_section_image(self.filename, section_id)
        return self._data[section_id]

    def __setitem__(self, section_id, value):
        self._data[section_id] = value

    def clear_cache(self):
        self._data = defaultdict(lambda: None)


class Exposure(Base, FileOnDiskMixin, SpatiallyIndexed):

    __tablename__ = "exposures"

    header = sa.Column(
        JSONB,
        nullable=False,
        default={},
        doc=(
            "Header of the raw exposure. "
            "Only keep a subset of the keywords, "
            "and re-key them to be more consistent. "
            "This will only include global values, "
            "not those associated with a specific section. "
        )
    )

    mjd = sa.Column(
        sa.Double,
        nullable=False,
        index=True,
        doc="Modified Julian date of the exposure (MJD=JD-2400000.5)."
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

    instrument = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Name of the instrument used to take the exposure. '
    )

    section_id = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Section ID of the exposure. '
    )

    telescope = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Telescope used to take the exposure. '
    )

    project = sa.Column(
        sa.Text,
        index=True,
        nullable=False,
        doc='Name of the project, (could also be a proposal ID). '
    )

    target = sa.Column(
        sa.Text,
        index=True,
        nullable=False,
        doc='Name of the target object or field id. '
    )

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
        FileOnDiskMixin.__init__(self, *args, **kwargs)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values
        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self._data = None

        if self.filename is None and not self.nofile:
            raise ValueError("Must give a filename to initialize an Exposure object. ")

        if self.instrument is None:
            self.instrument = guess_instrument(self.filename)

        self._instrument_object = None

        if self.instrument_object is not None:
            if self.telescope is None:
                self.telescope = self.instrument_object.telescope

            # get the header from the file in its raw form
            raw_header_dictionary = self.instrument_object.read_header(self.get_fullpath())

            critical_info = self.instrument_object.extract_critical_header_info(raw_header_dictionary, EXPOSURE_COLUMN_NAMES)
            for k, v in critical_info.items():
                if k == 'instrument':
                    if self.instrument != v:
                        raise ValueError(f"Header instrument {v} does not match Exposure instrument {self.instrument}")
                elif k == 'telescope':
                    if self.telescope != v:
                        raise ValueError(
                            f"Header telescope {v} does not match Exposure telescope {self.instrument.telescope}"
                        )
                else:
                    setattr(self, k, v)

            auxiliary_names = EXPOSURE_HEADER_KEYS + self.instrument_object.get_auxiliary_exposure_header_keys()
            self.header = self.instrument_object.extract_auxiliary_header_info(raw_header_dictionary, auxiliary_names)

        if self.ra is not None and self.dec is not None:
            self.calculate_coordinates()

    @property
    def instrument_object(self):
        if self.instrument is not None:
            if self._instrument_object is None or self._instrument_object.name != self.instrument:
                self._instrument_object = get_instrument_instance(self.instrument)

        return self._instrument_object

    @instrument_object.setter
    def instrument_object(self, value):
        self._instrument_object = value

    @sa.orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        self._data = None
        self._instrument_object = None
        session = object_session(self)
        if session is not None:
            self.instrument_object.fetch_sections(session=session)

    def __repr__(self):

        filter_str = '--'
        if self.filter is not None:
            filter_str = self.filter
        if self.filter_array is not None:
            filter_str = f"[{', '.join(self.filter_array)}]"

        return (
            f"Exposure(id: {self.id}, "
            f"exp: {self.exp_time}s, "
            f"filt: {filter_str}, "
            f"from: {self.instrument}/{self.telescope})"
        )

    def __str__(self):
        return self.__repr__()

    def save(self):
        pass  # TODO: implement this! do we need this?

    def load(self, section_ids=None):
        if section_ids is None:
            section_ids = self.instrument.get_section_ids()

        if not isinstance(section_ids, list):
            section_ids = [section_ids]

        if not all([isinstance(sec_id, (str, int)) for sec_id in section_ids]):
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
            self._data = SectionData(self.get_fullpath(), self.instrument_object)
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


if __name__ == '__main__':

    from models.base import Session
    import models.instrument

    import numpy as np
    rnd_str = lambda n: ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n))

    e = Exposure(f"Demo_test_{rnd_str(5)}.fits", exp_time=30, mjd=58392.0, filter="F160W", ra=123, dec=-23, project='foo', target='bar')

    session = Session()

    session.add(e)
    session.commit()
