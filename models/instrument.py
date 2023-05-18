# TODO: make a base Instrument class that has all the common methods.
#  Subclass it for each instrument we want to use, e.g., DECam, LS4, etc.
#  Each one of these classes must implement load and save methods.
#  There are other things we may need from an instrument, like aperture, pixel scale, etc.

# TODO: what about CCD chips with different filters? we sort of assume all sections have the same filter
#  in each exposure, but for e.g., LS4 this will not be the case.
#  Maybe we need to add an optional filter_array column to the Exposure class,
#  where each exposure will have a list of filter names corresponding to the sections in that one frame.
#  Then when making Image objects from an Exposure object we would check if the filter_array,
#  and fall back to the regular filter column if filter_array is None.

import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

import sqlalchemy as sa
from sqlalchemy.schema import UniqueConstraint

from models.base import Base, SmartSession

from pipeline.utils import parse_dateobs

INSTRUMENT_FILENAME_REGEX = {}

HEADER_KEYS_WITH_DEFAULT = ['mjd', 'exp_time', 'filter', ]
HEADER_KEYS_WITHOUT_DEFAULT = []


class SensorSection(Base):
    """
    A class to represent a section of a sensor.
    This is most often associated with a CCD chip, but could be any
    section of a sensor. For example, a section of a CCD chip that
    is read out independently, or different channels in a dichroic imager.

    Any properties that are not set (e.g., set to None) on the sensor section
    will be replaced by the global value of the parent Instrument object.
    E.g., if the DemoInstrument has gain=2.0, and it's sensor section has
    gain=None, then the sensor section will have gain=2.0.
    If at any time the instrument changes, add a new SensorSection object
    (with appropriate validity range) to the database to capture the new
    instrument properties.
    Thus, a SensorSection can override global instrument values either for
    specific parts of the sensor (spatial variability) or for specific times
    (temporal variability).
    """

    __tablename__ = "sensor_sections"

    instrument = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='The name of the instrument this section belongs to. '
    )

    identifier = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='A unique identifier for this section. Can be, e.g., the CCD ID. '
    )

    validity_start = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc=(
            'The time when this section object becomes valid. '
            'If None, this section is valid from the beginning of time. '
            'Use the validity range to get updated versions of sections, '
            'e.g., after a change of CCD. '
        )
    )

    validity_end = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc=(
            'The time when this section object becomes invalid. '
            'If None, this section is valid until the end of time. '
            'Use the validity range to get updated versions of sections, '
            'e.g., after a change of CCD. '
        )
    )

    size_x = sa.Column(
        sa.Integer,
        nullable=True,
        doc='Number of pixels in the x direction. '
    )

    size_y = sa.Column(
        sa.Integer,
        nullable=True,
        doc='Number of pixels in the y direction. '
    )

    offset_x = sa.Column(
        sa.Integer,
        nullable=True,
        doc='Offset of the section in the x direction (in pixels). '
    )

    offset_y = sa.Column(
        sa.Integer,
        nullable=True,
        doc='Offset of the section in the y direction (in pixels). '
    )

    filter_array_index = sa.Column(
        sa.Integer,
        nullable=True,
        doc='Index in the filter array that specifies which filter this section is located under in the array. '
    )

    read_noise = sa.Column(
        sa.Float,
        nullable=True,
        doc='Read noise of the sensor section (in electrons). '
    )

    dark_current = sa.Column(
        sa.Float,
        nullable=True,
        doc='Dark current of the sensor section (in electrons/pixel/second). '
    )

    gain = sa.Column(
        sa.Float,
        nullable=True,
        doc='Gain of the sensor section (in electrons/ADU). '
    )

    saturation_limit = sa.Column(
        sa.Float,
        nullable=True,
        doc='Saturation level of the sensor section (in electrons). '
    )

    non_linearity_limit = sa.Column(
        sa.Float,
        nullable=True,
        doc='Non-linearity of the sensor section (in electrons). '
    )

    defective = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        index=True,
        doc='Whether this section is defective (i.e., if True, do not use it!). '
    )

    def __init__(self, identifier, **kwargs):
        """
        Create a new SensorSection object.
        Some parameters must be filled out for this object.
        Others (e.g., offsets) can be left at the default value.

        Parameters
        ----------
        identifier: str or int
            A unique identifier for this section. Can be, e.g., the CCD ID.
            Integers will be converted to strings.
        kwargs: dict
            Additional values like gain, saturation_limit, etc.
        """
        if not isinstance(identifier, (str, int)):
            raise ValueError(f"identifier must be a string or an integer. Got {type(identifier)}.")

        self.identifier = str(identifier)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"<SensorSection {self.identifier} ({self.size_x}x{self.size_y})>"

    def __eq__(self, other):
        """
        Check if the sensor section is identical to the other one.
        Returns True if all attributes are the same,
        not including database level attributes like id, created_at, etc.
        """

        for att in self.get_attribute_list():
            if getattr(self, att) != getattr(other, att):
                return False

        return True


class Instrument(Base):
    """
    Base class for an instrument.
    Instruments contain all the information about the instrument and telescope,
    that were used to produce an exposure.

    Subclass this base class to add methods that are unique to each instrument,
    e.g., loading files, reading headers, etc.

    Each instrument can have one or more SensorSection objects,
    each corresponding to a part of the focal plane (e.g., a CCD chip).
    These include additional info like the chip offset, size, etc.
    The sections can be generated dynamically (using hard-coded values),
    or loaded from the database using get_sections().

    If a sensor section has a non-null value for a given parameter
    (e.g., gain) then that value is used instead of the Instrument
    object's global value. Thus, a sensor section can be used to
    override the global parameter values.
    Sections can also be defined with a validity range,
    to reflect changes in the instrument (e.g., replacement of a CCD).
    This
    """
    def __init__(self, **kwargs):
        self.name = None  # name of the instrument (e.g., DECam)

        # telescope related properties
        self.telescope = None  # name of the telescope it is mounted on (e.g., Blanco)
        self.focal_ratio = None  # focal ratio of the telescope (e.g., 2.7)
        self.aperture = None  # telescope aperture in meters (e.g., 4.0)
        self.pixel_scale = None  # number of arc-seconds per pixel (e.g., 0.2637)

        # sensor related properties
        # these are average value for all sensor sections,
        # and if no sections can be loaded, or if the sections
        # do not define these properties, then the global values are used
        self.size_x = 512  # number of pixels in the x direction
        self.size_y = 1024  # number of pixels in the y direction
        self.read_time = None  # read time in seconds (e.g., 20.0)
        self.read_noise = None  # read noise in electrons (e.g., 7.0)
        self.dark_current = None  # dark current in electrons/pixel/second (e.g., 0.2)
        self.gain = None  # gain in electrons/ADU (e.g., 4.0)
        self.saturation_limit = None  # saturation limit in electrons (e.g., 100000)
        self.non_linearity_limit = None  # non-linearity limit in electrons (e.g., 100000)

        self.allowed_filters = None  # list of allowed filter names (e.g., ['g', 'r', 'i', 'z', 'Y'])

        # self.sections = self._make_sections()  # list of SensorSection objects (ordered by identifier)
        self.sections = None  # can populate this using fetch_sections(), then it would be a dict
        self._dateobs_for_sections = None  # what the dateobs was when loading sections
        self._dateobs_range_days = 1.0  # how many days away from dateobs needs to reload sections

        super(Base, self).__init__(**kwargs)

    def _make_new_section(self, identifier):
        raise NotImplementedError("Subclass this base class to add methods that are unique to each instrument.")

    def __repr__(self):
        return (
            f'<Instrument {self.name} on {self.telescope} ' 
            f'({self.aperture:.1f}m, {self.pixel_scale:.2f}"/pix, '
            f'[{",".join(self.allowed_filters)}])>'
        )

    def load(self, filename, section_ids=None):
        """
        Load a part of an exposure file, based on the section identifier.
        If the instrument does not have multiple sections, set section_ids=0.

        Parameters
        ----------
        filename: str
            The filename of the exposure file.
        section_ids: str, int, or list of str or int (optional)
            Choose which section to load.
            The section_id is the identifier of the SensorSection object.
            This can be a serial number which is converted to a string.
            If given as a list, will load all the sections mentioned in the list,
            and returns a list of data arrays.
            If None (or not given) will load all the sections in the instrument,
            and return a list of arrays.

        Returns
        -------
        data: np.ndarray or list of np.ndarray
            The data from the exposure file.
        """
        if section_ids is None:
            section_ids = self.get_section_ids()

        if isinstance(section_ids, (int, str)):
            return self.load_section(filename, section_ids)

        elif isinstance(section_ids, list):
            return [self.load_section(filename, section_id) for section_id in section_ids]

        else:
            raise ValueError(
                f"section_ids must be a string, int, or list of strings or ints. Got {type(section_ids)}"
            )

    def load_section_image(self, filename, section_id):
        """
        Load one section of an exposure file.
        This must be implemented by each subclass.

        Parameters
        ----------
        filename: str
            The filename of the exposure file.
        section_id: str or int
            The identifier of the SensorSection object.
            This can be a serial number which is converted to a string.

        Returns
        -------
        data: np.ndarray
            The data from the exposure file.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    def get_section_ids(self):
        """
        Get a list of SensorSection identifiers for this instrument.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    def check_section_id(self, section_id):
        """
        Check that the type and value of the section is compatible with the instrument.
        For example, many instruments will key the section by a running integer (e.g., CCD ID),
        while others will use a string (e.g., channel 'A').

        Will raise a meaningful error if not compatible.

        Subclasses should override this method to be more specific
        (e.g., to test if an integer is in range).
        """
        if not isinstance(section_id, (int, str)):
            raise ValueError(f"The section_id must be an integer or string. Got {type(section_id)}. ")

    def get_section(self, section_id):
        """Get a section from the sections dictionary. """
        self.check_section_id(section_id)

        if self.sections is None:
            raise RuntimeError("No sections loaded for this instrument. Use fetch_sections() first.")

        return self.sections[section_id]

    def fetch_sections(self, session=None, dateobs=None):
        """
        Get the sensor section objects associated with this instrument.

        Will try to get sections that are valid during the given date.
        If any sections are missing, they will be created using the
        hard coded values in _make_new_section().
        If multiple valid sections are found, use the latest one
        (the one with the most recent "modified" value).

        Will populate the self.sections attribute,
        and will lazy load that before checking against the DB.
        If the dateobs value is too far from that used the last time
        the sections were populated, then they will be cleared and reloaded.
        The time delta for this is set by self._dateobs_range_days (=1 by default).

        Parameters
        ----------
        session: sqlalchemy.orm.Session (optional)
            The database session to use. If None, will create a new session.
            Use session=False to avoid using the database entirely.
        dateobs: datetime or Time or float (as MJD) or string (optional)
            The date of the observation. If None, will use the current date.
            If there are multiple instances of a sensor section on the DB,
            only choose the ones valid during the observation.

        Returns
        -------
        sections: list of SensorSection
            The sensor sections associated with this instrument.
        """
        dateobs = parse_dateobs(dateobs, output='datetime')

        # if dateobs is too far from last time we loaded sections, reload
        if self._dateobs_for_sections is not None:
            if abs(self._dateobs_for_sections - dateobs) < timedelta(self._dateobs_range_days):
                self.sections = None

        # this should never happen, but still
        if self._dateobs_for_sections is None:
            self.sections = None

        # we are allowed to use the DB
        if self.sections is None:
            self.sections = {}
            self._dateobs_for_sections = dateobs  # track the date used to load
            if session is False:
                all_sec = []
            else:
                # load sections from DB
                with SmartSession(session) as session:
                    all_sec = session.scalars(
                        sa.select(SensorSection).where(
                            SensorSection.instrument == self.name,
                            SensorSection.validity_start <= dateobs,
                            SensorSection.validity_end >= dateobs,
                        ).order_by(SensorSection.modified.desc())
                    ).all()

            for sid in self.get_section_ids():
                sec = [s for s in all_sec if s.identifier == str(sid)]
                if len(sec) > 0:
                    self.sections[sid] = sec[0]
                else:
                    self.sections[sid] = self._make_new_section(sid)

        return self.sections

    def commit_sections(self, session=None, validity_start=None, validity_end=None):
        """
        Commit the sensor sections associated with this instrument to the database.
        This is used to update or add missing sections that were created from
        hard-coded values (i.e., using the _make_new_section() method).

        Parameters
        ----------
        session: sqlalchemy.orm.Session (optional)
            The database session to use. If None, will create a new session.
        validity_start: datetime or Time or float (as MJD) or string (optional)
            The start of the validity range for these sections.
            Only changes the validity start of sections that have validity_start=None.
            If None, will not modify any of the validity start values.
        validity_end: datetime or Time or float (as MJD) or string (optional)
            The end of the validity range for these sections.
            Only changes the validity end of sections that have validity_end=None.
            If None, will not modify any of the validity end values.
        """
        with SmartSession(session) as session:
            for sec in self.sections.values():
                if sec.validity_start is None and validity_start is not None:
                    sec.validity_start = validity_start
                if sec.validity_end is None and validity_end is not None:
                    sec.validity_end = validity_end
                session.add(sec)

            session.commit()

    def get_property(self, section_id, prop):
        """
        Get the value of a property for a given section of the instrument.
        If that property is not defined on the sensor section
        (e.g., if it is None) then the global value from the Instrument is used.

        Will raise an error if no sections were loaded (if sections=None).
        If sections were loaded but no section with the required id is found,
        will quietly use the global value.

        """

        if self.sections is None:
            raise ValueError('Must use fetch_sections() before calling this function. ')

        if section_id in self.sections:
            section = self.sections[section_id]
            if hasattr(section, prop) and getattr(section, prop) is not None:
                return getattr(section, prop)

        # first check if we can recover these properties from hard-coded functions:
        if prop == 'offset_x':
            return self.get_section_offsets(section_id)[0]
        elif prop == 'offset_y':
            return self.get_section_offsets(section_id)[1]
        elif prop == 'filter_array_index':
            return self.get_section_filter_array_index(section_id)
        else:  # just get the value from the object
            return getattr(self, prop)

    def get_section_offsets(self, section_id):
        """
        Get the offset of the given section from the origin of the detector.
        This can be used if the SensorSection object itself does not have
        values for offset_x and offset_y. Use this function in subclasses
        to hard-code the offsets.
        If the offsets need to be updated over time, they should be
        added to the SensorSection objects on the database.

        Parameters
        ----------
        section_id: int or str
            The identifier of the section.

        Returns
        -------
        offset: tuple
            The offsets in the x and y direction.
        """
        # this simple instrument defaults to zero offsets for ALL sections
        offset_x = 0
        offset_y = 0
        return offset_x, offset_y

    def get_section_filter_array_index(self, section_id):
        """
        Get the index in the filter array under which this section is placed.
        This can be used if the SensorSection object itself does not have
        a value for filter_array_index. Use this function in subclasses
        to hard-code the array index.
        If the array index need to be updated over time, it should be
        added to the SensorSection objects on the database.

        Parameters
        ----------
        section_id: int or str
            The identifier of the section.

        Returns
        -------
        idx: int
            The index in the filter array.
        """
        # this simple instrument has no filter array, so return zero
        idx = 0
        return idx

    def get_filename_regex(self):
        """
        Get the regular expression used to match filenames for this instrument.

        Returns
        -------
        regex: str
            The regular expression string.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    # TODO: should we read headers independently for each section?
    def read_header(self, filename):
        """
        Load the FITS header from filename.

        Parameters
        ----------
        filename: str
            The filename of the exposure file.

        Returns
        -------
        header: dict
            The header from the exposure file, as a dictionary.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    def normalize_header_key(key):
        """
        Normalize the header key to be all uppercase and
        remove spaces and underscores.
        """
        return key.upper().replace(' ', '').replace('_', '')

    def get_keys_without_default(self):
        """
        Get the keys that must be present in the header,
        and do not have a default value.

        Subclasses can override this method to add or remove keys.
        As the list returned is a copy of the global parameter,
        the subclass can modify the list without affecting the global value.
        """
        return list(HEADER_KEYS_WITHOUT_DEFAULT)

    def get_keys_with_default(self):
        """
        Get the keys that don't have to be present in the header,
        and do have a default value from the instrument class.

        Subclasses can override this method to add or remove keys.
        As the list returned is a copy of the global parameter,
        the subclass can modify the list without affecting the global value.
        """
        return list(HEADER_KEYS_WITH_DEFAULT)

    def parse_header_keys(self, header):
        """
        Parse the relevant columns: mjd, project, target,
        width, height, exp_time, filter, telescope, etc
        from self.header and into the column attributes.

        NOTE: this default method will parse the "normal"
        header keys, but subclasses for new instruments can
        augment or replace this method to parse other keys
        or keys that have different meaning for that instrument.

        Parameters
        ----------
        header: dict
            The header from the exposure file, as a dictionary.

        Returns
        -------
        header_values: dict
            The parsed header values.
            Some values will use the instrument's default,
            while others (like mjd or filter) will have None
            as the default (which could raise exceptions later
            on if they are missing from the header).
        """

        header_values = {}
        for k in self.get_keys_without_default():
            header_values[k] = None

        # use the instrument defaults for some values
        for k in self.get_keys_with_default():
            header_values[k] = getattr(self, k)

        # check if any values exist in the header.
        for k, v in header:
            norm_k = self.normalize_header_key(k)
            if norm_k in ['MJD-OBS', 'MJD']:
                header_values['mjd'] = v
            elif norm_k in ['PROPOSID', 'PROPOSAL', 'PROJECT']:
                header_values['project'] = v
            elif norm_k in ['OBJECT', 'TARGET', 'FIELD', 'FIELDID']:
                header_values['target'] = v
            elif norm_k in ['EXPTIME', 'EXPOSURE']:
                header_values['exp_time'] = v
            elif norm_k in ['FILTER', 'FILT']:
                header_values['filter'] = v
            elif norm_k in ['TELESCOP', 'TELESCOPE']:
                header_values['telescope'] = v
            elif norm_k in ['INSTRUME', 'INSTRUMENT']:
                header_values['instrument'] = v

        return header_values


class DemoInstrument(Instrument):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'DemoInstrument'
        self.telescope = 'DemoTelescope'
        self.aperture = 2.0
        self.focal_ratio = 5.0
        self.square_degree_fov = 0.5
        self.pixel_scale = 0.41
        self.read_time = 2.0
        self.read_noise = 1.5
        self.dark_current = 0.1
        self.gain = 2.0
        self.non_linearity_limit = 10000.0
        self.saturation_limit = 50000.0
        self.allowed_filters = ["g", "r", "i", "z", "Y"]

        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_section_ids(self):
        """
        Get a list of SensorSection identifiers for this instrument.
        """
        return [0]

    def check_section_id(self, section_id):
        """
        Check if the section_id is valid for this instrument.
        The demo instrument only has one section, so the section_id must be 0.
        """
        if not isinstance(section_id, int):
            raise ValueError(f"section_id must be an integer. Got {type(section_id)} instead.")
        if section_id != 0:
            raise ValueError(f"section_id must be 0 for this instrument. Got {section_id} instead.")

    def _make_new_section(self, identifier):
        """
        Make a single section for the DEMO instrument.
        The identifier must be a valid section identifier.

        Returns
        -------
        section: SensorSection
            A new section for this instrument.
        """
        return SensorSection(identifier, size_x=512, size_y=1024)

    def load_section_image(self, filename, section_id):
        """
        A spoof load method for this demo instrument.
        The data is just a random array.
        The instrument only has one section,
        so the section_id must be 0.

        Will fail if sections were not loaded using fetch_sections().

        Parameters
        ----------
        filename: str
            The filename of the exposure file.
            In this case the filename is not used.
        section_id: str or int
            The identifier of the SensorSection object.
            This instrument only has one section, so this must be 0.

        Returns
        -------
        data: np.ndarray
            The data from the exposure file.
        """

        section = self.get_section(section_id)

        return np.random.poisson(10, (section.size_y, section.size_x))

    def read_header(self, filename):
        return {}

    def get_filename_regex(self):
        return [r'Demo']


class DECam(Instrument):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'DECam'
        self.telescope = 'Blanco'
        self.aperture = 4.0
        self.focal_ratio = 2.7
        self.square_degree_fov = 3.0
        self.pixel_scale = 0.2637
        self.read_time = 20.0
        self.read_noise = 7.0
        self.dark_current = 0.1
        self.gain = 4.0
        self.saturation_limit = 100000
        self.non_linearity_limit = 200000
        self.allowed_filters = ["g", "r", "i", "z", "Y"]

        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_section_ids(self):
        """
        Get a list of SensorSection identifiers for this instrument.
        """
        return range(64)

    def check_section_id(self, section_id):
        """
        Check that the type and value of the section is compatible with the instrument.
        In this case, it must be an integer in the range [0, 63].
        """
        if not isinstance(section_id, int):
            raise ValueError(f"The section_id must be an integer. Got {type(section_id)}. ")

        if not 0 <= section_id <= 63:
            raise ValueError(f"The section_id must be in the range [0, 63]. Got {section_id}. ")

    def get_section_offsets(self, section_id):
        """
        Find the offset for a specific section.

        Parameters
        ----------
        section_id: int
            The identifier of the section.

        Returns
        -------
        offset_x: int
            The x offset of the section.
        offset_y: int
            The y offset of the section.
        """
        # TODO: this is just a placeholder, we need to put the actual offsets here!
        dx = section_id % 8
        dy = section_id // 8
        return dx * 2048, dy * 4096

    def _make_new_section(self, section_id):
        """
        Make a single section for the DECam instrument.
        The section_id must be a valid section identifier (int in this case).

        Returns
        -------
        section: SensorSection
            A new section for this instrument.
        """
        # TODO: we must improve this!
        #  E.g., we should add some info on the gain and read noise (etc) for each chip.
        #  Also need to fix the offsets, this is really not correct.
        (dx, dy) = self.get_section_offsets(section_id)
        return SensorSection(section_id, size_x=2048, size_y=4096, offset_x=dx, offset_y=dy)

    def get_filename_regex(self):
        return [r'c4d.*ori\.fits']


if __name__ == "__main__":
    inst = DemoInstrument()
