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
import sqlalchemy as sa
from sqlalchemy.schema import UniqueConstraint

from models.base import Base, SmartSession


INSTRUMENT_FILENAME_REGEX = {}


class SensorSection(Base):
    """
    A class to represent a section of a sensor.
    This is most often associated with a CCD chip, but could be any
    section of a sensor. For example, a section of a CCD chip that
    is read out independently, or different channels in a dichroic
    imager.
    """

    __tablename__ = "sensor_sections"

    identifier = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='A unique identifier for this section. Can be, e.g., the CCD ID. '
    )

    size_x = sa.Column(
        sa.Integer,
        nullable=False,
        doc='Number of pixels in the x direction. '
    )

    size_y = sa.Column(
        sa.Integer,
        nullable=False,
        doc='Number of pixels in the y direction. '
    )

    offset_x = sa.Column(
        sa.Integer,
        nullable=False,
        default=0,
        doc='Offset of the section in the x direction (in pixels). '
    )

    offset_y = sa.Column(
        sa.Integer,
        nullable=False,
        default=0,
        doc='Offset of the section in the y direction (in pixels). '
    )

    instrument_id = sa.Column(
        sa.Integer,
        sa.ForeignKey('instruments.id'),
        nullable=False,
        doc='The ID of the instrument this section belongs to. '
    )

    instrument = sa.orm.relationship(
        'Instrument',
        back_populates='sections',
        doc='The instrument this section belongs to. '
    )

    def __init__(self, identifier, size_x, size_y, offset_x=0, offset_y=0):
        """
        Create a new SensorSection object.
        Some parameters must be filled out for this object.
        Others (e.g., offsets) can be left at the default value.

        Parameters
        ----------
        identifier: str or int
            A unique identifier for this section. Can be, e.g., the CCD ID.
            Integers will be converted to strings.
        size_x: int
            Number of pixels in the x direction.
        size_y: int
            Number of pixels in the y direction.
        offset_x: int (optional)
            Offset of the section in the x direction (in pixels).
            Default is 0.
        offset_y: int (optional)
            Offset of the section in the y direction (in pixels).
            Default is 0.
        """
        if not isinstance(identifier, (str, int)):
            raise ValueError(f"identifier must be a string or an integer. Got {type(identifier)}.")

        self.identifier = str(identifier)
        self.size_x = size_x
        self.size_y = size_y
        self.offset_x = offset_x
        self.offset_y = offset_y

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

    def update(self, other):
        """
        Update the attributes of this object to match the other object.
        """
        for att in self.get_attribute_list():
            setattr(self, att, getattr(other, att))


class Instrument(Base):
    """
    Base class for an instrument.
    This class corresponds to rows in the database where we keep track
    of details of each instrument. For example, the pixel scale, aperture,
    etc. are all stored here. For the time being, this will also include
    the telescope information (where you'd have multiple rows of the same
    instrument for different telescopes if the instrument is mobile).

    Subclass this base class to add methods that are unique to each instrument,
    e.g., loading files, reading headers, etc.
    The subclass objects will still be stored on the same table,
    but would have other methods that are instrument-specific.

    Each instrument can have one or more SensorSection objects,
    each corresponding to a part of the focal plane (e.g., a CCD chip).
    These include additional info like the chip offset, size, etc.
    """

    __tablename__ = "instruments"

    name = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Name of the instrument. '
    )

    telescope = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Name of the telescope this instrument is attached to. '
    )

    aperture = sa.Column(
        sa.Float,
        nullable=False,
        doc='Aperture of the instrument (in meters). '
    )

    focal_ratio = sa.Column(
        sa.Float,
        nullable=False,
        doc='Focal ratio of the instrument. '
    )

    pixel_scale = sa.Column(
        sa.Float,
        nullable=False,
        doc='Pixel scale of the instrument (in arc-second/pixel). '
    )

    square_degree_fov = sa.Column(
        sa.Float,
        nullable=False,
        doc='Field of view of the instrument (in square degrees). '
    )

    read_time = sa.Column(
        sa.Float,
        nullable=False,
        doc='Read time of the instrument (in seconds). '
    )

    read_noise = sa.Column(
        sa.Float,
        nullable=False,
        doc='Read noise of the instrument (in electrons). '
    )

    dark_current = sa.Column(
        sa.Float,
        nullable=False,
        doc='Dark current of the instrument (in electrons/pixel/second). '
    )

    gain = sa.Column(
        sa.Float,
        nullable=False,
        doc='Gain of the instrument (in electrons/ADU). '
    )

    saturation_limit = sa.Column(
        sa.Float,
        nullable=False,
        doc='Saturation level of the instrument (in electrons). '
    )

    non_linearity_limit = sa.Column(
        sa.Float,
        nullable=False,
        doc='Non-linearity of the instrument (in electrons). '
    )

    allowed_filters = sa.Column(
        sa.ARRAY(sa.String),
        nullable=False,
        doc='A list of allowed string names of filters this instrument can use. '
    )

    filename_regex = sa.Column(
        sa.ARRAY(sa.Text),
        nullable=True,
        doc='A list of regular expression strings that can be used '
            'to match filenames that were taken using this instrument. '
    )

    sections = sa.orm.relationship(
        SensorSection,
        back_populates='instrument',
        cascade='all, delete-orphan',
        lazy='selectin',  # load these by default
        order_by=SensorSection.identifier,
        doc='A list of sections of the instrument. '
    )

    exposures = sa.orm.relationship(
        'Exposure',
        back_populates='instrument',
        cascade='all, delete-orphan',
        passive_deletes=True,
        doc='A list of Exposure objects made using this instrument. '
    )

    __table_args__ = (UniqueConstraint("name", "telescope", name="_instrument_name_telescope_uc"),)

    __mapper_args__ = {
        "polymorphic_on": "name",
        "polymorphic_identity": "instrument",
    }

    def __init__(self, **kwargs):
        self.name = None  # name of the instrument (e.g., DECam)
        self.telescope = None  # name of the telescope it is mounted on (e.g., Blanco)
        self.focal_ratio = None  # focal ratio of the telescope (e.g., 2.7)
        self.aperture = None  # telescope aperture in meters (e.g., 4.0)
        self.pixel_scale = None  # number of arc-seconds per pixel (e.g., 0.2637)
        self.read_time = None  # read time in seconds (e.g., 20.0)
        self.read_noise = None  # read noise in electrons (e.g., 7.0)
        self.dark_current = None  # dark current in electrons/pixel/second (e.g., 0.2)
        self.gain = None  # gain in electrons/ADU (e.g., 4.0)
        self.saturation_limit = None  # saturation limit in electrons (e.g., 100000)
        self.non_linearity_limit = None  # non-linearity limit in electrons (e.g., 100000)

        self.allowed_filters = None  # list of allowed filter names (e.g., ['g', 'r', 'i', 'z', 'Y'])

        self.sections = self._make_sections()  # list of SensorSection objects (ordered by identifier)

        super(Base, self).__init__(**kwargs)

    @staticmethod
    def _make_sections():
        raise NotImplementedError("Subclass this base class to add methods that are unique to each instrument.")

    def __repr__(self):
        return (
            f'<Instrument {self.name} on {self.telescope} ' 
            f'({self.aperture:.1f}m, {self.pixel_scale:.2f}"/pix, '
            f'[{",".join(self.allowed_filters)}])>'
        )

    def __eq__(self, other):
        """
        Compare this instrument to another instrument,
        and return true if all their attributes are the same.
        This does not include database level attributes such
        as id, created_at, etc.
        """

        # will also check the section list is the same, using SensorSection.__eq__
        for att in self.get_attribute_list():
            if getattr(self, att) != getattr(other, att):
                # print(f'Instrument attribute {att} is different: {getattr(self, att)} != {getattr(other, att)}')
                return False

        return True

    def update(self, other):
        """
        Update instance with new attribute values.
        Will also selectively update the sections list,
        by calling SensorSection.update() on any sections that
        are not the same.
        """

        for att in self.get_attribute_list():
            if att == 'sections':
                # allow a new list with new section objects to update existing database section rows:
                other_sections = {s.identifier: s for s in other.sections}
                self_sections = list(self.sections)  # make a copy of the list
                for s in self.sections:
                    if s.identifier in other_sections:
                        s.update(other_sections.pop(s.identifier))  # update sections from new list
                    else:
                        self_sections.remove(s)  # remove sections that don't exist in new list

                # add any new sections
                for s in other_sections.values():
                    self_sections.append(s)

                # make sure list is sorted by identifier at the end
                self_sections.sort(key=lambda s: s.identifier)

                # this assignment should also cause SQLA to update the array column
                self.sections = self_sections
                if self.sections is not None and self.id is not None:
                    [setattr(s, 'instrument_id', self.id) for s in self.sections]

            else:
                setattr(self, att, getattr(other, att))

    def load(self, filename, section_ids=None):
        """
        Load a part of an exposure file, based on the ccd_id.
        If the instrument does not have multiple CCDs, set ccd_id=0.

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

    def get_section_by_identifier(self, section_id):
        """
        Get the section with the given identifier,
        from the list of sections on this instrument.

        Parameters
        ----------
        section_id: str or int
            The identifier of the SensorSection object.
            This can be a serial number which is converted to a string.

        Returns
        -------
        section: SensorSection
            The section with the given identifier.
        """
        if not isinstance(section_id, (str, int)):
            raise ValueError(f"section_id must be a string or int. Got {type(section_id)}")
        section_id = str(section_id)
        section = next((section for section in self.sections if section.identifier == section_id), None)
        if section is None:
            raise ValueError(f"Could not find section with identifier {section_id}")

    def load_section(self, filename, section_id):
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
        section = self.get_section_by_identifier(section_id)

        raise NotImplementedError("This method must be implemented by the subclass.")

    def get_section_ids(self):
        """
        Get a list of SensorSection identifiers for this instrument.

        """
        return [section.identifier for section in self.sections]

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

    # TODO: when should this be called? Ideally on the first time we connect to the DB...
    @classmethod
    def _verify_instrument_on_db(cls, session=None):
        """
        Add this instrument to the database.
        This will add the instrument to the database, along with all the sections.
        This will make sure the instrument row on the DB is identical to the
        (sub-)class definition of the instrument, including all the hard coded defaults.

        Parameters
        ----------
        session: sqlalchemy.orm.session.Session
            The database session to use.
            If none is given, will open a new one and close it
            at the end of the function.
        """
        inst = cls()  # load the current hard-coded defaults
        with SmartSession(session) as session:
            db_inst = session.scalars(
                sa.select(Instrument).where(
                    Instrument.name == inst.name,
                    Instrument.telescope == inst.telescope,
                )
            ).first()  # should be unique combination of name/telescope

            if db_inst is None:
                session.add(inst)
                session.commit()
            else:
                if db_inst != inst:
                    print(f'Updating instrument {inst.name} on DB!')
                    db_inst.update(inst)  # also make sure sections are updated
                    session.commit()


class DemoInstrument(Instrument):

    __mapper_args__ = {
        "polymorphic_identity": "DemoInstrument",
    }

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
        self.filename_regex = ["Demo"]

        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def _make_sections():
        """
        Make a single section for the DEMO instrument.

        Returns
        -------
        sections: list of SensorSection
            The sections for this instrument.
        """
        return [SensorSection(0, 512, 1024)]

    def load_section(self, filename, section_id):
        """
        A spoof load method for this demo instrument.
        The data is just a random array.
        The instrument only has one section,
        so the input must be 0.

        Parameters
        ----------
        filename: str
            The filename of the exposure file.
            In this case the filename is not used.
        section_id: str or int
            The identifier of the SensorSection object.
            This instrument only has one section, so this must be 0 or "0".

        Returns
        -------
        data: np.ndarray
            The data from the exposure file.
        """
        section = self.get_section_by_identifier(section_id)

        return np.random.poisson(10, (section.size_y, section.size_x))

    def read_header(self, filename):
        return {}


class DECam(Instrument):
    __mapper_args__ = {
        "polymorphic_identity": "DECam",
    }

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
        self.filename_regex = None

        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def _make_sections():
        """
        Make the sections for the DECam instrument,
        including the sizes and offsets for each section.

        Returns
        -------
        sections: list
            A list of SensorSection objects.
        """
        # TODO: we must improve this!
        sections = []
        for i in range(62):
            sections.append(SensorSection(i, 2048, 4096, 0, i * 4096))

        return sections


if __name__ == "__main__":
    inst = DemoInstrument()
