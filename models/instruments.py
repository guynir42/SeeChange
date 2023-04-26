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


class SensorSection(Base):
    """
    A class to represent a section of a sensor.
    This is most often associated with a CCD chip, but could be any
    section of a sensor. For example, a section of a CCD chip that
    is read out independently, or different channels in a dichroic
    imager.
    """

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

    @staticmethod
    def get_attribute_list():
        """
        Get a list of all attributes of this object,
        not including internal SQLAlchemy attributes,
        and database level attributes like id, created_at, etc.
        """
        return ['identifier', 'size_x', 'size_y', 'offset_x', 'offset_y']

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

    pixels_scale = sa.Column(
        sa.Float,
        nullable=False,
        doc='Pixel scale of the instrument (in arcsec/pixel). '
    )

    aperture = sa.Column(
        sa.Float,
        nullable=False,
        doc='Aperture of the instrument (in meters). '
    )

    allowed_filters = sa.Column(
        sa.ARRAY(sa.String),
        nullable=False,
        doc='A list of allowed string names of filters this instrument can use. '
    )

    sections = sa.orm.relationship(
        SensorSection,
        backref='instrument',
        cascade='all, delete-orphan',
        lazy='selectin',  # load these by default
        order_by=SensorSection.identifier,
        doc='A list of sections of the instrument. '
    )

    __table_args__ = (UniqueConstraint("name", "telescope", name="_instrument_name_telescope_uc"),)

    def get_attribute_list(self):
        """
        Get a list of all attributes of this object,
        not including internal SQLAlchemy attributes,
        and database level attributes like id, created_at, etc.
        """
        return ['identifier', 'size_x', 'size_y', 'offset_x', 'offset_y']

    def __init__(self):
        self.name = None
        self.telescope = None
        self.pixel_scale = None
        self.aperture = None
        self.allowed_filters = []

        self.sections = []

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
                return False

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

            else:
                setattr(self, att, getattr(other, att))

    @staticmethod
    def get_attribute_list():
        """
        Get a list of all attributes of this object,
        not including internal SQLAlchemy attributes,
        and database level attributes like id, created_at, etc.
        """
        return ["name", "telescope", "pixel_scale", "aperture", "allowed_filters", "sections"]

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
    def verify_instrument_on_db(cls, session=None):
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
                    db_inst.update(inst)  # also make sure sections are updated
                    session.commit()


class Demo(Instrument):

    def __init__(self):
        self.pixel_scale = 0.263
        self.aperture = 4.0
        self.allowed_filters = ["g", "r", "i", "z", "Y"]
        self.sections = [SensorSection(0, 512, 1024)]

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
    pass

