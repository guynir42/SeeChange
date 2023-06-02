import os
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.types import Enum
from sqlalchemy.dialects.postgresql import JSONB

from pipeline.utils import read_fits_image, save_fits_image_file, save_fits_image_file_combined

from models.base import SeeChangeBase, Base, FileOnDiskMixin, SpatiallyIndexed
from models.exposure import Exposure, im_type_enum
from models.provenance import Provenance

import util.config as config

image_source_self_association_table = sa.Table(
    'image_sources',
    Base.metadata,
    sa.Column('source_id', sa.Integer, sa.ForeignKey('images.id', ondelete="CASCADE"), primary_key=True),
    sa.Column('combined_id', sa.Integer, sa.ForeignKey('images.id', ondelete="CASCADE"), primary_key=True),
)


class Image(Base, FileOnDiskMixin, SpatiallyIndexed):

    __tablename__ = 'images'

    exposure_id = sa.Column(
        sa.ForeignKey('exposures.id'),
        nullable=True,
        index=True,
        doc=(
            "ID of the exposure from which this image was derived. "
            "Only set for single-image objects."
        )
    )

    exposure = orm.relationship(
        'Exposure',
        cascade='save-update, merge, refresh-expire, expunge',
        doc=(
            "Exposure from which this image was derived. "
            "Only set for single-image objects."
        )
    )

    source_images = orm.relationship(
        "Image",
        secondary=image_source_self_association_table,
        primaryjoin='images.c.id == image_sources.c.combined_id',
        secondaryjoin='images.c.id == image_sources.c.source_id',
        passive_deletes=True,
        lazy='selectin',  # should be able to get source_images without a session!
        doc=(
            "Images used to produce a multi-image object "
            "(e.g., an images stack, reference, difference, super-flat, etc)."
        )
    )

    @property
    def is_multi_image(self):
        if self.exposure_id is not None:
            return False
        elif self.source_images is not None and len(self.source_images) > 0:
            return True
        else:
            return None  # for new objects that have not defined either exposure_id or source_images

    combine_method = sa.Column(
        Enum("coadd", "subtraction", name='image_combine_method'),
        nullable=True,
        index=True,
        doc=(
            "Type of combination used to produce this multi-image object. "
            "One of: coadd, subtraction. "
        )
    )

    type = sa.Column(
        im_type_enum,  # defined in models/exposure.py
        nullable=False,
        default="science",
        index=True,
        doc=(
            "Type of image. One of: science, reference, difference, bias, dark, flat. "
        )
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this image. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this image. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        doc=(
            "Provenance of this image. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this image. "
        )
    )

    header = sa.Column(
        JSONB,
        nullable=False,
        default={},
        doc=(
            "Header of the specific image for one section of the instrument. "
            "Only keep a subset of the keywords, "
            "and re-key them to be more consistent. "
        )
    )

    mjd = sa.Column(
        sa.Double,
        nullable=False,
        index=True,
        doc=(
            "Modified Julian date of the exposure (MJD=JD-2400000.5). "
            "Multi-exposure images will have the MJD of the first exposure."
        )
    )

    @property
    def start_mjd(self):
        """Time of the beginning of the exposure, or set of exposures (equal to mjd). """
        return self.mjd

    @property
    def mid_mjd(self):
        """
        Time of the middle of the exposures.
        For multiple, coadded exposures (e.g., references), this would
        be the middle between the start_mjd and end_mjd, regarless of
        how the exposures are spaced.
        """
        if self.start_mjd is None or self.end_mjd is None:
            return None
        return (self.start_mjd + self.end_mjd) / 2.0

    end_mjd = sa.Column(
        sa.Double,
        nullable=False,
        index=True,
        doc=(
            "Modified Julian date of the end of the exposure. "
            "Multi-image object will have the end_mjd of the last exposure."
        )
    )

    exp_time = sa.Column(
        sa.Float,
        nullable=False,
        index=True,
        doc="Exposure time in seconds. Multi-exposure images will have the total exposure time."
    )

    instrument = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Name of the instrument used to create this image. "
    )

    telescope = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Name of the telescope used to create this image. "
    )

    filter = sa.Column(sa.Text, nullable=False, index=True, doc="Name of the filter used to make this image. ")

    section_id = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Section ID of the image, possibly inside a larger mosiaced exposure. '
    )

    project = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Name of the project (could also be a proposal ID). '
    )

    target = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Name of the target object or field id. '
    )

    def __init__(self, *args, **kwargs):
        FileOnDiskMixin.__init__(self, *args, **kwargs)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        self.raw_data = None  # the raw exposure pixels (2D float or uint16 or whatever) not saved to disk!
        self._raw_header = None  # the header data taken directly from the FITS file
        self._data = None  # the underlying pixel data array (2D float array)
        self._flags = None  # the bit-flag array (2D int array)
        self._weight = None  # the inverse-variance array (2D float array)
        self._background = None  # an estimate for the background flux (2D float array)
        self._score = None  # the image after filtering with the PSF and normalizing to S/N units (2D float array)
        self._psf = None  # a small point-spread-function image (2D float array)

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.ra is not None and self.dec is not None:
            self.calculate_coordinates()  # galactic and ecliptic coordinates

    @orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        self.raw_data = None
        self._raw_header = None
        self._data = None
        self._flags = None
        self._weight = None
        self._background = None
        self._score = None
        self._psf = None

    @classmethod
    def from_exposure(cls, exposure, section_id):
        """
        Copy the raw pixel values and relevant metadata from a section of an Exposure.

        Parameters
        ----------
        exposure: Exposure
            The exposure object to copy data from.
        section_id: int or str
            The part of the sensor to use when making this image.
            For single section instruments this is usually set to 0.

        Returns
        -------
        image: Image
            The new Image object. It would not have any data variables
            except for raw_data (and all the metadata values).
            To fill out data, flags, weight, etc., the application must
            apply the proper preprocessing tools (e.g., bias, flat, etc).

        """
        if not isinstance(exposure, Exposure):
            raise ValueError(f"The exposure must be an Exposure object. Got {type(exposure)} instead.")

        new = cls()

        same_columns = [
            'type',
            'mjd',
            'end_mjd',
            'exp_time',
            'instrument',
            'telescope',
            'filter',
            'project',
            'target'
        ]

        # copy all the columns that are the same
        for column in same_columns:
            setattr(new, column, getattr(exposure, column))

        if exposure.filter_array is not None:
            idx = exposure.instrument_object.get_section_filter_array_index(section_id)
            new.filter = exposure.filter_array[idx]

        # TODO: must make a better effort to get the correct RA/Dec (from the header?)
        new.ra = exposure.ra
        new.dec = exposure.dec

        # TODO: read the header from the exposure file's individual section data

        new.section_id = section_id
        new.raw_data = exposure.data[section_id]

        # the exposure_id will be set automatically at commit time
        new.exposure = exposure

        return new

    def __repr__(self):

        output = (
            f"Image(id: {self.id}, "
            f"type: {self.type}, "
            f"exp: {self.exp_time}s, "
            f"filt: {self.filter}, "
            f"from: {self.instrument}/{self.telescope}"
        )

        multi_type = str(self.combine_method) if self.is_multi_image else None

        if multi_type is not None:
            output += f", multi_type: {multi_type}"

        output += ")"

        return output

    def __str__(self):
        return self.__repr__()

    def save(self, filename=None):
        """
        Save the data (along with flags, weights, etc.) to disk.
        The format to save is determined by the config file.
        Use the filename to override the default naming convention.

        Parameters
        ----------
        filename: str (optional)
            The filename to use to save the data.
            If not provided, the default naming convention will be used.
        """
        pass  # TODO: finish this

    def load(self):
        """
        Load the image data from disk.
        This includes the _data property,
        but can also load the _flags, _weight,
        _background, _score, and _psf properties.

        """

        if self.filepath is None:
            raise ValueError("The filepath is not set. Cannot load the image.")

        cfg = config.Config.get()
        single_file = cfg.value('storage.images.single_file')

        if single_file:
            filename = self.get_fullpath()
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"Could not find the image file: {filename}")
            self._data, self._raw_header = read_fits_image(filename, ext=0, output='both')
            # TODO: do we know what the extensions are for weight/flags/etc? are they ordered or named?
            self._flags = read_fits_image(filename, ext=1)  # TODO: is the flags always extension 1??
            self._weight = read_fits_image(filename, ext=2)  # TODO: is the weight always extension 2??

        else:  # save each data array to a separate file
            # assumes there are filepath_extensions so get_fullpath() will return a list (as_list guarantees this)
            for filename in self.get_fullpath(as_list=True):
                if not os.path.isfile(filename):
                    raise FileNotFoundError(f"Could not find the image file: {filename}")

                self._data, self._raw_header = read_fits_image(filename, output='both')

    @property
    def data(self):
        """
        The underlying pixel data array (2D float array).
        """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def raw_header(self):
        if self._raw_header is None and self.filepath is not None:
            self.load()
        return self._raw_header

    @raw_header.setter
    def raw_header(self, value):
        self._raw_header = value

    @property
    def flags(self):
        """
        The bit-flag array (2D int array).
        """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._flags

    @flags.setter
    def flags(self, value):
        self._flags = value

    @property
    def weight(self):
        """
        The inverse-variance array (2D float array).
        """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    @property
    def background(self):
        """
        An estimate for the background flux (2D float array).
        """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._background

    @background.setter
    def background(self, value):
        self._background = value

    @property
    def score(self):
        """
        The image after filtering with the PSF and normalizing to S/N units (2D float array).
        """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @property
    def psf(self):
        """
        A small point-spread-function image (2D float array).
        """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._psf

    @psf.setter
    def psf(self, value):
        self._psf = value


if __name__ == '__main__':
    im = Image()
    print(im.id)
