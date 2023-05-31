
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.schema import CheckConstraint
from sqlalchemy.types import Enum
from sqlalchemy.dialects.postgresql import JSONB

from models.base import SeeChangeBase, Base, FileOnDiskMixin, SpatiallyIndexed
from models.exposure import Exposure, im_type_enum
from models.provenance import Provenance

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
        index=True,
        doc=(
            "Type of image. One of: science, reference, difference, bias, dark, flat. "
        )
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id'),
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

    def __init__(self, **kwargs):
        FileOnDiskMixin.__init__(self, **kwargs)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

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
        self._data = None
        self._flags = None
        self._weight = None
        self._background = None
        self._score = None
        self._psf = None

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


if __name__ == '__main__':
    im = Image()
    print(im.id)
