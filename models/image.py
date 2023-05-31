
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import JSONB

from models.base import Base, FileOnDiskMixin, SpatiallyIndexed


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
        'Image',
        cascade='save-update, merge, refresh-expire, expunge',
        doc=(
            "Images used to produce a multi-image object "
            "(e.g., an images stack, reference, difference, super-flat, etc)."
        )
    )

    is_multi_image = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        index=True,
        doc=(
            "Whether this image is a multi-image object. "
            "Multi-image objects will have a null exposure, "
            "and a list of Image objects from which it was derived. "
        )
    )

    combine_method = sa.Column(
        sa.Text,
        nullable=True,
        index=True,
        doc=(
            "Type of combination used to produce this multi-image object. "
            "One of: coadd, subtraction. "
        )
    )

    type = sa.Column(
        sa.Text,
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

    exp_time = sa.Column(
        sa.Float,
        nullable=False,
        index=True,
        doc="Exposure time in seconds. Multi-exposure images will have the total exposure time."
    )

    filter = sa.Column(sa.Text, nullable=False, index=True, doc="Filter name")

    section_id = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Section ID of the image, possibly inside a larger mosiaced exposure. '
    )



if __name__ == '__main__':
    im = Image()
    print(im.id)
