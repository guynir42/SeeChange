
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_property

from models.base import Base, FileOnDiskMixin, SpatiallyIndexed
from models.enums_and_bitflags import cutouts_format_dict, cutouts_format_converter


class Cutouts(Base, FileOnDiskMixin, SpatiallyIndexed):

    __tablename__ = 'cutouts'

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=cutouts_format_converter('fits'),
        doc="Format of the file on disk. Should be fits, hdf5, csv or npy. "
            "Saved as integer but is converter to string when loaded. "
    )

    @hybrid_property
    def format(self):
        return cutouts_format_converter(self._format)

    @format.expression
    def format(cls):
        # ref: https://stackoverflow.com/a/25272425
        return sa.case(cutouts_format_dict, value=cls._format)

    @format.setter
    def format(self, value):
        self._format = cutouts_format_converter(value)

    source_list_id = sa.Column(
        sa.ForeignKey('source_lists.id'),
        nullable=False,
        index=True,
        doc="ID of the source list this cutout is associated with. "
    )

    source_list = orm.relationship(
        'SourceList',
        doc="The source list this cutout is associated with. "
    )

    new_image_id = sa.Column(
        sa.ForeignKey('images.id'),
        nullable=False,
        index=True,
        doc="ID of the new science image this cutout is associated with. "
    )

    new_image = orm.relationship(
        'Image',
        primaryjoin="Cutouts.new_image_id==Image.id",
        doc="The new science image this cutout is associated with. "
    )

    ref_image_id = sa.Column(
        sa.ForeignKey('images.id'),
        nullable=False,
        index=True,
        doc="ID of the reference image this cutout is associated with. "
    )

    ref_image = orm.relationship(
        'Image',
        primaryjoin="Cutouts.ref_image_id==Image.id",
        doc="The reference image this cutout is associated with. "
    )

    sub_image_id = sa.Column(
        sa.ForeignKey('images.id'),
        nullable=False,
        index=True,
        doc="ID of the subtraction image this cutout is associated with. "
    )

    sub_image = orm.relationship(
        'Image',
        primaryjoin="Cutouts.sub_image_id==Image.id",
        doc="The subtraction image this cutout is associated with. "
    )

    pixel_x = sa.Column(
        sa.Integer,
        nullable=False,
        doc="X pixel coordinate of the center of the cutout. "
    )

    pixel_y = sa.Column(
        sa.Integer,
        nullable=False,
        doc="Y pixel coordinate of the center of the cutout. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this cutout. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this cutout. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this cutout. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this cutout. "
        )
    )

