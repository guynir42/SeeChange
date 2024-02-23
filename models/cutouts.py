
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.ext.associationproxy import association_proxy

from models.base import Base, SeeChangeBase, AutoIDMixin, FileOnDiskMixin, SpatiallyIndexed
from models.enums_and_bitflags import CutoutsFormatConverter


class Cutouts(Base, AutoIDMixin, FileOnDiskMixin, SpatiallyIndexed):

    __tablename__ = 'cutouts'

    # a unique constraint on the provenance and the source list, but also on the index in the list
    __table_args__ = (
        UniqueConstraint(
            'index_in_sources', 'sources_id', 'provenance_id', name='_cutouts_index_sources_provenance_uc'
        ),
    )

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=CutoutsFormatConverter.convert('fits'),
        doc="Format of the file on disk. Should be fits, hdf5, csv or npy. "
            "Saved as integer but is converter to string when loaded. "
    )

    @hybrid_property
    def format(self):
        return CutoutsFormatConverter.convert(self._format)

    @format.expression
    def format(cls):
        # ref: https://stackoverflow.com/a/25272425
        return sa.case(CutoutsFormatConverter.dict, value=cls._format)

    @format.setter
    def format(self, value):
        self._format = CutoutsFormatConverter.convert(value)

    sources_id = sa.Column(
        sa.ForeignKey('source_lists.id', name='cutouts_source_list_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the source list (of detections in the difference image) this cutout is associated with. "
    )

    sources = orm.relationship(
        'SourceList',
        doc="The source list (of detections in the difference image) this cutout is associated with. "
    )

    index_in_sources = sa.Column(
        sa.Integer,
        nullable=False,
        doc="Index of this cutout in the source list (of detections in the difference image). "
    )

    sub_image_id = association_proxy('sources', 'image_id')
    sub_image = association_proxy('sources', 'image')

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
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='cutouts_provenance_id_fkey'),
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

    _bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        default=0,
        index=True,
        doc='Bitflag for these cutouts. Good cutouts have a bitflag of 0. '
            'Bad cutouts are each bad in their own way (i.e., have different bits set). '
            'Will include all the bits from data used to make these cutouts '
            '(e.g., the exposure it is based on). '
    )

    @hybrid_property
    def bitflag(self):
        return self._bitflag | self.image.bitflag

    @bitflag.expression
    def bitflag(cls):
        sa.select(Cutouts).where(
            Cutouts._bitflag,
            Cutouts.ref_image.bitflag,
            Cutouts.new_image.bitflag,
            Cutouts.sub_image.bitflag,
            Cutouts.source_list.bitflag,
        ).label('bitflag')

    @bitflag.setter
    def bitflag(self, value):
        self._bitflag = value

    description = sa.Column(
        sa.Text,
        nullable=True,
        doc='Free text comment about this source list, e.g., why it is bad. '
    )

    @property
    def new_image(self):
        """Get the aligned new image using the sub_image. """
        return self.sub_image.new_aligned_image

    @property
    def ref_image(self):
        """Get the aligned reference image using the sub_image. """
        return self.sub_image.ref_aligned_image

    def __init__(self, *args, **kwargs):
        FileOnDiskMixin.__init__(self, *args, **kwargs)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        self.source_row = None
        self._sub_data = None
        self._sub_weight = None
        self._sub_flag = None
        self._ref_data = None
        self._ref_weight = None
        self._ref_flag = None
        self._new_data = None

        self._bitflag = 0

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        FileOnDiskMixin.init_on_load(self)
        self.source_row = None
        self._sub_data = None
        self._sub_weight = None
        self._sub_flag = None
        self._ref_data = None
        self._ref_weight = None
        self._ref_flag = None
        self._new_data = None

    @staticmethod
    def from_detections(detections, source_index, provenance=None, **kwargs):
        """Create a Cutout object from a row in the SourceList.

        The SourceList must have a valid image attribute, and that image should have exactly two
        upstream_images: the reference and new image. Each Cutout will have three small stamps
        from the new, reference, and subtraction images.

        Parameters
        ----------
        detections: SourceList
            The source list from which to create the cutout.
        source_index: int
            The index of the source in the source list from which to create the cutout.
        provenance: Provenance, optional
            The provenance of the cutout. If not given, will leave as None (to be filled externally).
        kwargs: dict
            Can include any of the following keys, in the format: {im}_{att}, where
            the {im} can be "sub", "ref", or "new", and the {att} can be "data", "weight", or "flags".
            These are optional, to be used to fill the different data attributes of this object.

        Returns
        -------
        cutout: Cutout
            The cutout object.
        """
        cutout = Cutouts()
        cutout.sources = detections
        cutout.index_in_sources = source_index
        cutout.source_row = detections.data[source_index]
        cutout.pixel_x = detections.x[source_index]
        cutout.pixel_y = detections.y[source_index]
        cutout.provenance = provenance

        # add the data, weight, and flags to the cutout from kwargs
        for im in ['sub', 'ref', 'new']:
            for att in ['data', 'weight', 'flags']:
                setattr(cutout, f'{im}_{att}', kwargs.get(f'{im}_{att}', None))

        # update the bitflag
        cutout._upstream_bitflag = detections.bitflag

        return cutout
