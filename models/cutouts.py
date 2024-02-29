import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.ext.associationproxy import association_proxy

from models.base import Base, SeeChangeBase, AutoIDMixin, FileOnDiskMixin, SpatiallyIndexed, HasBitFlagBadness
from models.enums_and_bitflags import CutoutsFormatConverter


class Cutouts(Base, AutoIDMixin, FileOnDiskMixin, SpatiallyIndexed, HasBitFlagBadness):

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
        default=CutoutsFormatConverter.convert('hdf5'),
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

        self._source_row = None
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
        self.format = 'hdf5'  # the default should match the column-defined default above!
        self._source_row = None
        self._sub_data = None
        self._sub_weight = None
        self._sub_flag = None
        self._ref_data = None
        self._ref_weight = None
        self._ref_flag = None
        self._new_data = None

    def __repr__(self):
        return (
            f"<Cutouts {self.id} "
            f"from SourceList {self.sources_id} "
            f"(number {self.index_in_sources}) "
            f"from Image {self.sub_image_id} "
            f"at x,y= {self.pixel_x}, {self.pixel_y}>"
        )

    @staticmethod
    def get_data_attributes():
        names = ['source_row']
        for im in ['sub', 'ref', 'new']:
            for att in ['data', 'weight', 'flags']:
                names.append(f'{im}_{att}')
        return names

    @property
    def has_data(self):
        for att in self.get_data_attributes():
            if getattr(self, att) is None:
                return False

    @property
    def sub_nandata(self):
        if self.sub_data is None or self.sub_flags is None:
            return None
        return np.where(self.sub_flags > 0, np.nan, self.sub_data)

    @property
    def ref_nandata(self):
        if self.ref_data is None or self.ref_flags is None:
            return None
        return np.where(self.ref_flags > 0, np.nan, self.ref_data)

    @property
    def new_nandata(self):
        if self.new_data is None or self.new_flags is None:
            return None
        return np.where(self.new_flags > 0, np.nan, self.new_data)

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
        cutout.source_row = dict(detections.data[source_index])
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

    def save(self, filename=None, **kwargs):
        """Save a single Cutouts object into a file.

        Parameters
        ----------
        filename: str, optional
            The (relative/full path) filename to save to. If not given, will use the default filename.
        kwargs: dict
            Any additional keyword arguments to pass to the FileOnDiskMixin.save method.
        """
        if not self.has_data:
            raise RuntimeError("The Cutouts data is not loaded. Cannot save.")

    @classmethod
    def save_list(cls, cutouts_list):
        pass

    def load(self, filepath):
        pass

    def load_list(self, filepath):
        pass


# use these two functions to quickly add the "property" accessor methods
def load_attribute(object, att):
    """Load the data for a given attribute of the object."""
    if not hasattr(object, f'_{att}'):
        raise AttributeError(f"The object {object} does not have the attribute {att}.")
    if getattr(object, f'_{att}') is None:
        if object.filepath is None:
            return None  # objects just now created and not saved cannot lazy load data!
        object.load()  # can lazy-load all data

    # after data is filled, should be able to just return it
    return getattr(object, f'_{att}')


def set_attribute(object, att, value):
    """Set the value of the attribute on the object. """
    setattr(object, f'_{att}', value)


# add "@property" functions to all the data attributes
for att in Cutouts.get_data_attributes():
    setattr(
        Cutouts,
        att,
        property(
            fget=lambda self, att=att: load_attribute(self, att),
            fset=lambda self, value, att=att: set_attribute(self, att, value),
        )
    )
