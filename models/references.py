import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.associationproxy import association_proxy

from models.base import Base, AutoIDMixin, SmartSession
from models.image import Image, ImageProducts


class Reference(Base, AutoIDMixin):
    """
    A table that refers to each reference ImageProducts object,
    based on the validity time range, and the object/field it is targeting.
    The ImageProducts includes the Image but also the PSF, SourceList, WorldCoordinates, and ZeroPoint.
    """

    __tablename__ = 'references'

    products_id = sa.Column(
        sa.ForeignKey('image_products.id', ondelete='CASCADE', name='references_image_products_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the reference image products that this object is referring to. "
    )

    products = orm.relationship(
        'ImageProducts',
        lazy='selectin',
        cascade='save-update, merge, refresh-expire, expunge',
        foreign_keys=[products_id],
        doc="The reference image products that this entry is referring to. "
    )

    image_id = association_proxy("products", "image_id")
    image = association_proxy("products", "image", )
    sources_id = association_proxy("products", "sources_id")
    sources = association_proxy("products", "sources")
    psf_id = association_proxy("products", "psf_id")
    psf = association_proxy("products", "psf")
    wcs_id = association_proxy("products", "wcs_id")
    wcs = association_proxy("products", "wcs")
    zp_id = association_proxy("products", "zp_id")
    zp = association_proxy("products", "zp")

    # the following can't be association products (as far as I can tell) because they need to be indexed
    target = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc=(
            'Name of the target object or field id. '
            'This string is used to match the reference to new images, '
            'e.g., by matching the field ID on a pre-defined grid of fields. '
        )
    )

    filter = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Filter used to make the images for this reference image. "
    )

    section_id = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Section ID of the reference image. "
    )

    # this allows choosing a different reference for images taken before/after the validity time range
    validity_start = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc="The start of the validity time range of this reference image. "
    )

    validity_end = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc="The end of the validity time range of this reference image. "
    )

    # this badness is in addition to the regular bitflag of the underlying products
    # it can be used to manually kill a reference and replace it with another one
    # even if they share the same time validity range
    is_bad = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc="Whether this reference image is bad. "
    )

    bad_reason = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "The reason why this reference image is bad. "
            "Should be a single pharse or a comma-separated list of reasons. "
        )
    )

    bad_comment = sa.Column(
        sa.Text,
        nullable=True,
        doc="Any additional comments about why this reference image is bad. "
    )

    # this table doesn't have provenance.
    # The underlying products will have their own provenance for the "coaddition" process.

    def __setattr__(self, key, value):
        if key == 'image':
            if value is not None:
                self.filter = value.filter
                self.target = value.target
                self.section_id = value.section_id
                # first assign the image, then do stuff with it
                super().__setattr__(key, value)
                if self.products is not None:
                    self.image.psf = self.products.psf
                    self.image.sources = self.products.sources
                    self.image.wcs = self.products.wcs
                    self.image.zp = self.products.zp
                return

        if key == 'products':
            if value is not None:
                self.filter = value.image.filter
                self.target = value.image.target
                self.section_id = value.image.section_id

            # first assign the products, then do stuff with it
            super().__setattr__(key, value)
            self.image = value.image
            self.image.psf = value.psf
            self.image.sources = value.sources
            self.image.wcs = value.wcs
            self.image.zp = value.zp
            return

        super().__setattr__(key, value)

    @orm.reconstructor
    def init_on_load(self):
        self.products.image.sources = self.products.sources
        self.products.image.psf = self.products.psf
        self.products.image.wcs = self.products.wcs
        self.products.image.zp = self.products.zp

    def __setattr__(self, key, value):
        if key == 'image':
            self.target = value.target
            self.filter = value.filter
            self.section_id = value.section_id

        super().__setattr__(key, value)