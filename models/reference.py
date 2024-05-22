import sqlalchemy as sa
from sqlalchemy import orm, func

from models.base import Base, AutoIDMixin, SmartSession
from models.image import Image
from models.provenance import Provenance
from models.source_list import SourceList
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint


class Reference(Base, AutoIDMixin):
    """
    A table that refers to each reference Image object,
    based on the validity time range, and the object/field it is targeting.
    The provenance of this table (tagged with the "reference" process)
    will have as its upstream IDs the provenance IDs of the image,
    the source list, the PSF, the WCS, and the zero point.

    This means that the reference should always come loaded
    with the image and all its associated products,
    based on the provenance given when it was created.
    """

    __tablename__ = 'refs'   # 'references' is a reserved postgres word

    image_id = sa.Column(
        sa.ForeignKey('images.id', ondelete='CASCADE', name='references_image_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the reference image that this object is referring to. "
    )

    image = orm.relationship(
        'Image',
        lazy='selectin',
        cascade='save-update, merge, refresh-expire, expunge',
        foreign_keys=[image_id],
        doc="The reference image that this entry is referring to. "
    )

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

    instrument = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Name of the instrument used to make the images for this reference image. "
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

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='references_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this reference. "
            "The provenance will contain a record of the code version "
            "and the parameters used to produce this reference. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this reference. "
            "The provenance will contain a record of the code version "
            "and the parameters used to produce this reference. "
        )
    )

    def __init__(self, **kwargs):
        self.sources = None
        self.psf = None
        self.wcs = None
        self.zp = None
        super().__init__(**kwargs)

    def __setattr__(self, key, value):
        if key == 'image' and value is not None:
            self.target = value.target
            self.instrument = value.instrument
            self.filter = value.filter
            self.section_id = value.section_id
            self.sources = value.sources
            self.psf = value.psf
            self.wcs = value.wcs
            self.zp = value.zp

        super().__setattr__(key, value)

    @orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        self.sources = None
        self.psf = None
        self.wcs = None
        self.zp = None
        this_object_session = orm.Session.object_session(self)
        if this_object_session is not None:  # if just loaded, should usually have a session!
            self.load_upstream_products(this_object_session)

    def make_provenance(self):
        """Make a provenance for this reference image. """
        upstreams = [self.image.provenance]
        for att in ['image', 'sources', 'psf', 'wcs', 'zp']:
            if getattr(self, att) is not None:
                upstreams.append(getattr(self, att).provenance)
            else:
                raise ValueError(f'Reference must have a valid {att}.')

        self.provenance = Provenance(
            code_version=self.image.provenance.code_version,
            process='reference',
            parameters={},  # do we need any parameters for a reference's provenance?
            upstreams=upstreams,
        )

    def get_upstream_provenances(self):
        """Collect the provenances for all upstream objects.
        Assumes all the objects are already committed to the DB
        (or that at least they have provenances with IDs).

        Returns
        -------
        list of Provenance objects:
            a list of unique provenances, one for each data type.
        """
        prov = []
        if self.image is None or self.image.provenance is None or self.image.provenance.id is None:
            raise ValueError('Reference must have a valid image with a valid provenance ID.')
        prov.append(self.image.provenance)

        # TODO: it seems like we should require that Reference always has all of these when saved
        if self.sources is not None and self.sources.provenance is not None and self.sources.provenance.id is not None:
            prov.append(self.sources.provenance)
        if self.psf is not None and self.psf.provenance is not None and self.psf.provenance.id is not None:
            prov.append(self.psf.provenance)
        if self.wcs is not None and self.wcs.provenance is not None and self.wcs.provenance.id is not None:
            prov.append(self.wcs.provenance)
        if self.zp is not None and self.zp.provenance is not None and self.zp.provenance.id is not None:
            prov.append(self.zp.provenance)
        return prov

    def load_upstream_products(self, session=None):
        """Make sure the reference image has its related products loaded.

        This only works after the image and products are committed to the database,
        with provenances consistent with what is saved in this Reference's provenance.
        """
        with SmartSession(session) as session:
            prov_ids = self.provenance.upstream_ids

            sources = session.scalars(
                sa.select(SourceList).where(
                    SourceList.image_id == self.image_id,
                    SourceList.provenance_id.in_(prov_ids),
                )
            ).all()
            if len(sources) > 1:
                raise ValueError(
                    f"Image {self.image_id} has more than one SourceList matching upstream provenance."
                )
            elif len(sources) == 1:
                self.image.sources = sources[0]
                self.sources = sources[0]

            psfs = session.scalars(
                sa.select(PSF).where(
                    PSF.image_id == self.image_id,
                    PSF.provenance_id.in_(prov_ids),
                )
            ).all()
            if len(psfs) > 1:
                raise ValueError(
                    f"Image {self.image_id} has more than one PSF matching upstream provenance."
                )
            elif len(psfs) == 1:
                self.image.psf = psfs[0]
                self.psf = psfs[0]

            if self.sources is not None:
                wcses = session.scalars(
                    sa.select(WorldCoordinates).where(
                        WorldCoordinates.sources_id == self.sources.id,
                        WorldCoordinates.provenance_id.in_(prov_ids),
                    )
                ).all()
                if len(wcses) > 1:
                    raise ValueError(
                        f"Image {self.image_id} has more than one WCS matching upstream provenance."
                    )
                elif len(wcses) == 1:
                    self.image.wcs = wcses[0]
                    self.wcs = wcses[0]

                zps = session.scalars(
                    sa.select(ZeroPoint).where(
                        ZeroPoint.sources_id == self.sources.id,
                        ZeroPoint.provenance_id.in_(prov_ids),
                    )
                ).all()
                if len(zps) > 1:
                    raise ValueError(
                        f"Image {self.image_id} has more than one ZeroPoint matching upstream provenance."
                    )
                elif len(zps) == 1:
                    self.image.zp = zps[0]
                    self.zp = zps[0]

    def merge_all(self, session):
        """Merge the reference into the session, along with Image and products. """

        new_ref = session.merge(self)
        new_ref.image = self.image.merge_all(session)

        return new_ref

    @staticmethod
    def check_reference(
            ref,
            instrument,
            filter,
            ra,
            dec,
            target,
            section_id,
            obs_time,
            minovfrac=0.85,
            must_match_instrument=True,
            must_match_filter=True,
            must_match_target=False,
            must_match_section=False,
    ):
        """Check if the given reference is valid for the given instrument, filter, coordinates,
        and observation time.  You may also require, instead of coordinates,
        that the target and section_id match.

        If the reference has is_bad==True, it will also not be considered valid.

        Parameters
        ----------
        ref: Reference object
            The reference object to check against the given inputs of a new image/exposure.
        instrument: str
            The name of the instrument used to produce the image/exposure.
        filter: str
            The filter of the image/exposure.
        ra: float
            The right ascension of the center of the image/exposure. Given in degrees.
            If minovfrac is not given (or is zero) will not require overlap of reference
            and image coordinates. Instead, use target and section_id to check the reference.
            By default, the check IS done using coordinates and ignores target and section_id.
        dec: float
            The declination of the center of the image/exposure. Given in degrees.
            If minovfrac is not given (or is zero) will not require overlap of reference
            and image coordinates. Instead, use target and section_id to check the reference.
            By default, the check IS done using coordinates and ignores target and section_id.
        target: str
            The target of the image/exposure, or the name of the field.
            This is only used when must_match_target is True.
            By default, this isn't used, and the coordinate overlap is used.
        section_id: str
            The section ID of the image/exposure.
            This is only used when must_match_section is True.
            By default, this isn't used, and the coordinate overlap is used.
        obs_time: datetime
            The observation time of the image. This is used to check that the reference
            has the correct validity start and end times.
        minovfrac: float, default 0.85
            Area of overlap region must be at least this fraction of the
            area of the search image/exposure for the reference to match.
            (Warning: calculation implicitly assumes that images are
            aligned N/S and E/W.)  Make this None or <= 0 to not consider
            overlap fraction when finding a reference.
            In such cases, it is recommended to use the target and section_id
            to check references instead.
        must_match_instrument: bool, default True
            If True, only approve a reference from the same instrument
            as that of the image/exposure.
        must_match_filter: bool, default True
            If True, only approve a reference whose filter matches the
            filter of the image/exposure.
        must_match_target: bool, default False
            If True, only approve a reference if the "target" field of the
            reference image matches the "target" field of the image/exposure.
        must_match_section: bool, default False
            If True, only approve a reference if the "section_id" field of
            the reference image matches that of the image/exposure.

        Returns
        -------
        bool:
            True if the reference is valid for the given inputs.
        """
        return (
                (ref.validity_start is None or ref.validity_start <= obs_time) and
                (ref.validity_end is None or ref.validity_end >= obs_time) and
                ref.filter == filter and ref.target == target and
                ref.is_bad is False
        )

    @staticmethod
    def get_reference(
            instrument,
            filter,
            ra,
            dec,
            target,
            section_id,
            obs_time,
            maxoffset=0.1,
            minovfrac=None,
            must_match_instrument=True,
            must_match_filter=True,
            must_match_target=False,
            must_match_section=False,
            session=None
    ):
        """
        Get a reference for a given instrument, filter, coordinates, and observation time.
        (or, if you prefer, the given target, and section_id instead of coordinates).

        Will only consider references whose validity date range
        includes the given obs_time.

        If minovfrac is given, it will return the reference that has the
        highest ovfrac.

        If minovfrac is not given, it will return the first reference found
        that matches the other criteria.  Be careful with this.

        References with is_bad==True will not be considered.

        Parameters
        ----------
        instrument: str
            The name of the instrument used to produce the image/exposure.
        filter: str
            The filter of the image/exposure.
        ra: float
            The right ascension of the center of the image/exposure. Given in degrees.
            If minovfrac is not given (or is zero) will not require overlap of reference
            and image coordinates. Instead, use target and section_id to match the reference.
            By default, the match IS done using coordinates and ignores target and section_id.
        dec: float
            The declination of the center of the image/exposure. Given in degrees.
            If minovfrac is not given (or is zero) will not require overlap of reference
            and image coordinates. Instead, use target and section_id to match the reference.
            By default, the match IS done using coordinates and ignores target and section_id.
        target: str
            The target of the image/exposure, or the name of the field.
            This is only used when must_match_target is True.
            By default, this isn't used, and the coordinate overlap is used.
        section_id: str
            The section ID of the image/exposure.
            This is only used when must_match_section is True.
            By default, this isn't used, and the coordinate overlap is used.
        obs_time: datetime
            The observation time of the image. This is used to find a reference
            with the correct validity start and end times.
        maxoffset: float, default 0.1
            Maximum offset in degrees between the given ra/dec and the reference's center coordinates.
        minovfrac: float, default None
            Area of overlap region must be at least this fraction of the
            area of the search image/exposure for the reference to be good.
            (Warning: calculation implicitly assumes that images are
            aligned N/S and E/W.)  Make this None or <= 0 to not consider
            overlap fraction when finding a reference.
            In such cases, it is recommended to use the target and section_id
            to identify references instead.
        must_match_instrument: bool, default True
            If True, only find a reference from the same instrument
            as that of the image/exposure.
        must_match_filter: bool, default True
            If True, only find a reference whose filter matches the
            filter of the image/exposure.
        must_match_target: bool, default False
            If True, only find a reference if the "target" field of the
            reference image matches the "target" field of the image/exposure.
        must_match_section: bool, default False
            If True, only find a reference if the "section_id" field of
            the reference image matches that of the image/exposure.
        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will open a new session
            and close it at the end of the function.

        Returns
        -------
        ref: Image object
            The reference image for this image, or None if no reference is found.


        NOTE: If, by unlikely chance, multiple references have
        identical overlap fractions, an undeterministically chosen
        reference will be returned.  Ideally, by construction, you will
        never have this situation in your database; you will only have a
        single valid reference image for a given instrument/filter/date
        that has an appreciable overlap with any possible image from
        that instrument.  The software does not enforce this, however.
        """
        with SmartSession(session) as session:
            stmt = sa.select(Reference, Image).where(
                Image.id == Reference.image_id,
                sa.or_(
                    Reference.validity_start.is_(None),
                    Reference.validity_start <= obs_time
                ),
                sa.or_(
                    Reference.validity_end.is_(None),
                    Reference.validity_end >= obs_time
                ),
                Reference.is_bad.is_(False),
            )
            if must_match_instrument:
                stmt = stmt.where(Reference.instrument == instrument)
            if must_match_filter:
                stmt = stmt.where(Reference.filter == filter)
            if must_match_target:
                stmt = stmt.where(Reference.target == target)
            if must_match_section:
                stmt = stmt.where(Reference.section_id == section_id)

            # get the latest references first.
            stmt = stmt.order_by(Reference.created_at.desc())

            if maxoffset is not None and maxoffset > 0:
                stmt = stmt.where(func.q3c_radial_query(ra, dec, Reference.ra, Reference.dec, maxoffset))
            elif minovfrac is not None and minovfrac > 0:
                # stmt = stmt.where(Image.containing(ra, dec))  # use FourCorner's filter method

                maxov = minovfrac
                for try_ref, try_refim in session.execute(stmt).all():
                    ovfrac = try_refim._overlap_frac(ra, dec)
            else:
                ref, refim = session.execute(stmt).first()

        return ref
