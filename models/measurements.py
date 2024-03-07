
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

from models.base import Base, SeeChangeBase, AutoIDMixin, SpatiallyIndexed


class Measurements(Base, AutoIDMixin, SpatiallyIndexed):

    __tablename__ = 'measurements'

    __table_args__ = (
        UniqueConstraint('cutouts_id', 'provenance_id', name='_measurements_cutouts_provenance_uc'),
    )

    cutouts_id = sa.Column(
        sa.ForeignKey('cutouts.id', name='measurements_cutouts_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the cutout this measurement is associated with. "
    )

    cutouts = orm.relationship(
        'Cutouts',
        doc="The cutout this measurement is associated with. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='measurements_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the provenance of this measurement. "
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc="The provenance of this measurement. "
    )

    mjd = sa.Column(
        sa.Float,
        nullable=False,
        index=True,
        doc="MJD of the measurement. "
    )

    exptime = sa.Column(
        sa.Float,
        nullable=False,
        index=True,
        doc="Exposure time of the measurement. "
    )

    filter = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc="Filter of the measurement. "
    )

    flux_psf = sa.Column(
        sa.Float,
        nullable=False,
        doc="PSF flux of the measurement. "
    )

    flux_psf_err = sa.Column(
        sa.Float,
        nullable=False,
        doc="PSF flux error of the measurement. "
    )

    flux_apertures = sa.Column(
        sa.ARRAY(sa.Float),
        nullable=False,
        doc="Aperture fluxes of the measurement. "
    )

    flux_apertures_err = sa.Column(
        sa.ARRAY(sa.Float),
        nullable=False,
        doc="Aperture flux errors of the measurement. "
    )

    aper_radii = sa.Column(
        sa.ARRAY(sa.Float),
        nullable=False,
        doc="Radii of the apertures used for calculating flux, in pixels. "
    )

    best_aperture = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=-1,
        doc="The index of the aperture that was chosen as the best aperture for this measurement. "
            "Set to -1 to select the PSF flux instead of one of the apertures. "
    )

    mag_psf = sa.Column(
        sa.Float,
        nullable=False,
        index=True,
        doc="PSF magnitude of the measurement. "
    )

    mag_psf_err = sa.Column(
        sa.Float,
        nullable=False,
        doc="PSF magnitude error of the measurement. "
    )

    mag_apertures = sa.Column(
        sa.ARRAY(sa.Float),
        nullable=False,
        doc="Aperture magnitudes of the measurement. "
    )

    mag_apertures_err = sa.Column(
        sa.ARRAY(sa.Float),
        nullable=False,
        doc="Aperture magnitude errors of the measurement. "
    )

    magnitude = sa.Column(
        sa.Float,
        nullable=False,
        index=True,
        doc="Magnitude of the measurement, "
            "defined as the magnitude of the best aperture (or PSF if best_aperture=-1). "
    )

    # the error on the magnitude is not indexed/searchable so we can use a simple property
    @property
    def magnitude_err(self):
        if self.best_aperture == -1:
            return self.mag_psf_err
        return self.mag_apertures_err[self.best_aperture]

    limmag = sa.Column(
        sa.Float,
        nullable=False,
        index=True,
        doc="Limiting magnitude of the measurement. Useful in case of non-detections. "
    )

    background = sa.Column(
        sa.Float,
        nullable=False,
        doc="Background of the measurement, from a local annulus. Given as counts per pixel. "
    )

    background_err = sa.Column(
        sa.Float,
        nullable=False,
        doc="RMS error of the background measurement, from a local annulus. Given as counts per pixel. "
    )

    area_psf = sa.Column(
        sa.Float,
        nullable=False,
        doc="Area of the PSF used for calculating flux. Remove a * background from the flux measurement. "
    )

    area_apertures = sa.Column(
        sa.ARRAY(sa.Float),
        nullable=False,
        doc="Areas of the apertures used for calculating flux. Remove a * background from the flux measurement. "
    )

    offset_x = sa.Column(
        sa.Float,
        nullable=False,
        doc="Offset in x from the center of the cutout. "
    )

    offset_y = sa.Column(
        sa.Float,
        nullable=False,
        doc="Offset in y from the center of the cutout. "
    )

    width = sa.Column(
        sa.Integer,
        nullable=False,
        index=True,
        doc="Width of the source in the cutout. "
            "Given by the average of the 2nd moments of the distribution of counts in the aperture. "
    )

    elongation = sa.Column(
        sa.Float,
        nullable=False,
        doc="Elongation of the source in the cutout. "
            "Given by the ratio of the 2nd moments of the distribution of counts in the aperture. "
            "Values close to 1 indicate a round source, while values close to 0 indicate an elongated source. "
    )

    position_angle = sa.Column(
        sa.Float,
        nullable=False,
        doc="Position angle of the source in the cutout. "
            "Given by the angle of the major axis of the distribution of counts in the aperture. "
    )

    badness_scores = sa.Column(
        JSONB,
        nullable=False,
        index=True,
        doc="Badness scores of the measurement. "
            "This includes analytical cuts and external scores like real-bogus. "
            "The higher the score, the more likely the measurement is to be an artefact. "
            "Compare (using >=) to the built-in thresholds in the provenance parameters "
            "to decide if the measurement passes or not. "
    )

    passing = sa.Column(
        sa.Boolean,
        nullable=False,
        index=True,
        doc="Whether the measurement passes the badness cuts. "
    )

    def __init__(self, cutouts, **kwargs):
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values
        self.cutouts = cutouts
        self.cutouts_id = cutouts.id

        # these are loaded from the cutouts and upstream objects and saved to DB for easy searches
        self.mjd = cutouts.sources.image.mjd
        self.exptime = cutouts.sources.image.exptime
        self.filter = cutouts.sources.image.filter
        self.limmag = cutouts.sources.image.lim_mag_estimate  # TODO: must update this when doing Issue #143
        self.ra = cutouts.source_row['ra']
        self.dec = cutouts.source_row['dec']

        # these are not saved in the DB but are useful to have around
        self.zp = cutouts.sources.image.zp.zp
        self.dzp = cutouts.sources.image.zp.dzp
        self.fwhm_pixels = cutouts.sources.image.get_psf().fwhm_pixels
        self.psf = cutouts.sources.image.get_psf().get_clip(x=cutouts.x, y=cutouts.y)
        self.pixel_scale = cutouts.sources.image.wcs.get_pixel_scale()

        # TODO: continue this

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        __table_args__ = (
            sa.Index(
                "ix_measurements_scores_gin",
                "badness_scores",
                postgresql_using="gin",
            ),
        )

    def __repr__(self):
        return (
            f"<Measurements {self.id} "
            f"from SourceList {self.sources_id} "
            f"(number {self.index_in_sources}) "
            f"from Image {self.sub_image_id} "
            f"at x,y= {self.x}, {self.y}>"
        )

    @orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        self.zp = self.cutouts.sources.image.zp.zp
        self.dzp = self.cutouts.sources.image.zp.dzp
        self.fwhm_pixels = self.cutouts.sources.image.psf.fwhm_pixels
        self.psf = self.cutouts.sources.image.get_psf().get_clip(x=self.cutouts.x, y=self.cutouts.y)
        self.pixel_scale = self.cutouts.sources.image.wcs.get_pixel_scale()

    def set_apertures(self, radii, units='pixels'):
        """Set the apertures to be used for calculating flux and magnitude.

        Parameters
        ----------
        radii : float or array-like
            The radii of the apertures to be used.
        units : str
            The units of the radii. Can be 'pixels', 'arcsec', or 'fwhm'.
            - pixels: the radii are already in pixels.
            - arcsec: the radii are in arcseconds, and will be converted to pixels using the pixel scale.
            - fwhm: the radii are in units of the FWHM of the PSF, and will be converted to pixels using the FWHM.
        """
        if isinstance(radii, (int, float)):
            radii = [radii]

        if units == 'pixels':
            self.aper_radii = radii
        elif units == 'arcsec':
            self.aper_radii = radii / self.pixel_scale
        elif units == 'fwhm':
            self.aper_radii = radii * self.fwhm_pixels
        else:
            raise ValueError(f'Unknown units: {units}. Use "pixels", "arcsec", or "fwhm".')

    def calculate_magnitudes(self):
        """Fill out the magnitudes using the zero point and fluxes. """
        pass





