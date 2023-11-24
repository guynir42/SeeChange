import os
import math
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.schema import CheckConstraint
from sqlalchemy.ext.associationproxy import association_proxy

from astropy.time import Time
from astropy.wcs import WCS
from astropy.io import fits
import astropy.coordinates
import astropy.units as u

from pipeline.utils import read_fits_image, save_fits_image_file

from models.base import (
    Base,
    SeeChangeBase,
    SmartSession,
    AutoIDMixin,
    FileOnDiskMixin,
    SpatiallyIndexed,
    FourCorners,
    HasBitFlagBadness,
    _logger
)
from models.exposure import Exposure
from models.instrument import get_instrument_instance
from models.enums_and_bitflags import (
    ImageFormatConverter,
    ImageTypeConverter,
    image_badness_inverse,
)

import util.config as config


image_products_association_table = sa.Table(
    'image_products_association',
    Base.metadata,
    sa.Column('prod_id',
              sa.Integer,
              sa.ForeignKey('image_products.id', ondelete="CASCADE", name='image_products_association_prod_id_fkey'),
              primary_key=True),
    sa.Column('image_id',
              sa.Integer,
              sa.ForeignKey('images.id', ondelete="CASCADE", name='image_products_association_image_id_fkey'),
              primary_key=True),
)


class ImageProducts(Base, AutoIDMixin):

    __tablename__ = 'image_products'

    image_id = sa.Column(
        sa.ForeignKey('images.id', ondelete='CASCADE', name='image_products_image_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the image that these products are associated with. "
    )

    image = orm.relationship(
        'Image',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc="The image that these products are associated with. "
    )

    psf_id = sa.Column(
        sa.ForeignKey('psfs.id', name='image_products_psf_id_fkey'),
        nullable=True,
        index=True,
        doc="ID of the PSF object associated with this image. "
    )

    psf = orm.relationship(
        'PSF',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc="The PSF object associated with this image. "
    )

    sources_id = sa.Column(
        sa.ForeignKey('source_lists.id', name='image_products_source_list_id_fkey'),
        nullable=True,
        index=True,
        doc="ID of the source list associated with this image. "
    )

    sources = orm.relationship(
        'SourceList',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc="The source list associated with this image. "
    )

    wcs_id = sa.Column(
        sa.ForeignKey('world_coordinates.id', name='image_products_wcs_id_fkey'),
        nullable=True,
        index=True,
        doc="ID of the WCS object associated with this image. "
    )

    wcs = orm.relationship(
        'WorldCoordinates',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc="The WCS object associated with this image. "
    )

    zp_id = sa.Column(
        sa.ForeignKey('zero_points.id', name='image_products_zp_id_fkey'),
        nullable=True,
        index=True,
        doc="ID of the zero point object associated with this image. "
    )

    # might need this to sort the loaded objects
    mjd = sa.Column(
        sa.Float,
        nullable=False,
        index=True,
        doc="Modified Julian date of the image. "
    )

    def remove_data_from_disk(self, remove_folders=True):
        """Call remove_data_from_disk on all the associated objects that are FileOnDiskMixin objects."""
        if self.image is not None:
            self.image.remove_data_from_disk(remove_folders=remove_folders)
        if self.psf is not None:
            self.psf.remove_data_from_disk(remove_folders=remove_folders)
        if self.sources is not None:
            self.sources.remove_data_from_disk(remove_folders=remove_folders)

    def delete_from_disk_and_database(self, session=None, commit=True, remove_folders=True):
        """Call delete_from_disk_and_database on all the associated objects that are FileOnDiskMixin objects.
        Objects that are not FileOnDiskMixin objects are just deleted from the database.
        """
        if session is None and not commit:
            raise RuntimeError("When session=None, commit must be True!")

        with SmartSession(session) as session:
            if self.image is not None:
                self.image.delete_from_disk_and_database(session=session, commit=False, remove_folders=remove_folders)
            if self.psf is not None:
                self.psf.delete_from_disk_and_database(session=session, commit=False, remove_folders=remove_folders)
            if self.sources is not None:
                self.sources.delete_from_disk_and_database(session=session, commit=False, remove_folders=remove_folders)
            if self.wcs is not None and self.wcs.id is not None:
                session.delete(self.wcs)
            if self.zp is not None and self.zp.id is not None:
                session.delete(self.zp)
            session.delete(self)
            if commit:
                session.commit()

    def __setattr__(self, key, value):
        if key == 'image':
            if value is not None:
                self.mjd = value.mjd
            else:
                self.mjd = None

        object.__setattr__(self, key, value)


class Image(Base, AutoIDMixin, FileOnDiskMixin, SpatiallyIndexed, FourCorners, HasBitFlagBadness):

    __tablename__ = 'images'

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=ImageFormatConverter.convert('fits'),
        doc="Format of the file on disk. Should be fits or hdf5. "
    )

    @hybrid_property
    def format(self):
        return ImageFormatConverter.convert(self._format)

    @format.inplace.expression
    @classmethod
    def format(cls):
        # ref: https://stackoverflow.com/a/25272425
        return sa.case(ImageFormatConverter.dict, value=cls._format)

    @format.inplace.setter
    def format(self, value):
        self._format = ImageFormatConverter.convert(value)

    exposure_id = sa.Column(
        sa.ForeignKey('exposures.id', ondelete='SET NULL', name='images_exposure_id_fkey'),
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

    upstream_products = orm.relationship(
        'ImageProducts',
        secondary=image_products_association_table,
        cascade='save-update, merge, refresh-expire, expunge',
        passive_deletes=True,
        lazy='selectin',
        order_by='ImageProducts.mjd',
        doc=(
            "ImageProduct for all images that were used to make this image. "
        )
    )

    ref_products_id = sa.Column(
        sa.Integer,
        nullable=True,
        index=True,
        doc=(
            "ID of the reference image products used to produce a this image. "
            "This is the image that other images are warped and scaled to match. "
        )
    )

    new_products_id = sa.Column(
        sa.Integer,
        nullable=True,
        index=True,
        doc=(
            "ID of the new entry used to produce a difference image. "
            "When making a warped or difference image, this will refer to the new (science) image. "
            "For coadded images this will be None. "
        )
    )

    is_sub = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        index=True,
        doc='Is this a subtraction image.'
    )

    is_coadd = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        index=True,
        doc='Is this image made by stacking multiple images.'
    )

    is_warped = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        index=True,
        doc='Has this image been warped to align with a reference.'
    )

    _type = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=ImageTypeConverter.convert('Sci'),
        index=True,
        doc=(
            "Type of image. One of: [Sci, Diff, Bias, Dark, DomeFlat, SkyFlat, TwiFlat, Warped] "
            "or any of the above types prepended with 'Com' for combined "
            "(e.g., a ComSci image is a science image combined from multiple exposures)."
            "Saved as an integer in the database, but converted to a string when read. "
        )
    )

    @hybrid_property
    def type(self):
        return ImageTypeConverter.convert(self._type)

    @type.inplace.expression
    @classmethod
    def type(cls):
        return sa.case(ImageTypeConverter.dict, value=cls._type)

    @type.inplace.setter
    def type(self, value):
        self._type = ImageTypeConverter.convert(value)

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='images_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this image. "
            "The provenance will contain a record of the code version "
            "and the parameters used to produce this image. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this image. "
            "The provenance will contain a record of the code version "
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
    def observation_time(self):
        """Translation of the MJD column to datetime object."""
        if self.mjd is None:
            return None
        else:
            return Time(self.mjd, format='mjd').datetime

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

    filter = sa.Column(sa.Text, nullable=True, index=True, doc="Name of the filter used to make this image. ")

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

    preproc_bitflag = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=0,
        index=False,
        doc='Bitflag specifying which preprocessing steps have been completed for the image.'
    )

    astro_cal_done = sa.Column(
        sa.BOOLEAN,
        nullable=False,
        default=False,
        index=False,
        doc=( 'Has a WCS been solved for this image.  This should be set to true after astro_cal '
              'has been run, or for images (like subtractions) that are derived from other images '
              'with complete WCSes that can be copied.  This does not promise that the "latest and '
              'greatest" astrometric calibration is what is in the image header, only that there is '
              'one from the pipeline that should be good for visual identification of positions.' )
    )

    sky_sub_done = sa.Column(
        sa.BOOLEAN,
        nullable=False,
        default=False,
        index=False,
        doc='Has the sky been subtracted from this image. '
    )

    fwhm_estimate = sa.Column(
        sa.Float,
        nullable=True,
        index=True,
        doc=(
            'FWHM estimate for the image, in arcseconds, '
            'from the first time the image was processed.'
        )
    )

    zero_point_estimate = sa.Column(
        sa.Float,
        nullable=True,
        index=True,
        doc=(
            'Zero point estimate for the image, '
            'from the first time the image was processed.'
        )
    )

    lim_mag_estimate = sa.Column(
        sa.Float,
        nullable=True,
        index=True,
        doc=(
            'Limiting magnitude estimate for the image, '
            'from the first time the image was processed.'
        )
    )

    bkg_mean_estimate = sa.Column(
        sa.Float,
        nullable=True,
        index=True,
        doc=(
            'Background estimate for the image, '
            'from the first time the image was processed.'
        )
    )

    bkg_rms_estimate = sa.Column(
        sa.Float,
        nullable=True,
        index=True,
        doc=(
            'Background RMS estimate for the image, '
            'from the first time the image was processed.'
        )
    )

    __table_args__ = (
        CheckConstraint(
            sqltext='NOT(md5sum IS NULL AND md5sum_extensions IS NULL)',
            name='md5sum_or_md5sum_extensions_check'
        ),
    )

    def _get_inverse_badness(self):
        """Get a dict with the allowed values of badness that can be assigned to this object"""
        return image_badness_inverse

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
        self.sources = None  # the sources extracted from this Image (optionally loaded)
        self.psf = None  # the point-spread-function object (optionally loaded)
        self.wcs = None  # the WorldCoordinates object (optionally loaded)
        self.zp = None  # the zero-point object (optionally loaded)

        self._instrument_object = None
        self._bitflag = 0

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.ra is not None and self.dec is not None:
            self.calculate_coordinates()  # galactic and ecliptic coordinates

    @orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        FileOnDiskMixin.init_on_load(self)
        self.raw_data = None
        self._raw_header = None
        self._data = None
        self._flags = None
        self._weight = None
        self._background = None
        self._score = None
        self.sources = None
        self.psf = None
        self.wcs = None
        self.zp = None

        self._instrument_object = None

    def set_corners_from_header_wcs( self ):
        wcs = WCS( self._raw_header )
        ras = []
        decs = []
        data = self.raw_data if self.raw_data is not None else self.data
        width = data.shape[1]
        height = data.shape[0]
        xs = [ 0., width-1., 0., width-1. ]
        ys = [ 0., height-1., height-1., 0. ]
        scs = wcs.pixel_to_world( xs, ys )
        if isinstance( scs[0].ra, astropy.coordinates.Longitude ):
            ras = [ i.ra.to_value() for i in scs ]
            decs = [ i.dec.to_value() for i in scs ]
        else:
            ras = [ i.ra.value_in(u.deg).value for i in scs ]
            decs = [ i.dec.value_in(u.deg).value for i in scs ]
        ras, decs = FourCorners.sort_radec( ras, decs )
        self.ra_corner_00 = ras[0]
        self.ra_corner_01 = ras[1]
        self.ra_corner_10 = ras[2]
        self.ra_corner_11 = ras[3]
        self.dec_corner_00 = decs[0]
        self.dec_corner_01 = decs[1]
        self.dec_corner_10 = decs[2]
        self.dec_corner_11 = decs[3]

    def make_image_products(self):
        """Generate an ImageProducts object from this image and any loaded PSF, SourceList, WorldCoordinates, etc."""
        prod = ImageProducts()
        prod.image = self
        prod.psf = self.psf
        prod.sources = self.sources
        prod.wcs = self.wcs
        prod.zp = self.zp

        return prod

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
            'target',
            'format',
        ]

        # copy all the columns that are the same
        for column in same_columns:
            setattr(new, column, getattr(exposure, column))

        if exposure.filter_array is not None:
            idx = exposure.instrument_object.get_section_filter_array_index(section_id)
            new.filter = exposure.filter_array[idx]

        new.section_id = section_id
        new.raw_data = exposure.data[section_id]
        new.instrument_object = exposure.instrument_object

        # read the header from the exposure file's individual section data
        new._raw_header = exposure.section_headers[section_id]

        # Because we will later be writing out float data (BITPIX=-32)
        # -- or whatever the type of raw_data is -- we have to make sure
        # there aren't any vestigal BSCALE and BZERO keywords in the
        # header.
        for delkw in [ 'BSCALE', 'BZERO' ]:
            if delkw in new.raw_header:
                del new.raw_header[delkw]

        # numpy array axis ordering is backwards from FITS ordering
        width = new.raw_data.shape[1]
        height = new.raw_data.shape[0]

        names = ['ra', 'dec'] + new.instrument_object.get_auxiliary_exposure_header_keys()
        header_info = new.instrument_object.extract_header_info(new._raw_header, names)
        # TODO: get the important keywords translated into the searchable header column

        # figure out the RA/Dec of each image

        # first see if this instrument has a special method of figuring out the RA/Dec
        new.ra, new.dec = new.instrument_object.get_ra_dec_for_section(exposure, section_id)

        # if not (which is true for most instruments!), try to read the WCS from the header:
        try:
            wcs = WCS(new._raw_header)
            sc = wcs.pixel_to_world(width // 2, height // 2)
            new.ra = sc.ra.to(u.deg).value
            new.dec = sc.dec.to(u.deg).value
        except:
            pass  # can't do it, just leave RA/Dec as None

        # if that fails, try to get the RA/Dec from the section header
        if new.ra is None or new.dec is None:
            new.ra = header_info.pop('ra', None)
            new.dec = header_info.pop('dec', None)

        # if that fails, just use the RA/Dec of the global exposure
        if new.ra is None or new.dec is None:
            new.ra = exposure.ra
            new.dec = exposure.dec

        new.header = header_info  # save any additional header keys into a JSONB column

        # Figure out the 4 corners  Start by trying to use the WCS
        gotcorners = False
        try:
            new.set_corners_from_header_wcs()
            _logger.debug( 'Got corners from WCS' )
            gotcorners = True
        except:
            pass

        # If that didn't work, then use ra and dec and the instrument scale
        # TODO : take into account standard instrument orientation!
        # (Can be done after the decam_pull PR is merged)

        if not gotcorners:
            halfwid = new.instrument_object.pixel_scale * width / 2. / math.cos( new.dec * math.pi / 180. ) / 3600.
            halfhei = new.instrument_object.pixel_scale * height / 2. / 3600.
            ra0 = new.ra - halfwid
            ra1 = new.ra + halfwid
            dec0 = new.dec - halfhei
            dec1 = new.dec + halfhei
            new.ra_corner_00 = ra0
            new.ra_corner_01 = ra0
            new.ra_corner_10 = ra1
            new.ra_corner_11 = ra1
            new.dec_corner_00 = dec0
            new.dec_corner_01 = dec1
            new.dec_corner_10 = dec0
            new.dec_corner_11 = dec1
            gotcorners = True

        # the exposure_id will be set automatically at commit time
        # ...but we have to set it right now because other things are
        # going to check to see if exposure.id matches image.exposure.id
        new.exposure_id = exposure.id
        new.exposure = exposure

        return new

    @classmethod
    def from_images(cls, images):
        """
        Create a new Image object from a list of other Image objects.
        This is the first step in making a multi-image (usually a coadd).
        The output image doesn't have any data, and is created with
        nofile=True. It is up to the calling application to fill in the
        data, flags, weight, etc. using the appropriate preprocessing tools.
        After that, the data needs to be saved to file, and only then
        can the new Image be added to the database.

        Parameters
        ----------
        images: list of Image objects
            The images to combine into a new Image object.

        Returns
        -------
        output: Image
            The new Image object. It would not have any data variables or filepath.
        """
        if len(images) < 1:
            raise ValueError("Must provide at least one image to combine.")

        output = Image(nofile=True)

        # for each attribute, check that all the images have the same value
        for att in ['section_id', 'instrument', 'telescope', 'type', 'filter', 'project', 'target']:
            values = set([getattr(image, att) for image in images])
            if len(values) != 1:
                raise ValueError(f"Cannot combine images with different {att} values: {values}")
            output.__setattr__(att, values.pop())
        # TODO: should RA and Dec also be exactly the same??
        output.ra = images[0].ra
        output.dec = images[0].dec
        output.ra_corner_00 = images[0].ra_corner_00
        output.ra_corner_01 = images[0].ra_corner_01
        output.ra_corner_10 = images[0].ra_corner_10
        output.ra_corner_11 = images[0].ra_corner_11
        output.dec_corner_00 = images[0].dec_corner_00
        output.dec_corner_01 = images[0].dec_corner_01
        output.dec_corner_10 = images[0].dec_corner_10
        output.dec_corner_11 = images[0].dec_corner_11

        # exposure time is usually added together
        output.exp_time = sum([image.exp_time for image in images])

        # start MJD and end MJD
        output.mjd = min([image.mjd for image in images])
        output.end_mjd = max([image.end_mjd for image in images])

        # TODO: what about the header? should we combine them somehow?
        output.header = images[0].header
        output.raw_header = images[0].raw_header

        base_type = images[0].type
        if not base_type.startswith('Com'):
            output.type = 'Com' + base_type

        # make sure to make the image product objects
        for im in images:
            prod = im.make_image_products()
            output.upstream_products.append(prod)

        # mark the first image as the reference (this can be overriden later when actually warping into one image)
        output.ref_products_id = output.upstream_products[0].id

        # Note that "data" is not filled by this method, also the provenance is empty!
        return output

    @classmethod
    def from_new_and_ref(cls, new_image, ref_image, im_type=None):
        """
        Create a new Image object from a reference Image object and a new Image object.
        This is the first step in making a difference image.
        The output image doesn't have any data, and is created with
        nofile=True. It is up to the calling application to fill in the
        data, flags, weight, etc. using the appropriate preprocessing tools.
        After that, the data needs to be saved to file, and only then
        can the new Image be added to the database.

        Parameters
        ----------
        new_image: Image object
            The new image to use.
        ref_image: Image object
            The reference image to use.
        im_type: str (optional)
            If given, will also set the output image's type.
            Use "diff" or "warped", which will be prepended
            with "Com" if the new image is a "Com..." type image
            Note that if not given, the output type will remain as "Sci".
        Returns
        -------
        output: Image
            The new Image object. It would not have any data variables or filepath.
        """
        output = Image(nofile=True)

        # for each attribute, check the two images have the same value
        for att in ['instrument', 'telescope', 'project', 'section_id', 'filter', 'target']:
            ref_value = getattr(ref_image, att)
            new_value = getattr(new_image, att)

            if att == 'section_id':
                ref_value = str(ref_value)
                new_value = str(new_value)

            if att in ['section_id', 'filter', 'target'] and ref_value != new_value:
                raise ValueError(
                    f"Cannot combine images with different {att} values: "
                    f"{ref_value} and {new_value}. "
                )

            # assign the values from the new image
            output.__setattr__(att, new_value)
        # TODO: should RA and Dec also be exactly the same??

        output.upstream_products = [ref_image.make_image_products(), new_image.make_image_products()]
        output.ref_products_id = output.upstream_products[0].id
        output.new_products_id = output.upstream_products[1].id

        # get some more attributes from the new image
        for att in ['exp_time', 'mjd', 'end_mjd', 'header', 'raw_header', 'ra', 'dec',
                    'ra_corner_00', 'ra_corner_01', 'ra_corner_10', 'ra_corner_11',
                    'dec_corner_00', 'dec_corner_01', 'dec_corner_10', 'dec_corner_11' ]:
            output.__setattr__(att, getattr(new_image, att))

        if im_type.lower == 'diff':
            output.type = 'Diff'
        elif im_type.lower == 'warped':
            output.type = 'Warped'
        else:
            raise ValueError(f"im_type must be 'diff' or 'warped'. Got {im_type} instead.")
        if new_image.type.startswith('Com'):
            output.type = 'ComDiff'

        # Note that "data" is not filled by this method, also the provenance is empty!
        return output

    @property
    def instrument_object(self):
        if self.instrument is not None:
            if self._instrument_object is None or self._instrument_object.name != self.instrument:
                self._instrument_object = get_instrument_instance(self.instrument)

        return self._instrument_object

    @instrument_object.setter
    def instrument_object(self, value):
        self._instrument_object = value

    @property
    def filter_short(self):
        if self.filter is None:
            return None
        return self.instrument_object.get_short_filter_name(self.filter)

    def __repr__(self):

        output = (
            f"Image(id: {self.id}, "
            f"type: {self.type}, "
            f"exp: {self.exp_time}s, "
            f"filt: {self.filter_short}, "
            f"from: {self.instrument}/{self.telescope}"
        )

        output += ")"

        return output

    def __str__(self):
        return self.__repr__()

    def invent_filepath(self):
        """Create a relative file path for the object.

        Create a file path relative to data root for the object based on its
        metadata.  This is used when saving the image to disk.  Data
        products that depend on an image and are also saved to disk
        (e.g., SourceList) will just append another string to the Image
        filename.

        """
        prov_hash = inst_name = im_type = date = time = filter = ra = dec = dec_int_pm = ''
        section_id = section_id_int = ra_int = ra_int_h = ra_frac = dec_int = dec_frac = 0

        if self.provenance is not None:
            prov_hash = self.provenance.id
        if self.instrument_object is not None:
            inst_name = self.instrument_object.get_short_instrument_name()
        if self.type is not None:
            im_type = self.type

        if self.mjd is not None:
            t = Time(self.mjd, format='mjd', scale='utc').datetime
            date = t.strftime('%Y%m%d')
            time = t.strftime('%H%M%S')

        if self.filter_short is not None:
            filter = self.filter_short

        if self.section_id is not None:
            section_id = str(self.section_id)
            try:
                section_id_int = int(self.section_id)
            except ValueError:
                section_id_int = 0

        if self.ra is not None:
            ra = self.ra
            ra_int, ra_frac = str(float(ra)).split('.')
            ra_int = int(ra_int)
            ra_int_h = ra_int // 15
            ra_frac = int(ra_frac)

        if self.dec is not None:
            dec = self.dec
            dec_int, dec_frac = str(float(dec)).split('.')
            dec_int = int(dec_int)
            dec_int_pm = f'p{dec_int:02d}' if dec_int >= 0 else f'm{dec_int:02d}'
            dec_frac = int(dec_frac)

        cfg = config.Config.get()
        default_convention = "{inst_name}_{date}_{time}_{section_id}_{filter}_{im_type}_{prov_hash:.6s}"
        name_convention = cfg.value('storage.images.name_convention', default=None)
        if name_convention is None:
            name_convention = default_convention

        filename = name_convention.format(
            inst_name=inst_name,
            im_type=im_type,
            date=date,
            time=time,
            filter=filter,
            ra=ra,
            ra_int=ra_int,
            ra_int_h=ra_int_h,
            ra_frac=ra_frac,
            dec=dec,
            dec_int=dec_int,
            dec_int_pm=dec_int_pm,
            dec_frac=dec_frac,
            section_id=section_id,
            section_id_int=section_id_int,
            prov_hash=prov_hash,
        )

        # TODO: which elements of the naming convention are really necessary?
        #  and what is a good way to make sure the filename actually depends on them?
        return filename

    def save(self, filename=None, only_image=False, just_update_header=True, **kwargs ):
        """Save the data (along with flags, weights, etc.) to disk.
        The format to save is determined by the config file.
        Use the filename to override the default naming convention.

        Will save the standard image extensions : image, weight, mask.
        Does not save the source list or psf or other things that have
        their own objects; those need to be saved separately.  (Also see
        pipeline.datastore.)

        Parameters
        ----------
        filename: str (optional)
            The filename to use to save the data.  If not provided, will
            use what is in self.filepath; if that is None, then the
            default naming convention willbe used.  self.filepath will
            be updated to this name

        only_image: bool, default False
            If the image is stored as multiple files (i.e. image,
            weight, and flags extensions are all stored as seperate
            files, rather than as HDUs within one file), then _only_
            write the image out.  The use case for this is for
            "first-look" headers with astrometric and photometric
            solutions; the image header gets updated in that case, but
            the weight and flags files stay the same, so they do not
            need to be updated.  You will usually want just_update_header
            to be True when only_image is True.

        just_update_header: bool, default True
            Ignored unless only_image is True and the image is stored as
            multiple files rather than as FITS extensions.  In this
            case, if just_udpate_header is True and the file already
            exists, don't write the data, just update the header.

        **kwargs: passed on to FileOnDiskMixin.save(), include:
            overwrite - bool, set to True if it's OK to overwrite exsiting files
            no_archive - bool, set to True to save only to local disk, otherwise also saves to the archive
            exists_ok, verify_md5 - complicated, see documentation on FileOnDiskMixin

        For images being saved to the database, you probably want to use
        overwrite=True, verify_md5=True, or perhaps overwrite=False,
        exists_ok=True, verify_md5=True.  For temporary images being
        saved as part of local processing, you probably want to use
        verify_md5=False and either overwrite=True (if you're modifying
        and writing the file multiple times), or overwrite=False,
        exists_ok=True (if you might call the save() method more than
        once on the same image, and you want to trust the filesystem to
        have saved it right).

        """
        if self.data is None:
            raise RuntimeError("The image data is not loaded. Cannot save.")

        if self.provenance is None:
            raise RuntimeError("The image provenance is not set. Cannot save.")

        if filename is not None:
            self.filepath = filename
        if self.filepath is None:
            self.filepath = self.invent_filepath()

        cfg = config.Config.get()
        single_file = cfg.value('storage.images.single_file', default=False)
        format = cfg.value('storage.images.format', default='fits')
        extensions = []
        files_written = {}

        if not only_image:
            # In order to ignore just_update_header if only_image is false,
            # we need to pass it as False on to save_fits_image_file
            just_update_header = False

        full_path = os.path.join(self.local_path, self.filepath)

        if format == 'fits':
            # save the imaging data
            extensions.append('.image.fits')
            imgpath = save_fits_image_file(full_path, self.data, self.raw_header,
                                           extname='image', single_file=single_file,
                                           just_update_header=just_update_header)
            files_written['.image.fits'] = imgpath
            # TODO: we can have extensions at the end of the self.filepath (e.g., foo.fits.flags)
            #  or we can have the extension name carry the file extension (e.g., foo.flags.fits)
            #  this should be configurable and will affect how we make the self.filepath and extensions.

            # save the other extensions
            array_list = ['flags', 'weight', 'background', 'score']
            # TODO: the list of extensions should be saved somewhere more central

            if single_file or ( not only_image ):
                for array_name in array_list:
                    array = getattr(self, array_name)
                    if array is not None:
                        extpath = save_fits_image_file(
                            full_path,
                            array,
                            self.raw_header,
                            extname=array_name,
                            single_file=single_file
                        )
                        array_name = '.' + array_name
                        if not array_name.endswith('.fits'):
                            array_name += '.fits'
                        extensions.append(array_name)
                        if not single_file:
                            files_written[array_name] = extpath

            if single_file:
                files_written = files_written['.image.fits']
                if not self.filepath.endswith('.fits'):
                    self.filepath += '.fits'

        elif format == 'hdf5':
            # TODO: consider writing a more generic utility to save_image_file that handles either fits or hdf5, etc.
            raise NotImplementedError("HDF5 format is not yet supported.")
        else:
            raise ValueError(f"Unknown image format: {format}. Use 'fits' or 'hdf5'.")

        # Save the file to the archive and update the database record
        # (as well as self.filepath, self.filepath_extensions, self.md5sum, self.md5sum_extensions)
        # (From what we did above, it's already in the right place in the local filestore.)
        if single_file:
            FileOnDiskMixin.save( self, files_written, **kwargs )
        else:
            if just_update_header:
                FileOnDiskMixin.save( self, files_written['.image.fits'], '.image.fits', **kwargs )
            else:
                for ext in extensions:
                    FileOnDiskMixin.save( self, files_written[ext], ext, **kwargs )

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
            self._data, self._raw_header = read_fits_image(filename, ext='image', output='both')
            self._flags = read_fits_image(filename, ext='flags')
            self._weight = read_fits_image(filename, ext='weight')
            self._background = read_fits_image(filename, ext='background')
            self._score = read_fits_image(filename, ext='score')
            # TODO: add more if needed!

        else:  # save each data array to a separate file
            if self.filepath_extensions is None:
                self._data, self._raw_header = read_fits_image( self.get_fullpath(), output='both' )
            else:
                gotim = False
                gotweight = False
                gotflags = False
                for extension, filename in zip( self.filepath_extensions, self.get_fullpath(as_list=True) ):
                    if not os.path.isfile(filename):
                        raise FileNotFoundError(f"Could not find the image file: {filename}")
                    if extension == '.image.fits':
                        self._data, self._raw_header = read_fits_image(filename, output='both')
                        gotim = True
                    elif extension == '.weight.fits':
                        self._weight = read_fits_image(filename, output='data')
                        gotweight = True
                    elif extension == '.flags.fits':
                        self._flags = read_fits_image(filename, output='data')
                        gotflags = True
                    else:
                        raise ValueError( f'Unknown image extension {extension}' )
                if not ( gotim and gotweight and gotflags ):
                    raise FileNotFoundError( "Failed to load at least one of image, weight, flags" )

    def get_upstreams(self, session=None):
        """
        Get the upstream images that were used to make this image.
        This includes the reference image, new image, and source images.

        Parameters
        ----------
        session: SQLAlchemy session (optional)
            The session to use to query the database. If not provided,
            will open a new session that automatically closes at
            the end of the function.

        Returns
        -------
        upstreams: list of Image objects
            The upstream images.
        """
        with SmartSession(session) as session:
            upstreams = []
            # get the exposure
            if self.exposure_id is not None:
                exposure = session.scalars(sa.select(Exposure).where(Exposure.id == self.exposure_id)).first()
                if exposure is not None:
                    upstreams.append(exposure)

            # get the reference entry
            for prod in self.upstream_products:
                upstreams.append(prod.image)
                upstreams.append(prod.psf)
                upstreams.append(prod.sources)
                upstreams.append(prod.wcs)
                upstreams.append(prod.zp)

        return upstreams

    def get_downstreams(self, session=None):
        """Get all the objects that were created based on this image. """
        # avoids circular import
        from models.source_list import SourceList
        from models.psf import PSF

        downstreams = []
        with SmartSession(session) as session:
            # source lists:
            sources = session.scalars(
                sa.select(SourceList).where(SourceList.image_id == self.id)
            ).all()
            downstreams += sources

            # psfs:
            psfs = session.scalars(
                sa.select(PSF).where(PSF.image_id == self.id)
            ).all()
            downstreams += psfs

            # now look for other images that were created based on this one
            images = session.scalars(
                sa.select(ImageProducts).where(ImageProducts.image_id == self.id).select_from(
                    ImageProducts.join(Image, Image.id == ImageProducts.image_id)
                )
            ).all()

            downstreams += images

            return downstreams

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
        if self._raw_header is None:
            self._raw_header = fits.Header()
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


if __name__ == '__main__':
    filename = '/home/guyn/Dropbox/python/SeeChange/data/DECam_examples/c4d_221104_074232_ori.fits.fz'
    e = Exposure(filename)
    im = Image.from_exposure(e, section_id=1)

