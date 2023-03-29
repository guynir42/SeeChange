import sqlalchemy as sa

from pipeline.parameters import Parameters

from models.base import SmartSession
from models.image import Image

from pipeline.utils import get_image_cache


class ParsAstrometry(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.cross_match_catalog = self.add_par(
            'cross_match_catalog',
            'Gaia',
            str,
            'Which catalog should be used for cross matching for astrometry. '
        )
        self.add_alias('catalog', 'cross_match_catalog')

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def _get_process_name(self):
        return 'astrometry'

    def _get_upstream_process_names(self):
        return ['preprocessor']


class Astrometry:
    def __init__(self, **kwargs):
        self.pars = ParsAstrometry()
        self.pars.update(kwargs)

    def run(self, exposure_id, ccd_id, image_cache=None, prov_cache=None, session=None):
        """
        Extract sources and use their positions to calculate the astrometric solution.

        Parameters
        ----------
        exposure_id : int
            The exposure ID in the database.
        ccd_id : int
            The CCD number in the camera's numbering scheme.
        image_cache : dict
            A dictionary of images that have already been loaded.
            Includes a dictionary of 'exposures' keyed by exposure_id.
            Also includes dictionaries for 'images', 'subtractions',
            'references' and 'cutouts'.
            Each one is keyed on a tuple of (exposure_id, ccd_id).
            If the required image is not found in the cache, it will be loaded
            from the database and then from disk.
        prov_cache: dict
            A cache of provenance objects that were used upstream in the analysis.
            Each provenance is keyed based on the process name (e.g., "preprocessing").
        session : sqlalchemy.orm.session.Session or SmartSession
            The database session to use.  If None, a new session will be created,
            that is closed at the end of each `with` statement.

        Returns
        -------
        image_cache : dict
            The updated image cache.
        prov_ids: list
            A list with the provenance ID that was used to make this image.
            This is used downstream to make new provenances with upstream_ids.
        """
        # TODO: implement the actual code to do this.
        #  Check if the chosen catalog is loaded into this object.
        #  If not, load it and save it on "self".
        #  Check if the Image exists in the cache or the database.
        #  If not, load it from the database and add it to the cache.
        #  Extract sources from the Image.
        #  Cross match the sources with the catalog.
        #  Calculate the astrometric solution.
        #  Save the astrometric solution to the Image object.
        #  Save the Image object to the cache and database.
        #  Update the FITS header with the WCS.

        # should have the upstream Provenance for the Preprocessor
        if prov_cache is None:
            prov_cache = {}

        # get the provenance:
        prov = self.pars.get_provenance(prov_cache=prov_cache, session=session)

        # get the image cache:
        image_cache = get_image_cache(image_cache)
        image = image_cache['images'].get((exposure_id, ccd_id))

        # check provenance in the cached image is consistent with the image object provenance
        if image is not None:
            im_prov = prov_cache['preprocess']
            if image.provenance.unique_hash != im_prov.unique_hash:
                image = None

        if image is None:
            with SmartSession(session) as session:
                image = session.scalars(
                    sa.select(Image).where(
                        Image.exposure_id == exposure_id, Image.ccd_id == ccd_id, Image.provenance_id == im_prov.id
                    )
                ).first()

        if image is None:
            raise ValueError(f'Image with exposure id: {exposure_id}, CCD id: {ccd_id} '
                             f'and provenance id: {im_prov.id} not found in the cache or database!')

        # TODO: extract sources from the image
        # TODO: get the catalog and save it in "self"
        # TODO: cross-match the sources with the catalog
        # TODO: save a WorldCoordinateSystem object to database
        # TODO: update the image's FITS header with the wcs

        prov_cache['astrometry'] = prov

        # make sure this is returned to be used in the next step
        return image_cache, prov_cache
