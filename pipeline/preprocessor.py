
import sqlalchemy as sa

from models.base import SmartSession
from models.exposure import Exposure
from models.image import Image

from pipeline.parameters import Parameters

from pipeline.utils import get_image_cache


class ParsPreprocessor(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.use_sky_subtraction = self.add_par('use_sky_subtraction', True, bool, 'Apply sky subtraction. ')

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def _get_process_name(self):
        return 'preprocessor'

    def _get_upstream_process_names(self):
        return []  # no upstream processes!


class Preprocessor:
    def __init__(self, **kwargs):
        self.pars = ParsPreprocessor()
        self.pars.update(kwargs)

    def run(self, exposure_id, ccd_id, image_cache=None, session=None):
        """
        Run dark and flat processing, and apply sky subtraction.

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
        #  Save the dark/flat to attributes on "self"
        #  Check if the Exposure is in the cache or the database. Done
        #  The exposure must exist at this point, so we can have an exposure_id! Done
        #  Check if the Image object already exists in the cache, then database. Done
        #  Apply the dark/flat/sky subtraction
        #  If not, create an Image for the specific CCD and add it to the cache and database.

        # since we starting from raw data, there is no use for the provenance cache for this process
        if prov_cache is None:
            prov_cache = {}

        # get the provenance for this step:
        prov = self.pars.get_provenance(prov_cache=prov_cache, session=session)

        # get the image cache:
        image_cache = get_image_cache(image_cache)
        image = image_cache['images'].get((exposure_id, ccd_id))

        # check provenance in the cached image is consistent with this object's provenance
        if image is not None:
            if image.provenance.unique_hash != prov.unique_hash:
                image = None

        if image is None:
            with SmartSession(session) as session:
                image = session.scalars(
                    sa.select(Image).where(
                        Image.exposure_id == exposure_id, Image.ccd_id == ccd_id, Image.provenance_id == prov.id
                    )
                ).first()

        if image is None:  # need to make new image
            # check if exposure is in the cache
            exposure = image_cache['exposures'].get(exposure_id)

            # if not, check if it is in the database
            if exposure is None:
                with SmartSession(session) as session:
                    exposure = session.scalars(sa.select(Exposure).where(Exposure.id == exposure_id)).first()

            if exposure is None:
                raise ValueError(f'Exposure {exposure_id} not found in cache or database.')

            # TODO: get the CCD image from the exposure
            image = Image(exposure_id=exposure_id, ccd_id=ccd_id, provenance=prov)

        if image is None:
            raise ValueError('Image cannot be None at this point!')

        # TODO: apply dark/flat/sky subtraction

        # make sure to update the cache with the (maybe new) image
        image_cache['images'][exposure_id, ccd_id] = image

        # add the provenance used to produce this as well
        prov_cache['preprocessor'] = prov

        # make sure this is returned to be used in the next step
        return image_cache, prov_cache
