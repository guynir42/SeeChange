
import sqlalchemy as sa

from pipeline.parameters import Parameters

from pipeline.utils import get_image_cache

from models.image import Image, CoaddImage, SubtractionImage
from models.base import SmartSession


class ParsSubtractor(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.algorithm = self.add_par(
            'algorithm',
            'hotpants',
            str,
            'Which algorithm to use. Possible values are: "hotpants", "zogy". '
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def _get_process_name(self):
        return 'subtraction'

    def _get_upstream_process_names(self):
        return ['preprocess', 'astrometry', 'calibration']


class Subtractor:
    def __init__(self, **kwargs):
        self.pars = ParsSubtractor()
        self.pars.update(kwargs)

        # TODO: add a reference cache here.

    def run(self, exposure_id, ccd_id, image_cache=None, prov_cache=None, session=None):
        """
        Get a reference image and subtract it from the new image.

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
        #  Check if there is a reference image in self.reference cache.
        #  If not, create one and check if the there are relevant images in the database.
        #  If not, raise an exception.
        #  If yes, get them and make a coadd-reference and save it to the cache.
        #  Check if the new Image exists in the cache or the database.
        #  Apply the chosen subtraction algorithm and produce a SubtractionImage object.
        #  Add it to the image_cache and also save it to DB and to disk.

        # should have the upstream Provenance for preprocess and astrometry
        if prov_cache is None:
            prov_cache = {}

        # get the provenance:
        prov = self.pars.get_provenance(prov_cache=prov_cache, session=session)

        # get the image cache:
        image_cache = get_image_cache(image_cache)
        image = image_cache['images'].get((exposure_id, ccd_id))

        # check provenance in the cached image is consistent with the image provenance
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

        # TODO: check if a coadd image exists in the cache or the database.
        #  If not, create one and save it to the cache and to the database.
        # TODO: get the Image and CoaddImage and make a SubtractionImage
        #  Save it to the cache and database.
        sub_image = SubtractionImage()

        image_cache['subtractions'][(exposure_id, ccd_id)] = sub_image

        prov_cache['subtraction'] = prov

        # make sure this is returned to be used in the next step
        return image_cache, prov_cache
