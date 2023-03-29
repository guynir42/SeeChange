
import sqlalchemy as sa

from pipeline.parameters import Parameters

from pipeline.utils import get_image_cache

from models.image import Image, CoaddImage, SubtractionImage
from models.base import SmartSession


class ParsDetector(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.threshold = self.add_par(
            'threshold',
            5.0,
            [float, int],
            'The number of standard deviations above the background '
            'to use as the threshold for detecting a source. '
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def _get_process_name(self):
        return 'detection'

    def _get_upstream_process_names(self):
        return ['subtraction']


class Detector:
    def __init__(self, **kwargs):
        self.pars = ParsDetector()
        self.pars.update(kwargs)

    def run(self, exposure_id, ccd_id, image_cache=None, prov_cache=None, session=None):
        """
        Search a subtraction image for new sources, create cutouts, and append them to an Object.

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
        #  Get the SubtractionImage from the image_cache or from the database.
        #  Locate all the sources above threshold.
        #  Save a cutout from the SubtractionImage for each source.
        #  Load the Image and ReferenceImage from the image_cache or from the database.
        #  Make cutouts from them too, add those to the new Cutout objects.
        #  Find the coordinates of each new source.
        #  Match that against the database of Objects.
        #  Make a new Object for sources that have no match.
        #  For each Object, add a new Sighting object that contains a Cutouts object.
        #  Save any new Objects, Sightings and Cutouts to the database (and Cutouts to disk).
        #  Return the list of Objects, that contain the Sightings, that contain the Cutouts.

        # should have the upstream Provenance for subtraction
        if prov_cache is None:
            prov_cache = {}

        # get the provenance:
        prov = self.pars.get_provenance(prov_cache=prov_cache, session=session)

        # get the image cache:
        image_cache = get_image_cache(image_cache)
        sub_image = image_cache['subtraction'].get((exposure_id, ccd_id))

        # check provenance in the cached image is consistent with the image provenance
        im_prov = prov_cache['subtraction']
        if sub_image is not None:
            if sub_image.provenance.unique_hash != im_prov.unique_hash:
                image = None

        if sub_image is None:
            with SmartSession(session) as session:
                sub_image = session.scalars(
                    sa.select(SubtractionImage).where(
                        SubtractionImage.exposure_id == exposure_id,
                        SubtractionImage.ccd_id == ccd_id,
                        SubtractionImage.provenance_id == im_prov.id
                    )
                ).first()

        if sub_image is None:
            raise ValueError(f'SubtractionImage with exposure id: {exposure_id}, CCD id: {ccd_id} '
                             f'and provenance id: {im_prov.id} not found in the cache or database!')


        # TODO: find all sources above threshold
        # TODO: save cutouts for each source
        # TODO: get the Image and ReferenceImage from the image_cache or from the database.
        # TODO: make cutouts from them too, add those to the new Cutout objects.
        # TODO: add the new Cutout objects to the image cache.
        # TODO: make Sighting objects for each source and add them to the database
        # TODO: match the coordinates of each Sighting to existing objects in the DB
        # TODO: any that are not associated should also have new Objects created and added to the DB

        cutouts_list = []

        image_cache['cutouts'][(exposure_id, ccd_id)] = cutouts_list  # TODO: is it easier to make a list of Sightings?

        prov_cache['detection'] = prov

        # make sure this is returned to be used in the next step
        return image_cache
