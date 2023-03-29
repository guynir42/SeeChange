
import sqlalchemy as sa

from pipeline.parameters import Parameters

from pipeline.utils import get_image_cache

from models.cutouts import Cutouts
from models.sightings import Sighting
from models.photometry import Photometry
from models.base import SmartSession


class ParsExtractor(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.photometry_method = self.add_par(
            'photometry_method',
            'aperture',
            str,
            'Type of photometry used. Possible values are "psf" and "aperture". '
        )
        self.aperture_radius = self.add_par(
            'aperture_radius',
            3.0,
            [float, list],
            'Radius of the aperture in pixels. Can give a list of values. '
        )  # TODO: should this be in pixels or in arcsec?

        self.real_bogus_version = self.add_par(
            'real_bogus_version',
            None,
            [str, None],
            'Which version of Real/Bogus deep learning code was used. '
            'If None, then deep learning will not be used. '
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def _get_process_name(self):
        return 'extraction'

    def _get_upstream_process_names(self):
        return ['detection']


class Extractor:
    def __init__(self, **kwargs):
        self.pars = ParsExtractor()
        self.pars.update(kwargs)

    def run(self, exposure_id, ccd_id, image_cache, prov_cache=None, session=None):
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
        #  For each object name, find an Object in the objects list or on the database.
        #  For each Object, get the Sighting with the Cutout that are relevant to this exposure_id.
        #  For each Cutouts calculate the photometry (flux, centroids).
        #  Apply analytic cuts to each stamp image, to rule out artefacts.
        #  Apply deep learning (real/bogus) to each stamp image, to rule out artefacts.
        #  Save the results as Photometry objects, append them to the Sightings objects.
        #  Commit the results to the database.

        # should have the upstream Provenance for detection
        if prov_cache is None:
            prov_cache = {}

        # get the provenance:
        prov = self.pars.get_provenance(prov_cache=prov_cache, session=session)

        # get the image cache:
        image_cache = get_image_cache(image_cache)
        cutouts_list = image_cache['cutouts'].get((exposure_id, ccd_id), [])
        cutouts_prov = prov_cache['cutouts']

        if len(cutouts_list) > 0:
            if any([c.provenance.id != cutouts_prov.id for c in cutouts_list]):
                cutouts_list = []  # wrong provenance, make a new list

        if len(cutouts_list) == 0:
            # try to load the cutouts from the database:
            with SmartSession(session) as session:
                cutouts_list = session.scalars(sa.select(Cutouts).where(
                    Cutouts.exposure_id == exposure_id, Cutouts.ccd_id == ccd_id
                )
            ).all()

        if len(cutouts_list) == 0:
            # TODO: is there any chance an image CCD will actually not have any sources?
            raise ValueError(f'No cutouts found for exposure_id={exposure_id}, ccd_id={ccd_id} '
                             f'and provenance id {cutouts_prov.id}!')

        # TODO: run the photometry, analytical cuts, and deep learning on each Cutout
        # TODO: make sure to update the results on the Sighting that is the parent of each Cutout

        for cutouts in cutouts_list:
            pass  # TODO: continue this

        image_cache['cutouts'][(exposure_id, ccd_id)] = cutouts_list  # TODO: is it easier to make a list of Sightings?

        prov_cache['extraction'] = prov

        return image_cache, prov_cache