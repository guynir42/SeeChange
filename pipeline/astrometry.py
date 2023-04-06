import sqlalchemy as sa

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

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

    def get_process_name(self):
        return 'astrometry'


class Astrometry:
    def __init__(self, **kwargs):
        self.pars = ParsAstrometry()
        self.pars.update(kwargs)

    def run(self, *args, **kwargs):
        """
        Extract sources and use their positions to calculate the astrometric solution.
        Arguments are parsed by the DataStore.parse_args() method.

        Returns a DataStore object with the products of the processing.
        """
        # TODO: implement the actual code to do this.
        #  Check if the chosen catalog is loaded into this object.
        #  If not, load it and save it on "self".
        #  Extract sources from the Image.
        #  Cross match the sources with the catalog.
        #  Calculate the astrometric solution.
        #  Save the astrometric solution to the Image object.
        #  Save the Image object to the cache and database.
        #  Update the FITS header with the WCS.
        ds = DataStore.from_args(*args, **kwargs)

        # get the provenance for this step:
        prov = ds.get_provenance('astrometry', self.pars.get_critical_pars(), session=ds.session)

        # try to find the world coordinates in memory or in the database:
        wcs = ds.get_wcs(prov, session=ds.session)

        if wcs is None:  # must create a new WorldCoordinate object

            # use the latest image in the data store,
            # or load using the provenance given in the
            # data store's upstream_provs, or just use
            # the most recent provenance for "preprocessing"
            image = ds.get_image(session=ds.session)

            if image is None:
                raise ValueError(f'Cannot find an image corresponding to the datastore inputs: {ds.get_inputs()}')

        # TODO: extract sources from the image
        # TODO: get the catalog and save it in "self"
        # TODO: cross-match the sources with the catalog
        # TODO: save a WorldCoordinateSystem object to database
        # TODO: update the image's FITS header with the wcs
        # TODO: add the resulting object to the data store

        # make sure this is returned to be used in the next step
        return ds
