

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.preprocessor import Preprocessor
from pipeline.astrometry import Astrometry
from pipeline.calibrator import Calibrator
from pipeline.subtractor import Subtractor
from pipeline.detector import Detector
from pipeline.extractor import Extractor


# should this come from db.py instead?
from models.base import SmartSession

config = {}  # TODO: replace this with Rob's config loader


# put all the top-level pipeline parameters in the init of this class:
class ParsPipeline(Parameters):

        def __init__(self):
            super().__init__()

            self.add_par('example_pipeline_parameter', 1, int, 'an example pipeline parameter')

            self._enforce_no_new_attrs = True  # lock against new parameters


class Pipeline:
    def __init__(self, **kwargs):
        # top level parameters
        self.pars = ParsPipeline(config.get('pipeline', {}))
        self.pars.update(kwargs.get('pipeline', {}))

        # dark/flat and sky subtraction tools
        preprocessor_config = config.get('preprocessor', {})
        preprocessor_config.update(kwargs.get('preprocessor', {}))
        self.pars.add_defaults_to_dict(preprocessor_config)
        self.preprocessor = Preprocessor(**preprocessor_config)

        # astrometric fit using a first pass of sextractor and then astrometric fit to Gaia
        astrometry_config = config.get('astrometry', {})
        astrometry_config.update(kwargs.get('astrometry', {}))
        self.pars.add_defaults_to_dict(astrometry_config)
        self.astrometry = Astrometry(**astrometry_config)

        # photometric calibration:
        calibrator_config = config.get('calibrator', {})
        calibrator_config.update(kwargs.get('calibrator', {}))
        self.pars.add_defaults_to_dict(calibrator_config)
        self.calibrator = Calibrator(**calibrator_config)

        # reference fetching and image subtraction
        subtractor_config = config.get('subtractor', {})
        subtractor_config.update(kwargs.get('subtractor', {}))
        self.pars.add_defaults_to_dict(subtractor_config)
        self.subtractor = Subtractor(**subtractor_config)

        # find sources and turn them into cutouts
        detector_config = config.get('detector', {})
        detector_config.update(kwargs.get('detector', {}))
        self.pars.add_defaults_to_dict(detector_config)
        self.detector = Detector(**detector_config)

        # extract photometry, analytical cuts, and deep learning models on the Cutouts:
        extractor_config = config.get('extractor', {})
        extractor_config.update(kwargs.get('extractor', {}))
        self.pars.add_defaults_to_dict(extractor_config)
        self.extractor = Extractor(**extractor_config)

    def run(self, *args, **kwargs):
        """
        Run the entire pipeline on a specific CCD in a specific exposure.
        Will open a database session and grab any existing data,
        and calculate and commit any new data that did not exist.
        """

        ds = DataStore.from_args(*args, **kwargs)

        if ds.coadd_id is None:
            # run dark/flat and sky subtraction tools, save the results as Image objects to DB and disk
            ds = self.preprocessor.run(ds)
        else:
            # start by getting a coadd image but run the rest of the pipeline as usual
            pass  # do something else!

        # extract sources and fit astrometric solution, save WCS into Image object and FITS headers
        ds = self.astrometry.run(ds)

        # cross-match against photometric catalogs and get zero point, save into Image object and FITS headers
        ds = self.calibrator.run(ds)

        # fetch reference images and subtract them, save SubtractedImage objects to DB and disk
        ds = self.subtractor.run(ds)

        # find sources, generate Sightings and Cutouts, make new Object if missing, append to existing Objects
        ds = self.detector.run(ds)

        # extract photometry, analytical cuts, and deep learning models on the Cutouts:
        ds = self.extractor(ds)

        return ds

    def run_with_session(self):
        """
        Run the entire pipeline using one session that is opened
        at the beginning and closed at the end of the session,
        just to see if that causes any problems with too many open sessions.
        """
        with SmartSession() as session:
            self.run(session=session)
