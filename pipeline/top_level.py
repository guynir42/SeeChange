import datetime

import sqlalchemy as sa

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore, UPSTREAM_STEPS
from pipeline.preprocessing import Preprocessor
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from pipeline.subtraction import Subtractor
from pipeline.detection import Detector
from pipeline.cutting import Cutter
from pipeline.measuring import Measurer

from util.config import Config

# should this come from db.py instead?
from models.base import SmartSession
from models.provenance import Provenance
from models.reference import Reference
from models.report import Report

# describes the pipeline objects that are used to produce each step of the pipeline
# if multiple objects are used in one step, replace the string with a sub-dictionary,
# where the sub-dictionary keys are the keywords inside the expected critical parameters
# that come from all the different objects.
PROCESS_OBJECTS = {
    'preprocessing': 'preprocessor',
    'extraction': 'extractor',  # the same object also makes the PSF (and background?)
    # TODO: when joining the astro/photo cal into extraction, use this format:
    # 'extraction': {
    #     'sources': 'extractor',
    #     'astro_cal': 'astro_cal',
    #     'photo_cal': 'photo_cal',
    # }
    'astro_cal': 'astro_cal',
    'photo_cal': 'photo_cal',
    'subtraction': 'subtractor',
    'detection': 'detector',
    'cutting': 'cutter',
    'measuring': 'measurer',
    # TODO: add one more for R/B deep learning scores
}


# put all the top-level pipeline parameters in the init of this class:
class ParsPipeline(Parameters):

    def __init__(self, **kwargs):
        super().__init__()

        self.example_pipeline_parameter = self.add_par(
            'example_pipeline_parameter', 1, int, 'an example pipeline parameter'
        )

        self._enforce_no_new_attrs = True  # lock against new parameters

        self.override(kwargs)


class Pipeline:
    def __init__(self, **kwargs):
        self.config = Config.get()

        # top level parameters
        self.pars = ParsPipeline(**(self.config.value('pipeline', {})))
        self.pars.augment(kwargs.get('pipeline', {}))

        # dark/flat and sky subtraction tools
        preprocessing_config = self.config.value('preprocessing', {})
        preprocessing_config.update(kwargs.get('preprocessing', {}))
        self.pars.add_defaults_to_dict(preprocessing_config)
        self.preprocessor = Preprocessor(**preprocessing_config)

        # source detection ("extraction" for the regular image!)
        extraction_config = self.config.value('extraction', {})
        extraction_config.update(kwargs.get('extraction', {'measure_psf': True}))
        self.pars.add_defaults_to_dict(extraction_config)
        self.extractor = Detector(**extraction_config)

        # astrometric fit using a first pass of sextractor and then astrometric fit to Gaia
        astro_cal_config = self.config.value('astro_cal', {})
        astro_cal_config.update(kwargs.get('astro_cal', {}))
        self.pars.add_defaults_to_dict(astro_cal_config)
        self.astro_cal = AstroCalibrator(**astro_cal_config)

        # photometric calibration:
        photo_cal_config = self.config.value('photo_cal', {})
        photo_cal_config.update(kwargs.get('photo_cal', {}))
        self.pars.add_defaults_to_dict(photo_cal_config)
        self.photo_cal = PhotCalibrator(**photo_cal_config)

        # reference fetching and image subtraction
        subtraction_config = self.config.value('subtraction', {})
        subtraction_config.update(kwargs.get('subtraction', {}))
        self.pars.add_defaults_to_dict(subtraction_config)
        self.subtractor = Subtractor(**subtraction_config)

        # source detection ("detection" for the subtracted image!)
        detection_config = self.config.value('detection', {})
        detection_config.update(kwargs.get('detection', {}))
        self.pars.add_defaults_to_dict(detection_config)
        self.detector = Detector(**detection_config)
        self.detector.pars.subtraction = True

        # produce cutouts for detected sources:
        cutting_config = self.config.value('cutting', {})
        cutting_config.update(kwargs.get('cutting', {}))
        self.pars.add_defaults_to_dict(cutting_config)
        self.cutter = Cutter(**cutting_config)

        # measure photometry, analytical cuts, and deep learning models on the Cutouts:
        measuring_config = self.config.value('measuring', {})
        measuring_config.update(kwargs.get('measuring', {}))
        self.pars.add_defaults_to_dict(measuring_config)
        self.measurer = Measurer(**measuring_config)

    def override_parameters(self, **kwargs):
        """Override some of the parameters for this object and its sub-objects, using Parameters.override(). """
        for key, value in kwargs.items():
            if key in PROCESS_OBJECTS:
                getattr(self, PROCESS_OBJECTS[key]).pars.override(value)
            else:
                self.pars.override({key: value})

    def augment_parameters(self, **kwargs):
        """Add some parameters to this object and its sub-objects, using Parameters.augment(). """
        for key, value in kwargs.items():
            if key in PROCESS_OBJECTS:
                getattr(self, PROCESS_OBJECTS[key]).pars.augment(value)
            else:
                self.pars.augment({key: value})

    def run(self, *args, **kwargs):
        """
        Run the entire pipeline on a specific CCD in a specific exposure.
        Will open a database session and grab any existing data,
        and calculate and commit any new data that did not exist.
        """

        try:
            ds, session = DataStore.from_args(*args, **kwargs)
        except Exception as e:
            ds = DataStore.catch_failure_to_parse(e, *args)

        if ds.exposure is None:
            raise RuntimeError('Not sure if there is a way to run this pipeline method without an exposure!')

        try:  # must make sure the exposure is on the DB
            with SmartSession(session) as dbsession:
                ds.exposure = dbsession.merge(ds.exposure)
                dbsession.commit()
        except Exception as e:
            raise RuntimeError('Failed to merge the exposure into the session!') from e

        try:  # make sure we have a reference for this exposure
            with SmartSession(session) as dbsession:
                reference = Reference.get_reference(ds.exposure.filter, ds.exposure.target, ds.exposure.observation_time, dbsession)
                if reference is None:
                    raise RuntimeError('No reference found!')
        except Exception as e:
            raise RuntimeError(f'Cannot get reference exposure {ds.exposure.filepath}!') from e


        try:  # create (and commit, if not existing) all provenances for the products
            with SmartSession(session) as dbsession:
                provs = self.make_provenance_tree(ds.exposure, session=dbsession, commit=True)
        except Exception as e:
            raise RuntimeError('Failed to create the provenance tree!') from e

        try:  # must make sure the report is on the DB
            report = Report(exposure=ds.exposure, section_id=ds.section_id)
            report.started_at = datetime.datetime.utcnow()
            prov = Provenance(
                process='report',
                code_version=ds.exposure.provenance.code_version,
                parameters={},
                upstreams=[provs['measuring']],
                is_testing=ds.exposure.provenance.is_testing,
            )
            report.provenance = prov
            with SmartSession(session) as dbsession:
                report = dbsession.merge(report)
                dbsession.commit()

            if report.exposure_id is None:
                raise RuntimeError('Report did not get a valid exposure_id!')
        except Exception as e:
            raise RuntimeError('Failed to create or merge a report for the exposure!') from e

        # run dark/flat preprocessing, cut out a specific section of the sensor
        # TODO: save the results as Image objects to DB and disk? Or save at the end?
        ds = self.preprocessor.run(ds, session)
        report.scan_datastore(ds, 'preprocessing', session)

        # extract sources and make a SourceList and PSF from the image
        ds = self.extractor.run(ds, session)
        report.scan_datastore(ds, 'extraction', session)

        # find astrometric solution, save WCS into Image object and FITS headers
        ds = self.astro_cal.run(ds, session)
        report.scan_datastore(ds, 'astro_cal', session)

        # cross-match against photometric catalogs and get zero point, save into Image object and FITS headers
        ds = self.photo_cal.run(ds, session)
        report.scan_datastore(ds, 'photo_cal', session)

        # fetch reference images and subtract them, save SubtractedImage objects to DB and disk
        ds = self.subtractor.run(ds, session)
        report.scan_datastore(ds, 'subtraction', session)

        # find sources, generate a source list for detections
        ds = self.detector.run(ds, session)
        report.scan_datastore(ds, 'detection', session)

        # make cutouts of all the sources in the "detections" source list
        ds = self.cutter.run(ds, session)
        report.scan_datastore(ds, 'cutting', session)

        # extract photometry, analytical cuts, and deep learning models on the Cutouts:
        ds = self.measurer.run(ds, session)
        report.scan_datastore(ds, 'measuring', session)

        return ds

    def run_with_session(self):
        """
        Run the entire pipeline using one session that is opened
        at the beginning and closed at the end of the session,
        just to see if that causes any problems with too many open sessions.
        """
        with SmartSession() as session:
            self.run(session=session)

    @staticmethod
    def get_reference(filter, target, obs_time, session=None):
        """
        Get a reference for a given filter, target, and observation time.

        Parameters
        ----------
        filter: str
            The filter of the image/exposure.
        target: str
            The target of the image/exposure, or the name of the field.  # TODO: can we replace this with coordinates?
        obs_time: datetime
            The observation time of the image.
        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.

        Returns
        -------
        ref: Image object
            The reference image for this image, or None if no reference is found.

        """
        return Reference.get_reference(filter, target, obs_time, session=session)

    def make_provenance_tree(self, exposure, session=None, commit=True):
        """Use the current configuration of the pipeline and all the objects it has
        to generate the provenances for all the processing steps.
        This will conclude with the reporting step, which simply has an upstreams
        list of provenances to the measuring provenance and to the machine learning score
        provenances. From those, a user can recreate the entire tree of provenances.

        Parameters
        ----------
        exposure : Exposure
            The exposure to use to get the initial provenance.
            This provenance should be automatically created by the exposure.
        session : SmartSession, optional
            The function needs to work with the database to merge existing provenances.
            If a session is given, it will use that, otherwise it will open a new session,
            which will also close automatically at the end of the function.
        commit: bool, optional, default True
            By default, the provenances are merged and committed inside this function.
            To disable this, set commit=False. This may leave the provenances in a
            transient state, and is most likely not what you want.

        Returns
        -------
        dict
            A dictionary of all the provenances that were created in this function,
            keyed according to the different steps in the pipeline.
            The provenances are all merged to the session.
        """
        with SmartSession(session) as session:
            # start by getting the exposure and reference
            exposure = session.merge(exposure)  # also merges the provenance and code_version
            reference = self.get_reference(exposure.filter, exposure.target, exposure.observation_time, session=session)

            provs = {'exposure': exposure.provenance}  # TODO: does this always work on any exposure?
            code_version = exposure.provenance.code_version
            is_testing = exposure.provenance.is_testing

            for step in PROCESS_OBJECTS:
                if isinstance(PROCESS_OBJECTS[step], dict):
                    parameters = {}
                    for key, value in PROCESS_OBJECTS[step].items():
                        parameters[key] = getattr(self, value).pars.get_critical_pars()
                else:
                    parameters = getattr(self, PROCESS_OBJECTS[step]).pars.get_critical_pars()

                # some preprocessing parameters (the "preprocessing_steps") doesn't come from the
                # config file, but instead comes from the preprocessing itself.
                # TODO: fix this as part of issue #147
                if step == 'preprocessing':
                    if 'preprocessing_steps' not in parameters:
                        parameters['preprocessing_steps'] = ['overscan', 'linearity', 'flat', 'fringe']

                # figure out which provenances go into the upstreams for this step
                up_steps = UPSTREAM_STEPS[step]
                if isinstance(up_steps, str):
                    up_steps = [up_steps]
                upstreams = []
                for upstream in up_steps:
                    if upstream == 'reference':
                        upstreams += reference.provenance.upstreams
                    else:
                        upstreams.append(provs[upstream])

                provs[step] = Provenance(
                    code_version=code_version,
                    process=step,
                    parameters=parameters,
                    upstreams=upstreams,
                    is_testing=is_testing,
                )

                provs[step] = session.merge(provs[step])

            if commit:
                session.commit()

            return provs
