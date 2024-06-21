import datetime
import time
import warnings

import numpy as np
import sqlalchemy as sa

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore, UPSTREAM_STEPS
from pipeline.preprocessing import Preprocessor
from pipeline.backgrounding import Backgrounder
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from pipeline.subtraction import Subtractor
from pipeline.detection import Detector
from pipeline.cutting import Cutter
from pipeline.measuring import Measurer
from pipeline.coaddition import CoaddPipeline

from models.base import SmartSession
from models.provenance import Provenance
from models.reference import Reference
from models.exposure import Exposure
from models.image import Image
from models.report import Report
from models.instrument import get_instrument_instance

from util.config import Config
from util.logger import SCLogger
from util.util import parse_env, parse_session
from util.radec import parse_sexigesimal_degrees

# describes the pipeline objects that are used to produce each step of the pipeline
# if multiple objects are used in one step, replace the string with a sub-dictionary,
# where the sub-dictionary keys are the keywords inside the expected critical parameters
# that come from all the different objects.
PROCESS_OBJECTS = {
    'preprocessing': 'preprocessor',
    'extraction': {
        'sources': 'extractor',
        'psf': 'extractor',
        'bg': 'backgrounder',
        'wcs': 'astrometor',
        'zp': 'photometor',
    },
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
            'example_pipeline_parameter', 1, int, 'an example pipeline parameter', critical=False
        )

        self.save_before_subtraction = self.add_par(
            'save_before_subtraction',
            True,
            bool,
            'Save intermediate images to the database, '
            'after doing extraction, background, and astro/photo calibration, '
            'if there is no reference, will not continue to doing subtraction'
            'but will still save the products up to that point. ',
            critical=False,
        )

        self.save_at_finish = self.add_par(
            'save_at_finish',
            True,
            bool,
            'Save the final products to the database and disk',
            critical=False,
        )

        self._enforce_no_new_attrs = True  # lock against new parameters

        self.override(kwargs)


class Pipeline:
    def __init__(self, **kwargs):
        config = Config.get()

        # top level parameters
        self.pars = ParsPipeline(**(config.value('pipeline', {})))
        self.pars.augment(kwargs.get('pipeline', {}))

        # dark/flat and sky subtraction tools
        preprocessing_config = config.value('preprocessing', {})
        preprocessing_config.update(kwargs.get('preprocessing', {}))
        self.pars.add_defaults_to_dict(preprocessing_config)
        self.preprocessor = Preprocessor(**preprocessing_config)

        # source detection ("extraction" for the regular image!)
        extraction_config = config.value('extraction.sources', {})
        extraction_config.update(kwargs.get('extraction', {}).get('sources', {}))
        extraction_config.update({'measure_psf': True})
        self.pars.add_defaults_to_dict(extraction_config)
        self.extractor = Detector(**extraction_config)

        # background estimation using either sep or other methods
        background_config = config.value('extraction.bg', {})
        background_config.update(kwargs.get('extraction', {}).get('bg', {}))
        self.pars.add_defaults_to_dict(background_config)
        self.backgrounder = Backgrounder(**background_config)

        # astrometric fit using a first pass of sextractor and then astrometric fit to Gaia
        astrometor_config = config.value('extraction.wcs', {})
        astrometor_config.update(kwargs.get('extraction', {}).get('wcs', {}))
        self.pars.add_defaults_to_dict(astrometor_config)
        self.astrometor = AstroCalibrator(**astrometor_config)

        # photometric calibration:
        photometor_config = config.value('extraction.zp', {})
        photometor_config.update(kwargs.get('extraction', {}).get('zp', {}))
        self.pars.add_defaults_to_dict(photometor_config)
        self.photometor = PhotCalibrator(**photometor_config)

        # make sure when calling get_critical_pars() these objects will produce the full, nested dictionary
        siblings = {
            'sources': self.extractor.pars,
            'bg': self.backgrounder.pars,
            'wcs': self.astrometor.pars,
            'zp': self.photometor.pars,
        }
        self.extractor.pars.add_siblings(siblings)
        self.backgrounder.pars.add_siblings(siblings)
        self.astrometor.pars.add_siblings(siblings)
        self.photometor.pars.add_siblings(siblings)

        # reference fetching and image subtraction
        subtraction_config = config.value('subtraction', {})
        subtraction_config.update(kwargs.get('subtraction', {}))
        self.pars.add_defaults_to_dict(subtraction_config)
        self.subtractor = Subtractor(**subtraction_config)

        # source detection ("detection" for the subtracted image!)
        detection_config = config.value('detection', {})
        detection_config.update(kwargs.get('detection', {}))
        self.pars.add_defaults_to_dict(detection_config)
        self.detector = Detector(**detection_config)
        self.detector.pars.subtraction = True

        # produce cutouts for detected sources:
        cutting_config = config.value('cutting', {})
        cutting_config.update(kwargs.get('cutting', {}))
        self.pars.add_defaults_to_dict(cutting_config)
        self.cutter = Cutter(**cutting_config)

        # measure photometry, analytical cuts, and deep learning models on the Cutouts:
        measuring_config = config.value('measuring', {})
        measuring_config.update(kwargs.get('measuring', {}))
        self.pars.add_defaults_to_dict(measuring_config)
        self.measurer = Measurer(**measuring_config)

    def override_parameters(self, **kwargs):
        """Override some of the parameters for this object and its sub-objects, using Parameters.override(). """
        for key, value in kwargs.items():
            if key in PROCESS_OBJECTS:
                if isinstance(PROCESS_OBJECTS[key], dict):
                    for sub_key, sub_value in PROCESS_OBJECTS[key].items():
                        if sub_key in value:
                            getattr(self, PROCESS_OBJECTS[key][sub_value]).pars.override(value[sub_key])
                elif isinstance(PROCESS_OBJECTS[key], str):
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

    def setup_datastore(self, *args, **kwargs):
        """Initialize a datastore, including an exposure and a report, to use in the pipeline run.

        Will raise an exception if there is no valid Exposure,
        if there's no reference available, or if the report cannot
        be posted to the database.

        After these objects are instantiated, the pipeline will proceed
        and record any exceptions into the report object before raising them.

        Parameters
        ----------
        Inputs should include the exposure and section_id, or a datastore
        with these things already loaded. If a session is passed in as
        one of the arguments, it will be used as a single session for
        running the entire pipeline (instead of opening and closing
        sessions where needed).

        Returns
        -------
        ds : DataStore
            The DataStore object that was created or loaded.
        session: sqlalchemy.orm.session.Session
            An optional session. If not given, this will be None
        """
        ds, session = DataStore.from_args(*args, **kwargs)

        if ds.exposure is None:
            raise RuntimeError('Cannot run this pipeline method without an exposure!')

        try:  # must make sure the exposure is on the DB
            ds.exposure = ds.exposure.merge_concurrent(session=session)
        except Exception as e:
            raise RuntimeError('Failed to merge the exposure into the session!') from e

        try:  # create (and commit, if not existing) all provenances for the products
            with SmartSession(session) as dbsession:
                provs = self.make_provenance_tree(ds.exposure, session=dbsession, commit=True)
        except Exception as e:
            raise RuntimeError('Failed to create the provenance tree!') from e

        try:  # must make sure the report is on the DB
            report = Report(exposure=ds.exposure, section_id=ds.section_id)
            report.start_time = datetime.datetime.utcnow()
            prov = Provenance(
                process='report',
                code_version=ds.exposure.provenance.code_version,
                parameters={},
                upstreams=[provs['measuring']],
                is_testing=ds.exposure.provenance.is_testing,
            )
            report.provenance = prov
            with SmartSession(session) as dbsession:
                # check how many times this report was generated before
                prev_rep = dbsession.scalars(
                    sa.select(Report).where(
                        Report.exposure_id == ds.exposure.id,
                        Report.section_id == ds.section_id,
                        Report.provenance_id == prov.id,
                    )
                ).all()
                report.num_prev_reports = len(prev_rep)
                report = dbsession.merge(report)
                dbsession.commit()

            if report.exposure_id is None:
                raise RuntimeError('Report did not get a valid exposure_id!')
        except Exception as e:
            raise RuntimeError('Failed to create or merge a report for the exposure!') from e

        ds.report = report

        return ds, session

    def run(self, *args, **kwargs):
        """
        Run the entire pipeline on a specific CCD in a specific exposure.
        Will open a database session and grab any existing data,
        and calculate and commit any new data that did not exist.

        Parameters
        ----------
        Inputs should include the exposure and section_id, or a datastore
        with these things already loaded. If a session is passed in as
        one of the arguments, it will be used as a single session for
        running the entire pipeline (instead of opening and closing
        sessions where needed).

        Returns
        -------
        ds : DataStore
            The DataStore object that includes all the data products.
        """
        try:  # first make sure we get back a datastore, even an empty one
            ds, session = self.setup_datastore(*args, **kwargs)
        except Exception as e:
            return DataStore.catch_failure_to_parse(e, *args)

        try:
            if ds.image is not None:
                SCLogger.info(f"Pipeline starting for image {ds.image.id} ({ds.image.filepath})")
            elif ds.exposure is not None:
                SCLogger.info(f"Pipeline starting for exposure {ds.exposure.id} ({ds.exposure}) section {ds.section_id}")
            else:
                SCLogger.info(f"Pipeline starting with args {args}, kwargs {kwargs}")

            if parse_env('SEECHANGE_TRACEMALLOC'):
                # ref: https://docs.python.org/3/library/tracemalloc.html#record-the-current-and-peak-size-of-all-traced-memory-blocks
                import tracemalloc
                tracemalloc.start()  # trace the size of memory that is being used

            with warnings.catch_warnings(record=True) as w:
                ds.warnings_list = w  # appends warning to this list as it goes along
                # run dark/flat preprocessing, cut out a specific section of the sensor

                SCLogger.info(f"preprocessor")
                ds = self.preprocessor.run(ds, session)
                ds.update_report('preprocessing', session=None)
                SCLogger.info(f"preprocessing complete: image id = {ds.image.id}, filepath={ds.image.filepath}")

                # extract sources and make a SourceList and PSF from the image
                SCLogger.info(f"extractor for image id {ds.image.id}")
                ds = self.extractor.run(ds, session)
                ds.update_report('extraction', session=None)

                # find the background for this image
                SCLogger.info(f"backgrounder for image id {ds.image.id}")
                ds = self.backgrounder.run(ds, session)
                ds.update_report('extraction', session=None)

                # find astrometric solution, save WCS into Image object and FITS headers
                SCLogger.info(f"astrometor for image id {ds.image.id}")
                ds = self.astrometor.run(ds, session)
                ds.update_report('extraction', session=None)

                # cross-match against photometric catalogs and get zero point, save into Image object and FITS headers
                SCLogger.info(f"photometor for image id {ds.image.id}")
                ds = self.photometor.run(ds, session)
                ds.update_report('extraction', session=None)

                if self.pars.save_before_subtraction:
                    t_start = time.perf_counter()
                    try:
                        SCLogger.info(f"Saving intermediate image for image id {ds.image.id}")
                        ds.save_and_commit(session=session)
                    except Exception as e:
                        ds.update_report('save intermediate', session=None)
                        SCLogger.error(f"Failed to save intermediate image for image id {ds.image.id}")
                        SCLogger.error(e)
                        raise e

                    ds.runtimes['save_intermediate'] = time.perf_counter() - t_start

                # fetch reference images and subtract them, save subtracted Image objects to DB and disk
                SCLogger.info(f"subtractor for image id {ds.image.id}")
                ds = self.subtractor.run(ds, session)
                ds.update_report('subtraction', session=None)

                # find sources, generate a source list for detections
                SCLogger.info(f"detector for image id {ds.image.id}")
                ds = self.detector.run(ds, session)
                ds.update_report('detection', session=None)

                # make cutouts of all the sources in the "detections" source list
                SCLogger.info(f"cutter for image id {ds.image.id}")
                ds = self.cutter.run(ds, session)
                ds.update_report('cutting', session=None)

                # extract photometry and analytical cuts
                SCLogger.info(f"measurer for image id {ds.image.id}")
                ds = self.measurer.run(ds, session)
                ds.update_report('measuring', session=None)

                # measure deep learning models on the cutouts/measurements
                # TODO: add this...

                if self.pars.save_at_finish:
                    t_start = time.perf_counter()
                    try:
                        SCLogger.info(f"Saving final products for image id {ds.image.id}")
                        ds.save_and_commit(session=session)
                    except Exception as e:
                        ds.update_report('save final', session)
                        SCLogger.error(f"Failed to save final products for image id {ds.image.id}")
                        SCLogger.error(e)
                        raise e

                    ds.runtimes['save_final'] = time.perf_counter() - t_start

                ds.finalize_report(session)

                return ds

        except Exception as e:
            ds.catch_exception(e)
        finally:
            # make sure the DataStore is returned in case the calling scope want to debug the pipeline run
            return ds

    def run_with_session(self):
        """
        Run the entire pipeline using one session that is opened
        at the beginning and closed at the end of the session,
        just to see if that causes any problems with too many open sessions.
        """
        with SmartSession() as session:
            self.run(session=session)

    def make_provenance_tree(self, exposure, reference=None, overrides=None, session=None, commit=True):
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
        reference: str or Reference object or Provenance object or None
            Can be a string matching a valid reference set. This tells the pipeline which
            provenance to load for the reference.
            Instead, can provide either a Reference object with a Provenance
            or the Provenance object of a reference directly.
            If not given, will simply load the most recently created reference provenance.
            # TODO: when we implement reference sets, we will probably not allow this input directly to
            #  this function anymore. Instead, you will need to define the reference set in the config,
            #  under the subtraction parameters.
        overrides: dict, optional
            A dictionary of provenances to override any of the steps in the pipeline.
            For example, set overrides={'preprocessing': prov} to use a specific provenance
            for the basic Image provenance.
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
        if overrides is None:
            overrides = {}

        with SmartSession(session) as session:
            # start by getting the exposure and reference
            # TODO: need a better way to find the relevant reference PROVENANCE for this exposure
            #  i.e., we do not look for a valid reference and get its provenance, instead,
            #  we look for a provenance based on our policy (that can be defined in the subtraction parameters)
            #  and find a specific provenance id that matches our policy.
            #  If we later find that no reference with that provenance exists that overlaps our images,
            #  that will be recorded as an error in the report.
            #  One way to do this would be to add a RefSet table that has a name (e.g., "standard") and
            #  a validity time range (which will be removed from Reference), maybe also the instrument.
            #  That would allow us to use a combination of name+obs_time to find a specific RefSet,
            #  which has a single reference provenance ID. If you want a custom reference,
            #  add a new RefSet with a new name.
            #  This also means that the reference making pipeline MUST use a single set of policies
            #  to create all the references for a given RefSet... we need to make sure we can actually
            #  make that happen consistently (e.g., if you change parameters or start mixing instruments
            #  when you make the references it will create multiple provenances for the same RefSet).
            if isinstance(reference, str):
                raise NotImplementedError('See issue #287')
            elif isinstance(reference, Reference):
                ref_prov = reference.provenance
            elif isinstance(reference, Provenance):
                ref_prov = reference
            elif reference is None:  # use the latest provenance that has to do with references
                ref_prov = session.scalars(
                    sa.select(Provenance).where(
                        Provenance.process == 'reference'
                    ).order_by(Provenance.created_at.desc())
                ).first()

            exp_prov = session.merge(exposure.provenance)  # also merges the code_version
            provs = {'exposure': exp_prov}
            code_version = exp_prov.code_version
            is_testing = exp_prov.is_testing

            for step in PROCESS_OBJECTS:
                if step in overrides:
                    provs[step] = overrides[step]
                else:
                    obj_name = PROCESS_OBJECTS[step]
                    if isinstance(obj_name, dict):
                        # get the first item of the dictionary and hope its pars object has siblings defined correctly:
                        obj_name = obj_name.get(list(obj_name.keys())[0])
                    parameters = getattr(self, obj_name).pars.get_critical_pars()

                    # figure out which provenances go into the upstreams for this step
                    up_steps = UPSTREAM_STEPS[step]
                    if isinstance(up_steps, str):
                        up_steps = [up_steps]
                    upstreams = []
                    for upstream in up_steps:
                        if upstream == 'reference':
                            if ref_prov is not None:
                                upstreams += ref_prov.upstreams
                        else:
                            upstreams.append(provs[upstream])

                    provs[step] = Provenance(
                        code_version=code_version,
                        process=step,
                        parameters=parameters,
                        upstreams=upstreams,
                        is_testing=is_testing,
                    )

                provs[step] = provs[step].merge_concurrent(session=session, commit=commit)

            if commit:
                session.commit()

            return provs


class ParsRefMaker(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.start_time = self.add_par(
            'start_time',
            None,
            (None, str, float, datetime.datetime),
            'Only use images taken after this time (inclusive). '
            'Time format can be MJD float, ISOT string, or datetime object. '
            'If None, will not limit the start time. ',
            critical=True,
        )

        self.end_time = self.add_par(
            'end_time',
            None,
            (None, str, float, datetime.datetime),
            'Only use images taken before this time (inclusive). '
            'Time format can be MJD float, ISOT string, or datetime object. '
            'If None, will not limit the end time. ',
            critical=True,
        )

        self.instrument = self.add_par(
            'instrument',
            None,
            (None, str, list),
            'Only use images from this instrument. If None, will not limit the instrument. '
            'If given as a list, will use any of the instruments in the list. '
            'Note that if "filter" is also given, it must match the number and order '
            'of the instruments given here. ',
            critical=True,
        )

        self.filter = self.add_par(
            'filter',
            None,
            (None, str, list),
            'Only use images with this filter. If None, will not limit the filter. '
            'If given as a list, will use any of the filters in the list. '
            'Note that if "instrument" is also given, it must match the number and order '
            'of the filters given here. ',  # TODO: allow a option to have multiple filters per instrument (use dict?)
            critical=True,
        )

        self.project = self.add_par(
            'project',
            None,
            (None, str, list),
            'Only use images from this project. If None, will not limit the project. '
            'If given as a list, will use any of the projects in the list. ',
            critical=True,
        )

        self.__image_query_pars__ = ['airmass', 'background', 'seeing', 'limmag', 'exp_time']

        for name in self.__image_query_pars__:
            for min_max in ['min', 'max']:
                self.add_limit_parameter(name, min_max)

        self.__docstrings__['min_limmag'] = ('Only use images with limmag larger (fainter) than this. '
                                             'If None, will not limit the minimal limmag. ')
        self.__docstrings__['max_limmag'] = ('Only use images with limmag smaller (brighter) than this. '
                                                'If None, will not limit the maximal limmag. ')

        self.min_number = self.add_par(
            'min_number',
            1,
            int,
            'Construct a reference only if there are at least this many images that pass all other criteria. ',
            critical=True,
        )

        self.max_number = self.add_par(
            'max_number',
            None,
            (None, int),
            'Construct a reference only if there are at most this many images that pass all other criteria. '
            'If None, will not limit the maximal number of images. ',
            critical=True,
        )

        self.seeing_quality_factor = self.add_par(
            'seeing_quality_factor',
            3.0,
            float,
            'linear combination coefficient for adding limiting magnitude and seeing FWHM '
            'when calculating the "image quality" used to rank images. ',
            critical=True,
        )

        self._enforce_no_new_attrs = True  # lock against new parameters

        self.override(kwargs)

    def add_limit_parameter(self, name, min_max='min'):
        """Add a parameter in a systematic way. """
        if min_max not in ['min', 'max']:
            raise ValueError('min_max must be either "min" or "max"')
        compare = 'larger' if min_max == 'min' else 'smaller'
        setattr(
            self,
            name,
            self.add_par(
                name,
                None,
                (None, float),
                f'Only use images with {name} {compare} than this value. '
                f'If None, will not limit the {min_max}imal {name}.',
                critical=True,
            )
        )

    def get_process_name(self):
        return 'referencing'


class RefMaker:
    def __init__(self, **kwargs):
        # first read the config file
        config = Config.get()
        cfg_dict = config.value('reference')
        cfg_dict.update(kwargs)  # user can provide override arguments

        # remove any parameters that deal with the pipeline first.
        pipe_dict = config.value('pipeline')  # this is the normal pipeline config
        pipe_dict.update(cfg_dict.pop('pipeline', {}))  # have an opportunity to override parameters
        self.pipeline = Pipeline(pipeline=pipe_dict)

        self.pars = ParsRefMaker(**cfg_dict)  # initialize without the pipeline parameters

        # first, make sure we can assemble the provenances up to extraction:
        self.im_provs = None  # the provenances used to make images going into the reference
        self.ex_provs = None  # the provenances used to make other products like SourceLists, that go into the reference
        self.ref_prov = None  # the provenance of the reference itself

        # these attributes tell us the place in the sky where we want to look for objects:
        self.ra = None  # in degrees
        self.dec = None  # in degrees
        self.target = None  # the name of the target / field ID / Object ID
        self.section_id = None  # a string with the section ID

    def setup_provenances(self):
        """Make the provenances for the images and all their products.

        These are used both to establish the provenance of the reference itself,
        and to look for images and associated products (like SourceLists) when
        building the reference.
        """
        instruments = self.pars.instrument
        if isinstance(instruments, str):
            instruments = [instruments]

        self.im_provs = []
        self.ex_provs = []

        for inst in instruments:
            # TODO: this assumes references are built up from regular images,
            #  each Image using a single Exposure. If in some weird future we'd
            #  like to build a reference from coadded images, this will fail.
            load_exposure = Exposure.make_provenance(inst)
            preprocessing = Provenance(
                process='preprocessing',
                code_version=load_exposure.code_version,  # TODO: allow loading versions for each process
                parameters=self.pipeline.preprocessor.pars.get_critical_pars(),
                upstreams=[load_exposure],
            )
            extraction = Provenance(
                process='extraction',
                code_version=preprocessing.code_version,  # TODO: allow loading versions for each process
                parameters=self.pipeline.extractor.pars.get_critical_pars(),  # includes parameters of siblings
                upstreams=[preprocessing],
            )

            # the exposure provenance is not included in the reference provenance's upstreams
            self.im_provs.append(preprocessing)
            self.ex_provs.append(extraction)

        upstreams = self.im_provs + self.ex_provs  # all the provenances that go into the reference
        self.ref_prov = Provenance(
            process='reference',
            code_version=load_exposure.code_version,
            parameters=self.pars.get_critical_pars(),
            upstreams=upstreams,
        )

    def parse_arguments(self, *args, **kwargs):
        """Figure out if the input parameters are given as coordinates or as target + section ID pairs.

        Possible combinations:
        - float + float: interpreted as RA/Dec in degrees
        - str + str: try to interpret as sexagesimal (RA as hours, Dec as degrees)
                     if it fails, will interpret as target + section ID
        # TODO: can we identify a reference with only a target/field ID without a section ID?
        Alternatively, can provide named arguments with the same combinations for either
        ra, dec or target, section_id.

        Returns
        -------
        session: sqlalchemy.orm.session.Session object or None
        """
        self.ra = None
        self.dec = None
        self.target = None
        self.section_id = None

        args, kwargs, session = parse_session(*args, **kwargs)  # first pick out any sessions

        if len(args) == 2:
            if isinstance(args[0], (float, int, np.number)) and isinstance(args[1], (float, int, np.number)):
                self.ra = float(args[0])
                self.dec = float(args[1])
            if isinstance(args[0], str) and isinstance(args[1], str):
                try:
                    self.ra = parse_sexigesimal_degrees(args[0], hours=True)
                    self.dec = parse_sexigesimal_degrees(args[1], hours=False)
                except ValueError:
                    self.target, self.section_id = args[0], args[1]
        else:
            raise ValueError('Invalid number of arguments given to RefMaker.parse_arguments()')

        return session

    def run(self, *args, **kwargs):
        """Check if a reference exists for the given coordiantes/field ID, and make it if it is missing.

        Arguments specifying where in the sky to look for / create the reference are parsed by parse_arguments().
        The remaining policy regarding which images to pick, and what provenance to use to find references,
        is defined by the parameters object of self and of self.pipeline.

        If one of the inputs is a session, will use that in the entire process.
        Otherwise, will open internal sessions and close them whenever they are not needed.

        Will return a Reference, or None in case it doesn't exist and cannot be created
        (e.g., because there are not enough images that pass the criteria).
        """
        session = self.parse_arguments(*args, **kwargs)

        self.setup_provenances()

        with SmartSession(session) as dbsession:
            # first merge the reference provenance
            self.ref_prov = self.ref_prov.merge_concurrent(session=dbsession, commit=True)

            # now look for the reference
            ref = Reference.get_reference(
                ra=self.ra,
                dec=self.dec,
                target=self.target,
                section_id=self.section_id,
                provenance_id=self.ref_prov,
                session=dbsession,
            )

            if ref is not None:
                return ref

            # no reference found, need to build one
            # first get all the images that could be used to build the reference
            instruments = self.pars.instrument
            if isinstance(instruments, str) or instruments is None:
                instruments = [instruments]
            filters = self.pars.filter
            if isinstance(filters, str) or filters is None:
                filters = [filters]
            projects = self.pars.project
            if isinstance(projects, str) or projects is None:
                projects = [projects]

            images = []  # can get images from different instrument+filter combinations
            # TODO: what about multiple filters per instrument?
            for inst, filt in zip(instruments, filters):
                prov = [p for p in self.im_provs if p.upstreams[0].parameters['instrument'] == inst]
                if len(prov) == 0:
                    raise RuntimeError(f'Cannot find a provenance for instrument "{inst}" in im_prov!')
                if len(prov) > 1:
                    raise RuntimeError(f'Found multiple provenances for instrument "{inst}" in im_prov!')
                prov = prov[0]

                query_pars = dict(
                    ra=self.ra,  # can be None!
                    dec=self.dec,  # can be None!
                    target=self.target,  # can be None!
                    section_id=self.section_id,  # can be None!
                    instrument=inst,  # can be None!
                    filter=filt,  # can be None!
                    provenance_id=prov.id,
                )

                for key in self.pars.__image_query_pars__:
                    for min_max in ['min', 'max']:
                        query_pars[f'{min_max}_{key}'] = getattr(self.pars, f'{min_max}_{key}')  # can be None!

                for project in projects:
                    query_pars['project'] = project  # can be None!
                    loaded_images = dbsession.scalars(Image.query_images(**query_pars)).all()
                    images += loaded_images

            if len(images) < self.pars.min_number:
                SCLogger.info(f'Found {len(images)} images, need at least {self.pars.min_number} to make a reference!')
                return None

            if len(images) > self.pars.max_number:
                coeff = abs(self.pars.seeing_quality_factor)  # abs is used to make sure the coefficient is negative
                for im in images:
                    im.quality = im.lim_mag_estimate - coeff * im.fwhm_estimate

                # sort the images by the quality
                images = sorted(images, key=lambda x: x.quality, reverse=True)
                images = images[:self.pars.max_number]

        # make the reference (note that we are out of the session block, to release it while we coadd)
        images = sorted(images, key=lambda x: x.mjd)  # sort the images in chronological order for coaddition
        coadd_pipeline = CoaddPipeline()
        coadd_image = coadd_pipeline.run(images)

        ref = Reference(image=coadd_image, provenance=self.ref_prov)

        with SmartSession(session) as dbsession:
            dbsession.add(ref)
            dbsession.commit()

        return ref
