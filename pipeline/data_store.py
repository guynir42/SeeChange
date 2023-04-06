import sqlalchemy as sa

from pipeline.utils import get_git_hash, get_latest_provenance

from models.base import SmartSession
from models.provenance import CodeHash, CodeVersion, Provenance
from models.exposure import Exposure
from models.image import Image, SubtractionImage
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.cutouts import Cutouts
from models.measurements import Measurements


UPSTREAM_NAMES = {
    'preprocessing': [],
    ''
    'astrometry': ['preprocessing'],
    'calibration': ['preprocessing', 'astrometry'],
    'subtraction': ['preprocessing', 'astrometry', 'calibration'],
    'detection': ['subtraction'],
    'extraction': ['detection'],
}

UPSTREAM_OBJECTS = {
    'preprocessing': 'image',
    'coaddition': 'image',
    'astrometry': 'wcs',
    'calibration': 'zp',
    'subtraction': 'sub_image',
    'detection': 'cutouts',
    'extraction': 'measurements',
}


class DataStore:
    """
    Create this object to parse user inputs and identify which data products need
    to be fetched from the database, and keep a cached version of the products for
    use downstream in the pipeline.
    """
    @staticmethod
    def from_args(*args, **kwargs):
        """
        Create a DataStore object from the given arguments.
        """
        if len(args) == 0:
            raise ValueError('No arguments given to DataStore constructor!')
        if len(args) == 1 and isinstance(args[0], DataStore):
            return args[0]
        else:
            return DataStore(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        """
        See the parse_args method for details on how to initialize this object.
        """
        # these are data products that can be cached in the store
        self.exposure = None  # single image, entire focal plane
        self.image = None  # single image from one CCD
        self.wcs = None  # astrometric solution
        self.zp = None  # photometric calibration
        self.ref_image = None  # to be used to make subtractions
        self.sub_image = None  # subtracted image
        self.cutouts = None  # cutouts around sources
        self.measurements = None  # photometry and other measurements for each source

        self.upstream_provs = None  # provenances to override the upstreams if no upstream objects exist

        self.session = None  # optional database session

        # these are identifiers used to find the data products in the database
        self.exp_id = None  # use this and ccd_id to find the raw image
        self.ccd_id = None  # use this and exp_id to find the raw image
        self.im_id = None  # use this to specify an image already in the database

        self.parse_args(*args, **kwargs)

    def parse_args(self, *args, **kwargs):
        """
        Parse the arguments to the DataStore constructor.
        Can initialize based on exposure and CCD ids,
        or give a specific image id or coadd id.

        Parameters
        ----------
        args: list
            A list of arguments to parse.
            Possible argument combinations are:
            - exp_id, ccd_id: give two integers
            - im_id: give a single integer

        kwargs: dict
            A dictionary of keyword arguments to parse.
            Using named arguments allows the user to
            explicitly assign the values to the correct
            attributes. These are parsed after the args
            list and can override it!

        """
        if len(args) == 0:
            raise ValueError('Must provide at least one argument to DataStore constructor.')

        if len(args) == 1 and isinstance(args[0], DataStore):
            # if the only argument is a DataStore, copy it
            self.__dict__ = args[0].__dict__.copy()
            return

        # catch any sessions given to the args list
        sessions = [arg for arg in args if isinstance(arg, (sa.orm.session.Session, SmartSession))]
        if len(sessions) > 0:
            self.session = sessions[-1]
        args = [arg for arg in args if not isinstance(arg, (sa.orm.session.Session, SmartSession))]

        # remove any provenances from the args list
        for arg in args:
            if isinstance(arg, Provenance):
                self.upstream_provs.append(arg)
        args = [arg for arg in args if not isinstance(arg, Provenance)]

        # parse the args list
        arg_types = [type(arg) for arg in args]
        if arg_types == [int, int]:  # exp_id, ccd_id
            self.exp_id, self.ccd_id = args
        elif arg_types == [int]:
            self.im_id = args[0]
        # TODO: add more options here
        #  example: get a string filename to parse a specific file on disk
        else:
            raise ValueError(
                'Invalid arguments to DataStore constructor, '
                f'got {arg_types}. '
                f'Expected [int, int] or [int]'
            )

        # parse the kwargs dict
        for key, val in kwargs.items():
            # override these attributes explicitly
            if key in ['exp_id', 'ccd_id', 'im_id', 'coadd_id']:
                if not isinstance(val, int):
                    raise ValueError(f'{key} must be an integer, got {type(val)}')
                setattr(self, key, val)

            # check for provenances
            if key in ['prov', 'provenances', 'upstream_provs', 'upstream_provenances']:
                new_provs = val
                if not isinstance(new_provs, list):
                    new_provs = [new_provs]

                for prov in new_provs:
                    if not isinstance(prov, Provenance):
                        raise ValueError(f'Provenance must be a Provenance object, got {type(prov)}')
                    self.upstream_provs.append(prov)

            # check for sessions
            if key in ['session']:
                if not isinstance(val, (sa.orm.session.Session, SmartSession)):
                    raise ValueError(f'Session must be a SQLAlchemy session or SmartSession, got {type(val)}')
                self.session = val

    def __setattr__(self, key, value):
        """
        Check some of the inputs before saving them.
        """

        if key in ['exp_id', 'ccd_id', 'im_id'] and not isinstance(value, int):
            raise ValueError(f'{key} must be an integer, got {type(value)}')

        if key == 'image' and not isinstance(value, Image):
            raise ValueError(f'Image must be an Image object, got {type(value)}')

        if key == 'wcs' and not isinstance(value, WorldCoordinates):
            raise ValueError(f'WCS must be a WorldCoordinates object, got {type(value)}')

        if key == 'zp' and not isinstance(value, ZeroPoint):
            raise ValueError(f'ZP must be a ZeroPoint object, got {type(value)}')

        if key == 'ref_image' and not isinstance(value, Image):
            raise ValueError(f'Reference image must be an Image object, got {type(value)}')

        if key == 'sub_image' and not isinstance(value, SubtractionImage):
            raise ValueError(f'Subtraction image must be a SubtractionImage object, got {type(value)}')

        if key == 'cutouts' and not isinstance(value, list):
            raise ValueError(f'cutouts must be a list of Cutout objects, got {type(value)}')

        if key == 'cutouts' and not all([isinstance(c, Cutouts) for c in value]):
            raise ValueError(f'cutouts must be a list of Cutouts objects, got list with {[type(c) for c in value]}')

        if key == 'measurements' and not isinstance(value, list):
            raise ValueError(f'measurements must be a list of Measurements objects, got {type(value)}')

        if key == 'measurements' and not all([isinstance(m, Measurements) for m in value]):
            raise ValueError(
                f'measurements must be a list of Measurement objects, got list with {[type(m) for m in value]}'
            )

        if key == 'upstream_provs' and not isinstance(value, list):
            raise ValueError(f'upstream_provs must be a list of Provenance objects, got {type(value)}')

        if key == 'upstream_provs' and not all([isinstance(p, Provenance) for p in value]):
            raise ValueError(
                f'upstream_provs must be a list of Provenance objects, got list with {[type(p) for p in value]}'
            )

        if key == 'session' and not isinstance(value, (sa.orm.session.Session, SmartSession)):
            raise ValueError(f'Session must be a SQLAlchemy session or SmartSession, got {type(value)}')

        super().__setattr__(key, value)

    def get_inputs(self):
        """Get a string with the relevant inputs. """

        if self.im_id is not None:
            return f'im_id={self.im_id}'
        elif self.exp_id is not None and self.ccd_id is not None:
            return f'exp_id={self.exp_id}, ccd_id={self.ccd_id}'
        else:
            raise ValueError('Could not get inputs for DataStore.')

    def get_provenance(self, process, pars_dict, upstream_provs=None, session=None):
        """
        Get the provenance for a given process.

        Parameters
        ----------
        process: str
            The name of the process, e.g., "preprocess", "calibration", "subtraction".
            Use a Parameter object's get_process_name().
        pars_dict: dict
            A dictionary of parameters used for the process.
            These include the critical parameters for this process.
            Use a Parameter object's get_critical_pars().
        upstream_provs: list of Provenance objects
            A list of provenances to use as upstreams for the current
            provenance that is requested. Any upstreams that are not
            given will be filled using objects that already exist
            in the data store, or by getting the most up-to-date
            provenance from the database.
            The upstream provenances can be given directly as
            a function parameter, or using the DataStore constructor.
            If given as a parameter, it will override the DataStore's
            self.upstream_provs attribute for that call.
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.

        Returns
        -------
        prov: Provenance
            The provenance for the given process.

        """
        if upstream_provs is None:
            upstream_provs = self.upstream_provs

        with SmartSession(session) as session:
            # check if this code version exists
            code_hash = session.scalars(sa.select(CodeHash).where(CodeHash.hash == get_git_hash())).first()
            if code_hash is None:
                raise ValueError('Cannot find code hash!')

            # check if we can find the upstream provenances
            upstreams = []
            for name in self.UPSTREAM_NAMES[process]:
                obj = getattr(self, UPSTREAM_OBJECTS[name], None)

                # first try to load an upstream that was given explicitly:
                if name in [p.process for p in upstream_provs]:
                    prov = [p for p in upstream_provs if p.process == name][0]

                # second, try to get a provenance from objects saved to the store:
                elif obj is not None and hasattr(obj, 'provenance') and obj.provenance is not None:
                    prov = obj.provenance

                # last, try to get the latest provenance from the database:
                else:
                    prov = get_latest_provenance(name, session=session)

                # can't find any provenance upstream, therefore
                # there can't be any provenance for this process
                if prov is None:
                    return None

                upstreams.append(prov)

            # we have a code version object and upstreams, we can make a provenance
            prov = Provenance(
                process=process,
                code_version=code_hash.code_version,
                parameters=pars_dict,
                upstreams=upstreams,
            )
            prov.update_hash()  # need a new object to calculate the hash, then check if it exists on the DB:
            existing_p = session.scalars(
                sa.select(Provenance).where(
                    Provenance.unique_hash == prov.unique_hash
                )
            ).first()

            if existing_p is not None:
                prov = existing_p

        return prov

    def _get_provenance_fallback(self, process, session=None):
        """
        Get the provenance for a given process, without knowing
        the parameters or code version.
        This simply looks for a matching provenance in the upstream_provs
        attribute, and if it is not there, it will call the latest provenance
        from the database.

        This will raise if no provenance can be found.
        """
        # see if it is in the upstream_provs
        prov_list = [p for p in self.upstream_provs if p.process == process]
        provenance = prov_list[0] if len(prov_list) > 0 else None

        # try getting the latest from the database
        if provenance is None:  # check latest provenance
            provenance = get_latest_provenance(process, session=session)
        if provenance is None:  # still can't find anything!
            raise ValueError(f'Cannot find the "{process}" provenance!')

        return provenance

    def get_raw_exposure(self, session=None):
        """
        Get the raw exposure from the database.
        """
        if self.exposure is None:
            if self.exp_id is None:
                raise ValueError('Cannot get raw exposure without exp_id!')

            with SmartSession(session) as session:
                self.exposure = session.scalars(sa.select(Exposure).where(Exposure.id == self.exp_id)).first()

        return self.exposure

    def get_image(self, provenance=None, session=None):
        """
        Get the pre-processed (or coadded) image, either from
        memory or from the database.
        If the store is initialized with an im_id,
        that image is returned, no matter the
        provenances or the local parameters.
        This is the only way to ask for a coadd image.
        If an image with such an id is not found,
        in memory or in the database, will raise
        an ValueError.
        If exp_id and ccd_id are given, will
        load an image that is consistent with
        that exposure and CCD ids, and also with
        the code version and critical parameters
        (using a matching of provenances).
        In this case we will only load a regular
        image, not a coadded image.
        If no matching image is found, will return None.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the image.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "preprocessing" process.
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.

        Returns
        -------
        image: Image object
            The image object, or None if no matching image is found.

        """
        # we were explicitly asked for a specific image id:
        if self.im_id is not None:
            if self.image is not None and isinstance(self.image, Image) and self.image.id == self.im_id:
                pass  # return self.image at the end of function...
            else:  # not found in local memory, get from DB
                with SmartSession(session) as session:
                    self.image = session.scalars(sa.select(Image).where(Image.id == self.im_id)).first()

            # we asked for a specific image, it should exist!
            if self.image is None:
                raise ValueError(f'Cannot find image with id {self.im_id}!')

        # this option is for when we are not sure which image id we need
        elif self.exp_id is not None and self.ccd_id is not None:

            # must compare the image to the current provenance
            if provenance is None:  # check if in upstream_provs/database
                provenance = self._get_provenance_fallback('preprocessing', session=session)

            if (
                self.image is not None and isinstance(self.image, Image)
                and self.image.exp_id == self.exp_id and self.image.ccd_id == self.ccd_id
            ):
                # make sure the image has the correct provenance
                if self.image is not None:
                    if self.image.provenance is None:
                        raise ValueError('Image has no provenance!')

                    # a mismatch of provenance and cached image:
                    if self.image.provenance.unique_hash != provenance.unique_hash:
                        self.image = None

            if self.image is None:  # load from DB
                with SmartSession(session) as session:
                    self.image = session.scalars(
                        sa.select(Image).where(
                            Image.exp_id == self.exp_id,
                            Image.ccd_id == self.ccd_id,
                            Image.provenance.has(unique_hash=provenance.unique_hash)
                        )
                    ).first()

        else:
            raise ValueError('Cannot get processed image without exp_id and ccd_id or im_id!')

        return self.image  # could return none if no image was found

    def get_wcs(self, provenance=None, session=None):
        """
        Get an astrometric solution (in the form of a WorldCoordinates),
        either from memory or from database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the wcs.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "astrometry" process.
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.

        Returns
        -------
        wcs: WorldCoordinates object
            The WCS object, or None if no matching WCS is found.

        """
        # not in memory, look for it on the DB
        if self.wcs is None:
            image = self.get_image(session=session)

            if provenance is None:  # check if in upstream_provs/database
                provenance = self._get_provenance_fallback('astrometry', session=session)

                # make sure the wcs has the correct provenance
                if self.wcs is not None:
                    if self.wcs.provenance is None:
                        raise ValueError('WorldCoordinates has no provenance!')

                    # a mismatch of provenance and cached image:
                    if self.wcs.provenance.unique_hash != provenance.unique_hash:
                        self.wcs = None

                if self.wcs is None:
                    with SmartSession(session) as session:
                        self.wcs = session.scalars(
                            sa.select(WorldCoordinates).where(
                                WorldCoordinates.image_id == image.id,
                                WorldCoordinates.provenance.has(unique_hash=provenance.unique_hash),
                            )
                        ).first()

        return self.wcs







