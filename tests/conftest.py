import os
import io
import warnings
import pytest
import uuid
import shutil
import pathlib

import requests
from bs4 import BeautifulSoup

import numpy as np

import sqlalchemy as sa

from astropy.io import fits, votable

from util.config import Config
from models.base import FileOnDiskMixin, SmartSession, CODE_ROOT, _logger
from models.provenance import CodeVersion, Provenance
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from pipeline.data_store import DataStore
from pipeline.catalog_tools import fetch_GaiaDR3_excerpt

from util import config
from util.archive import Archive
from util.retrydownload import retry_download

pytest_plugins = ['tests.fixtures.simulated', 'tests.fixtures.decam', 'tests.fixtures.ztf', 'tests.fixtures.ptf']


# idea taken from: https://shay-palachy.medium.com/temp-environment-variables-for-pytest-7253230bd777
# this fixture should be the first thing loaded by the test suite
@pytest.fixture(scope="session", autouse=True)
def tests_setup_and_teardown():
    # Will be executed before the first test
    # print('Initial setup fixture loaded! ')

    # make sure to load the test config
    test_config_file = str((pathlib.Path(__file__).parent.parent / 'tests' / 'seechange_config_test.yaml').resolve())

    Config.get(configfile=test_config_file, setdefault=True)

    yield
    # Will be executed after the last test
    # print('Final teardown fixture executed! ')

    with SmartSession() as session:
        # Tests are leaving behind (at least) exposures and provenances.
        # Ideally, they should all clean up after themselves.  Finding
        # all of this is a worthwhile TODO, but recursive_merge probably
        # means that finding all of them is going to be a challenge.
        # So, make sure that the database is wiped.  Deleting just
        # provenances and codeversions should do it, because most things
        # have a cascading foreign key into provenances.
        session.execute( sa.text( "DELETE FROM provenances" ) )
        session.execute( sa.text( "DELETE FROM code_versions" ) )
        session.commit()


# data that is included in the repo and should be available for tests
@pytest.fixture(scope="session")
def persistent_dir():
    return os.path.join(CODE_ROOT, 'data')


# this is a cache folder that should survive between test runs
@pytest.fixture(scope="session")
def cache_dir():
    path = os.path.join(CODE_ROOT, 'data/cache')
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


# this will be configured to FileOnDiskMixin.local_path, and used as temporary data location
@pytest.fixture(scope="session")
def data_dir():
    FileOnDiskMixin.configure_paths()
    temp_data_folder = FileOnDiskMixin.local_path
    os.makedirs(temp_data_folder, exist_ok=True)
    with open(os.path.join(temp_data_folder, 'placeholder'), 'w'):
        pass  # make an empty file inside this folder to make sure it doesn't get deleted on "remove_data_from_disk"

    # print(f'temp_data_folder: {temp_data_folder}')

    yield temp_data_folder

    # remove all the files created during tests
    # make sure the test config is pointing the data_dir
    # to a different location than the rest of the data
    # shutil.rmtree(temp_data_folder)


@pytest.fixture(scope="session")
def blocking_plots():
    """
    Control how and when plots will be generated.
    There are three options for the environmental variable "INTERACTIVE".
     - It is not set: do not make any plots. blocking_plots returns False.
     - It is set to a False value: make plots, but save them, and do not show on screen/block execution.
       In this case the blocking_plots returns False, but the tests that skip if INTERACTIVE is None will run.
     - It is set to a True value: make the plots, but stop the test execution until the figure is closed.

    If a test only makes plots and does not test functionality, it should be marked with
    @pytest.mark.skipif( os.getenv('INTERACTIVE') is None, reason='Set INTERACTIVE to run this test' )

    If a test makes a diagnostic plot, that is only ever used to visually inspect the results,
    then it should be surrounded by an if blocking_plots: statement. It will only run in interactive mode.

    If a test makes a plot that should be saved to disk, it should either have the skipif mentioned above,
    or have an if os.getenv('INTERACTIVE'): statement surrounding the plot itself.
    You may want to add plt.show(block=blocking_plots) to allow the figure to stick around in interactive mode,
    on top of saving the figure at the end of the test.
    """
    import matplotlib
    backend = matplotlib.get_backend()

    # make sure there's a folder to put the plots in
    if not os.path.isdir(os.path.join(CODE_ROOT, 'tests/plots')):
        os.makedirs(os.path.join(CODE_ROOT, 'tests/plots'))

    inter = os.getenv('INTERACTIVE', False)
    if isinstance(inter, str):
        inter = inter.lower() in ('true', '1', 'on', 'yes')

    if not inter:  # for non-interactive plots, use headless plots that just save to disk
        # ref: https://stackoverflow.com/questions/15713279/calling-pylab-savefig-without-display-in-ipython
        matplotlib.use("Agg")

    yield inter

    matplotlib.use(backend)


def rnd_str(n):
    return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n))


@pytest.fixture
def config_test():
    return Config.get()


@pytest.fixture(scope="session", autouse=True)
def code_version():
    with SmartSession() as session:
        cv = session.scalars(sa.select(CodeVersion).where(CodeVersion.id == 'test_v1.0.0')).first()
        if cv is None:
            cv = CodeVersion(id="test_v1.0.0")
            cv.update()
            session.add( cv )
            session.commit()
        cv = session.scalars(sa.select(CodeVersion).where(CodeVersion.id == 'test_v1.0.0')).first()

    yield cv

    try:
        with SmartSession() as session:
            session.execute(sa.delete(CodeVersion).where(CodeVersion.id == 'test_v1.0.0'))
            session.commit()
    except Exception as e:
        warnings.warn(str(e))


@pytest.fixture
def provenance_base(code_version):
    p = Provenance(
        process="test_base_process",
        code_version=code_version,
        parameters={"test_key": uuid.uuid4().hex},
        upstreams=[],
        is_testing=True,
    )

    with SmartSession() as session:
        p.code_version = session.merge(code_version)
        session.add(p)
        session.commit()
        session.refresh(p)
        pid = p.id

    yield p

    try:
        with SmartSession() as session:
            session.execute(sa.delete(Provenance).where(Provenance.id == pid))
            session.commit()
    except Exception as e:
        warnings.warn(str(e))


@pytest.fixture
def provenance_extra( provenance_base ):
    p = Provenance(
        process="test_base_process",
        code_version=provenance_base.code_version,
        parameters={"test_key": uuid.uuid4().hex},
        upstreams=[provenance_base],
        is_testing=True,
    )
    p.update_id()

    with SmartSession() as session:
        session.add(p)
        session.commit()
        session.refresh(p)
        pid = p.id

    yield p

    try:
        with SmartSession() as session:
            session.execute(sa.delete(Provenance).where(Provenance.id == pid))
            session.commit()
    except Exception as e:
        warnings.warn(str(e))


# use this to make all the pre-committed Image fixtures
@pytest.fixture(scope="session")
def provenance_preprocessing(code_version):
    p = Provenance(
        process="preprocessing",
        code_version=code_version,
        parameters={"test_key": "test_value"},
        upstreams=[],
        is_testing=True,
    )

    with SmartSession() as session:
        p.code_version = session.merge(code_version)
        session.add(p)
        session.commit()
        session.refresh(p)
        pid = p.id

    yield p

    try:
        with SmartSession() as session:
            session.execute(sa.delete(Provenance).where(Provenance.id == pid))
            session.commit()

    except Exception as e:
        warnings.warn(str(e))


@pytest.fixture
def archive():
    cfg = config.Config.get()
    archive_specs = cfg.value('archive')
    if archive_specs is None:
        raise ValueError( "archive in config is None" )
    archive = Archive( **archive_specs )
    yield archive

    try:
        # To tear down, we need to blow away the archive server's directory.
        # For the test suite, we've also mounted that directory locally, so
        # we can do that
        archivebase = f"{os.getenv('SEECHANGE_TEST_ARCHIVE_DIR')}/{cfg.value('archive.path_base')}"
        try:
            shutil.rmtree( archivebase )
        except FileNotFoundError:
            pass

    except Exception as e:
        warnings.warn(str(e))


@pytest.fixture
def example_image_with_sources_and_psf_filenames():
    image = pathlib.Path( FileOnDiskMixin.local_path ) / "test_data/test_ztf_image.fits"
    weight = pathlib.Path( FileOnDiskMixin.local_path ) / "test_data/test_ztf_image.weight.fits"
    flags = pathlib.Path( FileOnDiskMixin.local_path ) / "test_data/test_ztf_image.flags.fits"
    sources = pathlib.Path( FileOnDiskMixin.local_path ) / "test_data/test_ztf_image.sources.fits"
    psf = pathlib.Path( FileOnDiskMixin.local_path ) / "test_data/test_ztf_image.psf"
    psfxml = pathlib.Path( FileOnDiskMixin.local_path ) / "test_data/test_ztf_image.psf.xml"
    return image, weight, flags, sources, psf, psfxml


@pytest.fixture
def example_ds_with_sources_and_psf( example_image_with_sources_and_psf_filenames ):
    image, weight, flags, sources, psf, psfxml = example_image_with_sources_and_psf_filenames
    ds = DataStore()

    ds.image = Image( filepath=str( image.relative_to( FileOnDiskMixin.local_path ) ), format='fits' )
    with fits.open( image ) as hdul:
        ds.image._data = hdul[0].data
        ds.image._raw_header = hdul[0].header
    with fits.open( weight ) as hdul:
        ds.image._weight = hdul[0].data
    with fits.open( flags ) as hdul:
        ds.image_flags = hdul[0].data
    ds.image.set_corners_from_header_wcs()
    ds.image.ra = ( ds.image.ra_corner_00 + ds.image.ra_corner_01 +
                    ds.image.ra_corner_10 + ds.image.ra_corner_11 ) / 4.
    ds.image.dec = ( ds.image.dec_corner_00 + ds.image.dec_corner_01 +
                     ds.image.dec_corner_00 + ds.image.dec_corner_11 ) / 4.
    ds.image.calculate_coordinates()

    ds.sources = SourceList( filepath=str( sources.relative_to( FileOnDiskMixin.local_path ) ), format='sextrfits' )
    ds.sources.load( sources )
    ds.sources.num_sources = len( ds.sources.data )

    ds.psf = PSF( filepath=str( psf.relative_to( FileOnDiskMixin.local_path ) ), format='psfex' )
    ds.psf.load( download=False, psfpath=psf, psfxmlpath=psfxml )
    bio = io.BytesIO( ds.psf.info.encode( 'utf-8' ) )
    tab = votable.parse( bio ).get_table_by_index( 1 )
    ds.psf.fwhm_pixels = float( tab.array['FWHM_FromFluxRadius_Mean'][0] )

    return ds


@pytest.fixture
def example_source_list_filename( example_image_with_sources_and_psf_filenames ):
    image, weight, flags, sources, psf, psfxml = example_image_with_sources_and_psf_filenames
    return sources


@pytest.fixture
def example_psfex_psf_files():
    psfpath = ( pathlib.Path( FileOnDiskMixin.local_path )
                / "test_data/ztf_20190317307639_000712_zg_io.083_sources.psf" )
    psfxmlpath = ( pathlib.Path( FileOnDiskMixin.local_path )
                   / "test_data/ztf_20190317307639_000712_zg_io.083_sources.psf.xml" )
    if not ( psfpath.is_file() and psfxmlpath.is_file() ):
        raise FileNotFoundError( f"Can't read at least one of {psfpath}, {psfxmlpath}" )
    return psfpath, psfxmlpath


@pytest.fixture
def gaiadr3_excerpt( example_ds_with_sources_and_psf ):
    ds = example_ds_with_sources_and_psf
    catexp = fetch_GaiaDR3_excerpt( ds.image, minstars=50, maxmags=20, magrange=4)
    assert catexp is not None

    yield catexp

    with SmartSession() as session:
        catexp = catexp.recursive_merge( session )
        catexp.delete_from_disk_and_database( session=session )


