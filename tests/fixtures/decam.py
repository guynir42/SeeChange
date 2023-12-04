import pytest
import os
import re
import wget
import hashlib
import yaml
import subprocess
import shutil
import pathlib
import uuid

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS

from models.base import SmartSession, FileOnDiskMixin, _logger
from models.instrument import Instrument, get_instrument_instance
from models.decam import DECam  # need this import to make sure DECam is added to the Instrument list
from models.provenance import Provenance
from models.exposure import Exposure
from models.image import Image
from models.datafile import DataFile
from models.source_list import SourceList
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint

from pipeline.data_store import DataStore
from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator

from util.retrydownload import retry_download
from util.exceptions import SubprocessFailure


# Get the flat, fringe, and linearity for
# a couple of DECam chips and filters
# Need session scope; otherwise, things
# get mixed up when _get_default_calibrator
# is called from within another function.
@pytest.fixture( scope='session' )
def decam_default_calibrators(cache_dir, data_dir):
    # try to get the calibrators from the cache folder
    if os.path.isdir(os.path.join(cache_dir, 'DECam_default_calibrators')):
        shutil.copytree(
            os.path.join(cache_dir, 'DECam_default_calibrators'),
            os.path.join(data_dir, 'DECam_default_calibrators'),
            dirs_exist_ok=True,
        )

    decam = get_instrument_instance( 'DECam' )
    sections = [ 'N1', 'S1' ]
    filters = [ 'r', 'i', 'z', 'g']
    for sec in sections:
        for calibtype in [ 'flat', 'fringe' ]:
            for filt in filters:
                decam._get_default_calibrator( 60000, sec, calibtype=calibtype, filter=filt )
    decam._get_default_calibrator( 60000, sec, calibtype='linearity' )

    # store the calibration files in the cache folder
    if not os.path.isdir(os.path.join(cache_dir, 'DECam_default_calibrators')):
        os.makedirs(os.path.join(cache_dir, 'DECam_default_calibrators'), exist_ok=True)
    for folder in os.listdir(os.path.join(data_dir, 'DECam_default_calibrators')):
        if not os.path.isdir(os.path.join(cache_dir, 'DECam_default_calibrators', folder)):
            os.makedirs(os.path.join(cache_dir, 'DECam_default_calibrators', folder), exist_ok=True)
        for file in os.listdir(os.path.join(data_dir, 'DECam_default_calibrators', folder)):
            shutil.copy2(
                os.path.join(data_dir, 'DECam_default_calibrators', folder, file),
                os.path.join(cache_dir, 'DECam_default_calibrators', folder, file)
            )

    yield sections, filters

    imagestonuke = set()
    datafilestonuke = set()
    with SmartSession() as session:
        for sec in [ 'N1', 'S1' ]:
            for filt in [ 'r', 'i', 'z', 'g' ]:
                info = decam.preprocessing_calibrator_files( 'externally_supplied', 'externally_supplied',
                                                             sec, filt, 60000, nofetch=True, session=session )
                for filetype in [ 'zero', 'flat', 'dark', 'fringe', 'illumination', 'linearity' ]:
                    if ( f'{filetype}_fileid' in info ) and ( info[ f'{filetype}_fileid' ] is not None ):
                        if info[ f'{filetype}_isimage' ]:
                            imagestonuke.add( info[ f'{filetype}_fileid' ] )
                        else:
                            datafilestonuke.add( info[ f'{filetype}_fileid' ] )
        for imid in imagestonuke:
            im = session.get( Image, imid )
            im.delete_from_disk_and_database( session=session, commit=False )
        for dfid in datafilestonuke:
            df = session.get( DataFile, dfid )
            df.delete_from_disk_and_database( session=session, commit=False )
        session.commit()


@pytest.fixture(scope='module')
def decam_reduced_origin_exposures():
    decam = DECam()
    yield decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                       proposals='2023A-716082',
                                       skip_exposures_in_database=False,
                                       proc_type='instcal' )


@pytest.fixture(scope='module')
def decam_raw_origin_exposures():
    decam = DECam()
    yield decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                       proposals='2023A-716082',
                                       skip_exposures_in_database=False,
                                       proc_type='raw' )


@pytest.fixture(scope="session")
def decam_filename(data_dir, cache_dir):
    """Pull a DECam exposure down from the NOIRLab archives.

    Because this is a slow process (depending on the NOIRLab archive
    speed, it can take up to minutes), first look for this file
    in the cache_dir, and if it exists, and copy it. If not,
    actually download the image from NOIRLab into the cache_dir,
    and create a symlink to the temp_dir. That way, until the
    user manually deletes the cached file, we won't have to redo the
    slow NOIRLab download again.
    """
    base_name = 'c4d_221104_074232_ori.fits.fz'
    filename = os.path.join(data_dir, base_name)
    if not os.path.isfile(filename):
        cachedfilename = os.path.join(cache_dir, 'DECam', base_name)
        os.makedirs(os.path.dirname(cachedfilename), exist_ok=True)

        if not os.path.isfile(cachedfilename):
            url = 'https://astroarchive.noirlab.edu/api/retrieve/004d537b1347daa12f8361f5d69bc09b/'
            response = wget.download(url=url, out=cachedfilename)
            assert response == cachedfilename

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        shutil.copy2(cachedfilename, filename)

    return filename


@pytest.fixture(scope="session")
def decam_exposure(decam_filename, data_dir):
    filename = decam_filename

    with fits.open( filename, memmap=True ) as ifp:
        hdr = ifp[0].header
    exphdrinfo = Instrument.extract_header_info( hdr, [ 'mjd', 'exp_time', 'filter', 'project', 'target' ] )

    exposure = Exposure( filepath=filename, instrument='DECam', **exphdrinfo )
    exposure.save()  # save to archive and get an MD5 sum

    yield exposure

    # Just in case this exposure got loaded into the database
    exposure.delete_from_disk_and_database()


@pytest.fixture
def decam_raw_image( decam_exposure ):
    image = Image.from_exposure(decam_exposure, section_id='N1')
    image.data = image.raw_data.astype(np.float32)

    yield image

    image.delete_from_disk_and_database()


@pytest.fixture
def decam_datastore(
        datastore_factory,
        cache_dir,
        decam_exposure,
        decam_default_calibrators,  # not used directly, but makes sure this is pre-fetched from cache
):
    """Provide a datastore with all the products based on the DECam exposure

    To use this data store in a test where new data is to be generated,
    simply change the pipeline object's "test_parameter" value to a unique
    new value, so the provenance will not match and the data will be regenerated.

    EXAMPLE
    -------
    extractor.pars.test_parameter = uuid.uuid().hex
    extractor.run(datastore)
    assert extractor.has_recalculated is True
    """
    cache_dir = os.path.join(cache_dir, 'DECam')
    ds = datastore_factory(decam_exposure, 'N1', cache_dir, '115/c4d_20221104_074232_N1_g_Sci_HI4PX4')

    yield ds

    # cleanup
    if 'ds' in locals():
        ds.delete_everything()


@pytest.fixture
def decam_example_reduced_image_ds( code_version, decam_exposure ):
    """Provides a datastore with an image, source list, and psf.

    Preprocessing, source extraction, and PSF estimation were all
    performed as this fixture was written, and saved to
    data/test_data/DECam_examples, so that this fixture will return
    quickly. That does mean that if the code has evolved since those
    test files were created, the files may not be exactly consistent
    with the code_version in the provenances they have attached to them,
    but hopefully the actual data and database records are still good
    enough for the tests.

    Has an image (with weight and flags), sources, and psf. The data
    has not been loaded into the image, sources, or psf fields, but of
    course that will happen if you access (for instance)
    decam_example_reduced_image_ds.image.data. The provenances *have*
    been loaded into the session (and committed to the database), but
    the image, sources, and psf have not been added to the session.

    The DataStore has a session inside it.

    """

    exposure = decam_exposure
    # Have to spoof the md5sum field to let us add it to the database even
    # though we're not really saving it to the archive
    exposure.md5sum = uuid.uuid4()

    datadir = pathlib.Path( FileOnDiskMixin.local_path )
    filepathbase = 'test_data/DECam_examples/c4d_20221104_074232_N1_g_Sci_VWQNR2'
    fileextensions = { '.image.fits', '.weight.fits', '.flags.fits', '.sources.fits', '.psf', '.psf.xml' }

    # This next block of code generates the files read in this test.  We
    # don't want to rerun the code to generate them every single time,
    # because it's a fairly slow process and this is a function-scope
    # fixture.  As such, generate files with names ending in "_cached"
    # that we won't delete at fixture teardown.  Then, we'll copy those
    # files to the ones that the fixture really needs, and those copies
    # will be deleted at fixture teardown.  (We need this to be a
    # function-scope fixture because the tests that use the DataStore we
    # return may well modify it, and may well modify the files on disk.)

    # (Note: this will not regenerate the .yaml files
    # test_data/DECam_examples, which are used to reconstruct the
    # database object fields.  If tests fail after regenerating these
    # files, review those .yaml files to make sure they're still
    # current.)

    # First check if those files exist; if they don't, generate them.
    mustregenerate = False
    for ext in fileextensions:
        if not ( datadir / f'{filepathbase}{ext}_cached' ).is_file():
            _logger.info( f"{filepathbase}{ext}_cached missing, decam_example_reduced_image_ds fixture "
                          f"will regenerate all _cached files" )
            mustregenerate = True

    if mustregenerate:
        with SmartSession() as session:
            exposure = exposure.recursive_merge( session )
            session.add( exposure )
            session.commit()              # This will get cleaned up in the decam_exposure teardown
            prepper = Preprocessor()
            ds = prepper.run( exposure, 'N1', session=session )
            try:
                det = Detector( measure_psf=True )
                ds = det.run( ds )
                ds.save_and_commit()

                paths = []
                for obj in [ ds.image, ds.sources, ds.psf ]:
                    paths.extend( obj.get_fullpath( as_list=True ) )

                extextract = re.compile( '^(?P<base>.*)(?P<extension>\\..*\\.fits|\\.psf|\\.psf.xml)$' )
                extscopied = set()
                for src in paths:
                    match = extextract.search( src )
                    if match is None:
                        raise RuntimeError( f"Failed to parse {src}" )
                    if match.group('extension') not in fileextensions:
                        raise RuntimeError( f"Unexpected file extension on {src}" )
                    shutil.copy2( src, datadir/ f'{filepathbase}{match.group("extension")}_cached' )
                    extscopied.add( match.group('extension') )

                if extscopied != fileextensions:
                    raise RuntimeError( f"Extensions copied {extcopied} doesn't match expected {extstocopy}" )
            finally:
                ds.delete_everything()
    else:
        _logger.info( f"decam_example_reduced_image_ds fixture found all _cached files, not regenerating" )

    # Now make sure that the actual files needed are there by copying
    # the _cached files

    copiesmade = []
    try:
        for ext in [ '.image.fits', '.weight.fits', '.flags.fits', '.sources.fits', '.psf', '.psf.xml' ]:
            actual = datadir / f'{filepathbase}{ext}'
            if actual.exists():
                raise FileExistsError( f"{actual} exists, but at this point in the tests it's not supposed to" )
            shutil.copy2( datadir / f"{filepathbase}{ext}_cached", actual )
            copiesmade.append( actual )

        with SmartSession() as session:
            # The filenames will not match the provenance, because the filenames
            # are what they are, but as the code evoles the provenance tag is
            # going to change.  What's more, the provenances are different for
            # the images and for sources/psf.
            imgprov = Provenance( process="preprocessing", code_version=code_version, upstreams=[], is_testing=True )
            srcprov = Provenance( process="extraction", code_version=code_version,
                                  upstreams=[imgprov], is_testing=True )
            session.add( imgprov )
            session.add( srcprov )
            session.commit()
            session.refresh( imgprov )
            session.refresh( srcprov )

            with open( datadir / f'{filepathbase}.image.yaml' ) as ifp:
                imageyaml = yaml.safe_load( ifp )
            with open( datadir / f'{filepathbase}.sources.yaml' ) as ifp:
                sourcesyaml = yaml.safe_load( ifp )
            with open( datadir / f'{filepathbase}.psf.yaml' ) as ifp:
                psfyaml = yaml.safe_load( ifp )
            ds = DataStore( session=session )
            ds.image = Image( **imageyaml )
            ds.image.provenance = imgprov
            ds.image.filepath = filepathbase
            ds.sources = SourceList( **sourcesyaml )
            ds.sources.image = ds.image
            ds.sources.provenance = srcprov
            ds.sources.filepath = f'{filepathbase}.sources.fits'
            ds.psf = PSF( **psfyaml )
            ds.psf.image = ds.image
            ds.psf.provenance = srcprov
            ds.psf.filepath = filepathbase

            yield ds

            ds.delete_everything()
            session.delete( imgprov )
            session.delete( srcprov )
            session.commit()
            session.close()

    finally:
        for f in copiesmade:
            f.unlink( missing_ok=True )


# TODO : cache the results of this just like in
# decam_example_reduced_image_ds so they don't have to be regenerated
# every time this fixture is used.
@pytest.fixture
def decam_example_reduced_image_ds_with_wcs( decam_example_reduced_image_ds ):
    ds = decam_example_reduced_image_ds
    with open( ds.image.get_fullpath()[0], "rb" ) as ifp:
        md5 = hashlib.md5()
        md5.update( ifp.read() )
        origmd5 = uuid.UUID( md5.hexdigest() )

    xvals = [ 0, 0, 2047, 2047 ]
    yvals = [ 0, 4095, 0, 4095 ]
    origwcs = WCS( ds.image.raw_header )

    astrometor = AstroCalibrator( catalog='GaiaDR3', method='scamp', max_mag=[22.], mag_range=4.,
                                  min_stars=50, max_resid=0.15, crossid_radius=[2.0],
                                  min_frac_matched=0.1, min_matched_stars=10 )
    ds = astrometor.run( ds )

    return ds, origwcs, xvals, yvals, origmd5

    # Don't need to do any cleaning up, because no files were written
    # doing the WCS (it's all database), and the
    # decam_example_reduced_image_ds is going to do a
    # ds.delete_everything()


@pytest.fixture
def decam_example_reduced_image_ds_with_zp( decam_example_reduced_image_ds_with_wcs ):
    ds = decam_example_reduced_image_ds_with_wcs[0]
    ds.save_and_commit()
    photomotor = PhotCalibrator( cross_match_catalog='GaiaDR3' )
    ds = photomotor.run( ds )

    return ds, photomotor


@pytest.fixture
def ref_for_decam_example_image( provenance_base ):
    datadir = pathlib.Path( FileOnDiskMixin.local_path ) / 'test_data/DECam_examples'
    filebase = 'DECaPS-West_20220112.g.32'

    urlmap = { '.image.fits': '.fits.fz',
               '.weight.fits': '.weight.fits.fz',
               '.flags.fits': '.bpm.fits.fz' }
    for ext in [ '.image.fits', '.weight.fits', '.flags.fits' ]:
        path = datadir / f'{filebase}{ext}'
        cachedpath = datadir / f'{filebase}{ext}_cached'
        fzpath = datadir / f'{filebase}{ext}_cached.fz'
        if cachedpath.is_file():
            _logger.info( f"{path} exists, not redownloading." )
        else:
            url = ( f'https://portal.nersc.gov/cfs/m2218/decat/decat/templatecache/DECaPS-West_20220112.g/'
                    f'{filebase}{urlmap[ext]}' )
            retry_download( url, fzpath )
            res = subprocess.run( [ 'funpack', '-D', fzpath ] )
            if res.returncode != 0:
                raise SubprocessFailure( res )
        shutil.copy2( cachedpath, path )

    prov = provenance_base

    with open( datadir / f'{filebase}.image.yaml' ) as ifp:
        refyaml = yaml.safe_load( ifp )
    image = Image( **refyaml )
    image.provenance = prov
    image.filepath = f'test_data/DECam_examples/{filebase}'

    yield image

    # Just in case the image got added to the database:
    image.delete_from_disk_and_database()

    # And just in case the image was added to the database with a different name:
    for ext in [ '.image.fits', '.weight.fits', '.flags.fits' ]:
        ( datadir / f'{filebase}{ext}' ).unlink( missing_ok=True )


@pytest.fixture
def decam_small_image(decam_example_raw_image):
    image = decam_example_raw_image
    image.data = image.data[256:256+512, 256:256+512].copy()  # make it C-contiguous
    return image

