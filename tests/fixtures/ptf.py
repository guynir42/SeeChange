import pytest
import os
import shutil
import requests

import sqlalchemy as sa
from bs4 import BeautifulSoup
from datetime import datetime

from models.base import SmartSession, _logger
from models.ptf import PTF  # need this import to make sure PTF is added to the Instrument list
from models.exposure import Exposure
from util.retrydownload import retry_download


@pytest.fixture(scope='session')
def ptf_downloader(provenance_preprocessing, data_dir, cache_dir):
    cache_dir = os.path.join(cache_dir, 'PTF')

    def download_ptf_function(filename='PTF201104291667_2_o_45737_11.w.fits'):

        os.makedirs(cache_dir, exist_ok=True)
        cachedpath = os.path.join(cache_dir, filename)

        # first make sure file exists in the cache
        if os.path.isfile(cachedpath):
            _logger.info(f"{cachedpath} exists, not redownloading.")
        else:
            url = f'https://portal.nersc.gov/project/m2218/pipeline/test_images/{filename}'
            retry_download(url, cachedpath)  # make the cached copy

        if not os.path.isfile(cachedpath):
            raise FileNotFoundError(f"Can't read {cachedpath}. It should have been downloaded!")

        # copy the PTF exposure from cache to local storage:
        destination = os.path.join(data_dir, filename)

        if not os.path.isfile(destination):
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy(cachedpath, destination)

        exposure = Exposure(filepath=filename)

        return exposure

    return download_ptf_function


@pytest.fixture
def ptf_exposure(ptf_downloader):

    exposure = ptf_downloader()
    # check if this Exposure is already on the database
    with SmartSession() as session:
        existing = session.scalars(sa.select(Exposure).where(Exposure.filepath == exposure.filepath)).first()
        if existing is not None:
            _logger.info(f"Found existing Image on database: {existing}")
            # overwrite the existing row data using the JSON cache file
            for key in sa.inspect(exposure).mapper.columns.keys():
                value = getattr(exposure, key)
                if (
                        key not in ['id', 'image_id', 'created_at', 'modified'] and
                        value is not None
                ):
                    setattr(existing, key, value)
            exposure = existing  # replace with the existing row
        else:
            exposure.save()  # make sure it is up on the archive as well
            session.add(exposure)
            session.commit()

    yield exposure

    exposure.delete_from_disk_and_database()


@pytest.fixture
def ptf_datastore(datastore_factory, ptf_exposure, cache_dir):
    cache_dir = os.path.join(cache_dir, 'PTF')
    ptf_exposure.instrument_object.fetch_sections()
    ds = datastore_factory(
        ptf_exposure,
        11,
        cache_dir=cache_dir,
        cache_base_name='187/PTF_20110429_040004_11_R_Sci_5F5TAU',
        overrides={'extraction': {'threshold': 5}},
    )
    yield ds
    ds.delete_everything()


@pytest.fixture
def ptf_urls():
    base_url = f'https://portal.nersc.gov/project/m2218/pipeline/test_images/'
    r = requests.get(base_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    links = soup.find_all('a')
    filenames = [link.get('href') for link in links if link.get('href').endswith('.fits')]

    yield filenames


@pytest.fixture
def ptf_images_factory(ptf_urls, ptf_downloader, datastore_factory, cache_dir):
    cache_dir = os.path.join(cache_dir, 'PTF')

    cache_names = {}
    if os.path.isfile(os.path.join(cache_dir, 'manifest.txt')):
        with open(os.path.join(cache_dir, 'manifest.txt')) as f:
            text = f.read().splitlines()
        for line in text:
            filename, cache_name = line.split()
            cache_names[filename] = cache_name

    def factory(start_date='2009-04-04', end_date='2013-03-03'):
        start_time = datetime.strptime(start_date, '%Y-%m-%d') if start_date is not None else datetime(1, 1, 1)
        end_time = datetime.strptime(end_date, '%Y-%m-%d') if end_date is not None else datetime(3000, 1, 1)

        urls = []
        for url in ptf_urls:
            obstime = datetime.strptime(url[3:11], '%Y%m%d')
            if start_time <= obstime <= end_time:
                urls.append(url)

        images = []
        for url in urls:
            exp = ptf_downloader(url)
            exp.instrument_object.fetch_sections()
            try:
                ds = datastore_factory(
                    exp,
                    11,
                    cache_dir=cache_dir,
                    cache_base_name=cache_names.get(url, None),
                    overrides={'extraction': {'threshold': 5}},
                )
                if hasattr(ds, 'cache_base_name') and ds.cache_base_name is not None:
                    cache_names[url] = ds.cache_base_name
            except Exception as e:
                # print(e)  # TODO: should we be worried that some of these images can't complete their processing?
                continue  # I think we should fix this along with issue #150
            images.append(ds.image)

        # save the manifest file
        with open(os.path.join(cache_dir, 'manifest.txt'), 'w') as f:
            for key, value in cache_names.items():
                f.write(f'{key} {value}\n')

        return images

    return factory


@pytest.fixture(scope='session')
def ptf_reference_images(ptf_images_factory):
    images = ptf_images_factory('2009-04-04', '2009-04-05')

    yield images

    with SmartSession() as session:
        for image in images:
            image = image.recursive_merge(session)
            image.exposure.delete_from_disk_and_database(session=session, commit=False)
            image.sources.delete_from_disk_and_database(session=session, commit=False)
            image.delete_from_disk_and_database()
        session.commit()



