import pytest
import os
import shutil
import requests

from bs4 import BeautifulSoup


from models.base import SmartSession, FileOnDiskMixin, _logger
from models.image import Image
from util.retrydownload import retry_download


@pytest.fixture
def ptf_image_new(provenance_base):
    datadir = os.path.join(FileOnDiskMixin.local_path, 'test_data/PTF_examples')
    if not os.path.isdir(datadir):
        os.makedirs(datadir)

    filename = 'PTF201104234316_2_o_44887_11.w.fits'

    path = os.path.join(datadir, filename)
    cachedpath = filename + '_cached'

    if os.path.isfile(cachedpath):
        _logger.info(f"{path} exists, not redownloading.")
    else:
        url = f'https://portal.nersc.gov/project/m2218/pipeline/dec+04/ccd_11/{filename}'
        retry_download(url, cachedpath)  # make the cached copy

    shutil.copy2(cachedpath, path)  # make a temporary copy to be deleted at end of test
    prov = provenance_base

    # with open(datadir / f'{filebase}.image.yaml') as ifp:
    #     refyaml = yaml.safe_load(ifp)
    # image = Image(**refyaml)
    image = Image()
    image.provenance = prov
    image.filepath = filename

    yield image

    # Just in case the image got added to the database:
    image.delete_from_disk_and_database()


@pytest.fixture(scope='session')
def all_ptf_example_images(provenance_base):
    datadir = os.path.join(FileOnDiskMixin.local_path, 'test_data/PTF_examples')
    if not os.path.isdir(datadir):
        os.makedirs(datadir)

    url = f'https://portal.nersc.gov/project/m2218/pipeline/dec+04/ccd_11/'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    links = soup.find_all('a')
    filenames = [link.get('href') for link in links if link.get('href').endswith('.fits')]
    images = []
    for filename in filenames:
        path = os.path.join(datadir, filename)
        cachedpath = filename + '_cached'

        if os.path.isfile(cachedpath):
            _logger.info(f"{path} exists, not redownloading.")
        else:
            url = f'https://portal.nersc.gov/project/m2218/pipeline/dec+04/ccd_11/{filename}'
            retry_download(url, cachedpath)

        shutil.copy2(cachedpath, path)
        new_image = Image(filepath=path)
        new_image.provenance = provenance_base
        images.append(new_image)

    yield images

    with SmartSession() as session:
        for image in images:
            image.delete_from_disk_and_database(session=session, commit=False)
            session.delete(image)
        session.commit()


def test_get_ptf_image(ptf_image_new):
    print(ptf_image_new)
    print(ptf_image_new)
