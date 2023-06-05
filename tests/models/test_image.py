import pytest
import re

import numpy as np
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from models.base import SmartSession
from models.image import Image


def rnd_str(n):
    return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n))


def test_image_no_null_values(provenance_base):

    required = {
        'mjd': 58392.1,
        'end_mjd': 58392.1 + 30 / 86400,
        'exp_time': 30,
        'filter': 'r',
        'ra': np.random.uniform(0, 360),
        'dec': np.random.uniform(-90, 90),
        'instrument': 'DemoInstrument',
        'telescope': 'DemoTelescope',
        'project': 'foo',
        'target': 'bar',
        'provenance_id': provenance_base.id,
        'section_id': 1,
    }

    added = {}

    # use non-capturing groups to extract the column name from the error message
    expr = r'(?:null value in column )(".*")(?: of relation "images" violates not-null constraint)'

    try:
        im_id = None  # make sure to delete the image if it is added to DB
        image = Image(f"Demo_test_{rnd_str(5)}.fits", nofile=True)
        with SmartSession() as session:
            for i in range(len(required)):
                print(set(required.keys()) - set(added.keys()))
                # set the exposure to the values in "added" or None if not in "added"
                for k in required.keys():
                    setattr(image, k, added.get(k, None))

                # without all the required columns on image, it cannot be added to DB
                with pytest.raises(IntegrityError) as exc:
                    session.add(image)
                    session.commit()
                    im_id = image.id
                session.rollback()

                # a constraint on a column being not-null was violated
                match_obj = re.search(expr, str(exc.value))
                assert match_obj is not None

                # find which column raised the error
                colname = match_obj.group(1).replace('"', '')

                # add missing column name:
                added.update({colname: required[colname]})

        for k in required.keys():
            setattr(image, k, added.get(k, None))
        session.add(image)
        session.commit()
        im_id = image.id
        assert im_id is not None

    finally:
        # cleanup
        with SmartSession() as session:
            exposure = None
            if im_id is not None:
                exposure = session.scalars(sa.select(Image).where(Image.id == im_id)).first()
            if exposure is not None:
                session.delete(exposure)
                session.commit()


def test_image_coordinates():
    image = Image('foo.fits', ra=None, dec=None, nofile=True)
    assert image.ecllat is None
    assert image.ecllon is None
    assert image.gallat is None
    assert image.gallon is None

    with pytest.raises(ValueError, match='Object must have RA and Dec set'):
        image.calculate_coordinates()

    image = Image('foo.fits', ra=123.4, dec=None, nofile=True)
    assert image.ecllat is None
    assert image.ecllon is None
    assert image.gallat is None
    assert image.gallon is None

    image = Image('foo.fits', ra=123.4, dec=56.78, nofile=True)
    assert abs(image.ecllat - 35.846) < 0.01
    assert abs(image.ecllon - 111.838) < 0.01
    assert abs(image.gallat - 33.542) < 0.01
    assert abs(image.gallon - 160.922) < 0.01


def test_image_from_exposure(exposure, provenance_base):
    exposure.update_instrument()
    exposure.type = 'reference'

    # demo instrument only has one section
    with pytest.raises(ValueError, match='section_id must be 0 for this instrument.'):
        _ = Image.from_exposure(exposure, section_id=1)

    im = Image.from_exposure(exposure, section_id=0)
    assert im.section_id == 0
    assert im.mjd == exposure.mjd
    assert im.end_mjd == exposure.end_mjd
    assert im.exp_time == exposure.exp_time
    assert im.end_mjd == im.mjd + im.exp_time / 86400
    assert im.filter == exposure.filter
    assert im.instrument == exposure.instrument
    assert im.telescope == exposure.telescope
    assert im.project == exposure.project
    assert im.target == exposure.target
    assert im.combine_method is None
    assert not im.is_multi_image
    assert im.id is None  # need to commit to get IDs
    assert im.exposure_id is None  # need to commit to get IDs
    assert im.source_images == []
    assert im.filepath is None  # need to save file to generate a filename
    assert np.array_equal(im.raw_data, exposure.data[0])
    assert im.data is None
    assert im.flags is None
    assert im.weight is None
    assert im.nofile  # images are made without a file by default

    # TODO: add check for loading the header after we make a demo header maker
    # TODO: what should the RA/Dec be for an image that cuts out from an exposure?

    im_id = None
    try:
        with SmartSession() as session:
            with pytest.raises(IntegrityError, match='null value in column "provenance_id" of relation "images"'):
                session.add(im)
                session.commit()
            session.rollback()

            # must add the provenance!
            im.provenance = provenance_base
            with pytest.raises(IntegrityError, match='null value in column "filepath" of relation "images"'):
                session.add(im)
                session.commit()
            session.rollback()

            # must add the filepath!
            im.filepath = 'foo.fits'

            session.add(im)
            session.commit()

            assert im.id is not None
            assert im.provenance_id is not None
            assert im.provenance_id == provenance_base.id
            assert im.exposure_id is not None
            assert im.exposure_id == exposure.id

    finally:
        if im_id is not None:
            with SmartSession() as session:
                im = session.scalars(sa.select(Image).where(Image.id == im_id)).first()
                session.delete(im)
                session.commit()


def test_image_from_exposure_filter_array(exposure_filter_array):
    exposure_filter_array.update_instrument()
    im = Image.from_exposure(exposure_filter_array, section_id=0)
    filt = exposure_filter_array.filter_array[0]
    assert im.filter == filt


def test_image_with_multiple_source_images(exposure, exposure2, provenance_base):
    exposure.update_instrument()
    exposure2.update_instrument()

    # get a couple of images from exposure objects
    im1 = Image.from_exposure(exposure, section_id=0)
    im2 = Image.from_exposure(exposure2, section_id=0)
    if im1.mjd > im2.mjd:
        im1, im2 = im2, im1

    im1.provenance = provenance_base
    im1.filepath = 'foo1.fits'
    im2.provenance = provenance_base
    im2.filepath = 'foo2.fits'

    # make a new image from the two (we still don't have a coadd method for this)
    im = Image(
        exp_time=im1.exp_time + im2.exp_time,
        mjd=im1.mjd,
        end_mjd=im2.end_mjd,
        filter=im1.filter,
        instrument=im1.instrument,
        telescope=im1.telescope,
        project=im1.project,
        target=im1.target,
        combine_method='coadd',
        section_id=im1.section_id,
        ra=im1.ra,
        dec=im1.dec,
        filepath='foo.fits'
    )
    im.source_images = [im1, im2]
    im.provenance = provenance_base

    try:
        im_id = None
        im1_id = None
        im2_id = None
        with SmartSession() as session:
            session.add(im)
            session.commit()

            im_id = im.id
            assert im_id is not None
            assert im.exposure_id is None
            assert im.is_multi_image
            assert im.source_images == [im1, im2]
            assert np.isclose(im.mid_mjd, (im1.mjd + im2.mjd) / 2)

            # make sure source images are pulled into the database too
            im1_id = im1.id
            assert im1_id is not None
            assert im1.exposure_id is not None
            assert im1.exposure_id == exposure.id
            assert not im1.is_multi_image
            assert im1.source_images == []

            im2_id = im2.id
            assert im2_id is not None
            assert im2.exposure_id is not None
            assert im2.exposure_id == exposure2.id
            assert not im2.is_multi_image
            assert im2.source_images == []

    finally:  # make sure to clean up all images
        for id_ in [im_id, im1_id, im2_id]:
            if id_ is not None:
                with SmartSession() as session:
                    im = session.scalars(sa.select(Image).where(Image.id == id_)).first()
                    session.delete(im)
                    session.commit()


