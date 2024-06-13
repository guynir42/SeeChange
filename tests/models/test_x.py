# I'm adding this test in this temporary file just to figure out this weird bug that happens on GA but not locally.

import numpy as np

import sqlalchemy as sa

from models.base import SmartSession
from models.image import Image
from models.source_list import SourceList
from models.cutouts import Cutouts
from models.measurements import Measurements

import pdb
import matplotlib.pyplot as plt


def test_filtering_measurements(ptf_datastore):
    # printout the list of relevant environmental variables:
    import os
    print("SeeChange environment variables:")
    for key in [
        'INTERACTIVE',
        'LIMIT_CACHE_USAGE',
        'SKIP_NOIRLAB_DOWNLOADS',
        'RUN_SLOW_TESTS',
        'SEECHANGE_TRACEMALLOC',
    ]:
        print(f'{key}: {os.getenv(key)}')

    measurements = ptf_datastore.measurements
    from pprint import pprint
    print('measurements: ')
    pprint(measurements)

    if hasattr(ptf_datastore, 'all_measurements'):
        idx = [m.cutouts.index_in_sources for m in measurements]
        chosen = np.array(ptf_datastore.all_measurements)[idx]
        pprint([(m, m.is_bad, m.cutouts.sub_nandata[12, 12]) for m in chosen])

    print(f'new image values: {ptf_datastore.image.data[250, 240:250]}')
    print(f'ref_image values: {ptf_datastore.ref_image.data[250, 240:250]}')
    print(f'sub_image values: {ptf_datastore.sub_image.data[250, 240:250]}')

    print(f'number of images in ref image: {len(ptf_datastore.ref_image.upstream_images)}')
    for i, im in enumerate(ptf_datastore.ref_image.upstream_images):
        print(f'upstream image {i}: {im.data[250, 240:250]}')

    m = measurements[0]  # grab the first one as an example

    # pdb.set_trace()
    # test that we can filter on some measurements properties
    with SmartSession() as session:
        ms = session.scalars(sa.select(Measurements).where(Measurements.flux_apertures[0] > 0)).all()
        assert len(ms) == len(measurements)  # saved measurements will probably have a positive flux

        ms = session.scalars(sa.select(Measurements).where(Measurements.flux_apertures[0] > 200)).all()
        assert len(ms) < len(measurements)  # only some measurements have a flux above 200

        ms = session.scalars(
            sa.select(Measurements).join(Cutouts).join(SourceList).join(Image).where(
                Image.mjd == m.mjd, Measurements.provenance_id == m.provenance.id
            )).all()
        assert len(ms) == len(measurements)  # all measurements have the same MJD

        ms = session.scalars(
            sa.select(Measurements).join(Cutouts).join(SourceList).join(Image).where(
                Image.exp_time == m.exp_time, Measurements.provenance_id == m.provenance.id
            )).all()
        assert len(ms) == len(measurements)  # all measurements have the same exposure time

        ms = session.scalars(
            sa.select(Measurements).join(Cutouts).join(SourceList).join(Image).where(
                Image.filter == m.filter, Measurements.provenance_id == m.provenance.id
            )).all()
        assert len(ms) == len(measurements)  # all measurements have the same filter

        ms = session.scalars(sa.select(Measurements).where(Measurements.background > 0)).all()
        assert len(ms) <= len(measurements)  # only some of the measurements have positive background

        ms = session.scalars(sa.select(Measurements).where(
            Measurements.offset_x > 0, Measurements.provenance_id == m.provenance.id
        )).all()
        assert len(ms) <= len(measurements)  # only some of the measurements have positive offsets

        ms = session.scalars(sa.select(Measurements).where(
            Measurements.area_psf >= 0, Measurements.provenance_id == m.provenance.id
        )).all()
        assert len(ms) == len(measurements)  # all measurements have positive psf area

        ms = session.scalars(sa.select(Measurements).where(
            Measurements.width >= 0, Measurements.provenance_id == m.provenance.id
        )).all()
        assert len(ms) == len(measurements)  # all measurements have positive width

        # filter on a specific disqualifier score
        ms = session.scalars(sa.select(Measurements).where(
            Measurements.disqualifier_scores['negatives'].astext.cast(sa.REAL) < 0.1,
            Measurements.provenance_id == m.provenance.id
        )).all()
        assert len(ms) <= len(measurements)
