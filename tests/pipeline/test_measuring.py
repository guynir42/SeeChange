import uuid

import numpy as np


def test_measuring(measurer, decam_cutouts):
    measurer.pars.test_parameter = uuid.uuid4().hex
    sz = decam_cutouts[0].sub_data.shape
    decam_cutouts[0].sub_data = np.zeros_like(decam_cutouts[0].sub_data)
    decam_cutouts[0].sub_data[sz[0]//2, sz[1]//2] = 1
    ds = measurer.run(decam_cutouts)

    assert len(ds.measurements) == len(ds.cutouts)

    print(ds)