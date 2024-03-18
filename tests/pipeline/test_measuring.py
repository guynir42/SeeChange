import uuid

import numpy as np


def test_measuring(measurer, decam_cutouts):
    measurer.pars.test_parameter = uuid.uuid4().hex
    decam_cutouts[0] =
    ds = measurer.run(decam_cutouts)


    assert len(ds.measurements) == len(ds.cutouts)

    print(ds)