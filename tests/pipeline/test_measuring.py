import numpy as np


def test_measuring(measurer, decam_cutouts):
    ds = measurer.run(decam_cutouts)

    assert len(ds.measurements) == len(ds.cutouts)

    print(ds)