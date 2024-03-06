

def test_measuring(measurer, decam_cutouts):
    ds = measurer.run(decam_cutouts)
    print(ds)