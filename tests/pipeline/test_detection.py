import numpy as np


def test_detection_of_ptf_supernova(detector, ptf_subtraction1):
    print(ptf_subtraction1)

    ds = detector.run(ptf_subtraction1)

    assert ds.detections is not None
    assert ds.detections.num_sources > 0
