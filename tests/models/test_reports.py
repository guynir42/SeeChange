
from models.report import Report


def test_report_bitflags(decam_exposure, decam_reference):
    report = Report(exposure=decam_exposure, section_id='N1')

    # test that the progress steps flag is working
    assert report.progress_steps_bitflag == 0
    assert report.progress_steps == ''

    report.progress_steps = 'preprocessing'
    assert report.progress_steps_bitflag == 2 ** 1
    assert report.progress_steps == 'preprocessing'

    report.progress_steps = 'preprocessing, Extraction'
    assert report.progress_steps_bitflag == 2 ** 1 + 2 ** 2
    assert report.progress_steps == 'preprocessing, extraction'

    report.append_progress('photo_cal')
    assert report.progress_steps_bitflag == 2 ** 1 + 2 ** 2 + 2 ** 4
    assert report.progress_steps == 'preprocessing, extraction, photo_cal'

    report.append_progress('preprocessing')  # appending it again makes no difference
    assert report.progress_steps_bitflag == 2 ** 1 + 2 ** 2 + 2 ** 4
    assert report.progress_steps == 'preprocessing, extraction, photo_cal'

    report.append_progress('subtraction, cutting')  # append two at a time
    assert report.progress_steps_bitflag == 2 ** 1 + 2 ** 2 + 2 ** 4 + 2 ** 5 + 2 ** 7
    assert report.progress_steps == 'preprocessing, extraction, photo_cal, subtraction, cutting'

    # test that the products exist flag is working
    assert report.products_exist_bitflag == 0
    assert report.products_exist == ''

    report.products_exist = 'image'
    assert report.products_exist_bitflag == 2 ** 1
    assert report.products_exist == 'image'

    report.products_exist = 'image, sources'
    assert report.products_exist_bitflag == 2 ** 1 + 2 ** 2
    assert report.products_exist == 'image, sources'

    report.append_products_exist('psf')
    assert report.products_exist_bitflag == 2 ** 1 + 2 ** 2 + 2 ** 3
    assert report.products_exist == 'image, sources, psf'

    report.append_products_exist('image')  # appending it again makes no difference
    assert report.products_exist_bitflag == 2 ** 1 + 2 ** 2 + 2 ** 3
    assert report.products_exist == 'image, sources, psf'

    report.append_products_exist('sub_image, detections')  # append two at a time
    assert report.products_exist_bitflag == 2 ** 1 + 2 ** 2 + 2 ** 3 + 2 ** 7 + 2 ** 8
    assert report.products_exist == 'image, sources, psf, sub_image, detections'

    # test that the products committed flag is working
    assert report.products_committed_bitflag == 0
    assert report.products_committed == ''

    report.products_committed = 'sources'
    assert report.products_committed_bitflag == 2 ** 2
    assert report.products_committed == 'sources'

    report.products_committed = 'sources, zp'
    assert report.products_committed_bitflag == 2 ** 2 + 2 ** 6
    assert report.products_committed == 'sources, zp'

    report.append_products_committed('sub_image')
    assert report.products_committed_bitflag == 2 ** 2 + 2 ** 6 + 2 ** 7
    assert report.products_committed == 'sources, zp, sub_image'

    report.append_products_committed('sub_image, detections')  # append two at a time
    assert report.products_committed_bitflag == 2 ** 2 + 2 ** 6 + 2 ** 7 + 2 ** 8
    assert report.products_committed == 'sources, zp, sub_image, detections'

    report.append_products_committed('sub_image')  # appending it again makes no difference
    assert report.products_committed_bitflag == 2 ** 2 + 2 ** 6 + 2 ** 7 + 2 ** 8
    assert report.products_committed == 'sources, zp, sub_image, detections'


def test_inject_warnings(decam_datastore, decam_reference, pipeline_for_tests):
    pass


def test_inject_exceptions(decam_datastore, decam_reference, pipeline_for_tests):
    pass


