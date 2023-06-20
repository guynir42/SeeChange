import os

from models.base import SmartSession
from pipeline.top_level import Pipeline


def test_data_flow(exposure, reference_image):
    """Test that the pipeline runs end-to-end."""
    filename = None
    try:  # cleanup the file at the end
        with SmartSession() as session:
            reference_image = session.merge(reference_image)

            # make sure exposure matches the reference image
            exposure.target = reference_image.target
            exposure.filter = reference_image.filter
            section_id = int(reference_image.section_id)

            session.add(exposure)
            session.commit()
            exp_id = exposure.id

            filename = exposure.get_fullpath()
            open(filename, 'a').close()

        p = Pipeline()

        p.run(exp_id, section_id)

    finally:
        if filename is not None and os.path.isfile(filename):
            os.remove(filename)

