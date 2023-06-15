from models.base import SmartSession
from pipeline.pipeline import Pipeline


def test_data_flow(exposure):
    """Test that the pipeline runs end-to-end."""
    with SmartSession() as session:
        session.add(exposure)
        session.commit()
        exp_id = exposure.id

    p = Pipeline()

    p.run(exp_id, 0)
