from models.base import SmartSession
from pipeline.top_level import Pipeline


def test_data_flow(exposure):
    """Test that the pipeline runs end-to-end."""
    with SmartSession() as session:
        session.add(exposure)
        session.commit()
        exp_id = exposure.id
        exposure.save()

    p = Pipeline()

    p.run(exp_id, 0)
