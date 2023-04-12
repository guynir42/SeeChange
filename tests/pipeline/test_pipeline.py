from pipeline.pipeline import Pipeline

def test_data_flow():
    """Test that the pipeline runs end-to-end."""
    p = Pipeline()

    p.run()