from staging.worker import StagingWorker


def test_worker_has_run_method():
    worker = StagingWorker.__new__(StagingWorker)
    assert hasattr(worker, "run_job")


def test_worker_stage_methods_exist():
    worker = StagingWorker.__new__(StagingWorker)
    for method in ["run_job"]:
        assert callable(getattr(worker, method))
