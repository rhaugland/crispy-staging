import pytest
import boto3
from moto import mock_aws
from staging.storage import StorageClient


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="virtual-staging")
        yield StorageClient(bucket="virtual-staging", region="us-east-1")


def test_upload_and_download(s3):
    s3.upload(key="jobs/123/input/photo1.jpg", data=b"fake-image-data")
    result = s3.download(key="jobs/123/input/photo1.jpg")
    assert result == b"fake-image-data"


def test_save_checkpoint(s3):
    s3.save_checkpoint(job_id="123", stage="reconstruction", data=b"mesh-data")
    result = s3.download(key="jobs/123/checkpoints/reconstruction")
    assert result == b"mesh-data"


def test_list_results(s3):
    s3.upload(key="jobs/123/results/staged_0.jpg", data=b"img0")
    s3.upload(key="jobs/123/results/staged_1.jpg", data=b"img1")
    keys = s3.list_results(job_id="123")
    assert len(keys) == 2


def test_download_missing_key(s3):
    with pytest.raises(FileNotFoundError):
        s3.download(key="nonexistent")
