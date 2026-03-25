from __future__ import annotations

import boto3
from botocore.exceptions import ClientError

from staging.config import settings


class StorageClient:
    def __init__(
        self,
        bucket: str = settings.s3_bucket,
        region: str = settings.s3_region,
    ):
        self.bucket = bucket
        self.s3 = boto3.client("s3", region_name=region)

    def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return key

    def download(self, key: str) -> bytes:
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"Key not found: {key}")
            raise

    def save_checkpoint(self, job_id: str, stage: str, data: bytes) -> str:
        key = f"jobs/{job_id}/checkpoints/{stage}"
        return self.upload(key=key, data=data)

    def load_checkpoint(self, job_id: str, stage: str) -> bytes:
        return self.download(key=f"jobs/{job_id}/checkpoints/{stage}")

    def list_results(self, job_id: str) -> list[str]:
        prefix = f"jobs/{job_id}/results/"
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]
