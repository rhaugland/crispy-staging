from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    s3_bucket: str = "virtual-staging"
    s3_region: str = "us-east-1"
    luma_api_key: str = ""
    anthropic_api_key: str = ""
    redis_url: str = "redis://localhost:6379"
    max_photos: int = 4
    min_photos: int = 3
    max_file_size_mb: int = 20
    min_resolution: tuple[int, int] = (1920, 1080)
    max_resolution: tuple[int, int] = (8000, 6000)
    quality_gate_max_retries: int = 2
    checkpoint_ttl_days: int = 30
    result_ttl_days: int = 90

    model_config = {"env_prefix": "STAGING_"}


settings = Settings()
