"""Application settings loaded from the local .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_db: str = "aic_ttvn"
    postgres_user: str = "aic_app"
    postgres_password: str
    postgres_host_port: int = 55432
    minio_root_user: str
    minio_root_password: str
    minio_api_port: int = 9000
    milvus_port: int = 19530
    aic_kaggle_artifact_root: Path = Path(".runtime/artifacts/kaggle")
    aic_kaggle_batch: str = "l21"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@127.0.0.1:{self.postgres_host_port}/{self.postgres_db}"
        )

    @property
    def minio_endpoint(self) -> str:
        return f"127.0.0.1:{self.minio_api_port}"

    @property
    def milvus_uri(self) -> str:
        return f"http://127.0.0.1:{self.milvus_port}"


settings = Settings()
