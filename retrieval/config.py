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
    minio_endpoint_override: str = ""
    minio_public_endpoint: str = ""
    milvus_port: int = 19530
    milvus_uri_override: str = ""
    aic_kaggle_artifact_root: Path = Path(".runtime/artifacts/kaggle")
    aic_kaggle_batch: str = "l21"
    app_accounts: str
    app_admin_accounts: str = "TTVN_admin"
    app_secret_key: str

    @property
    def accounts(self) -> dict[str, str]:
        pairs = (pair.strip() for pair in self.app_accounts.split(","))
        return dict(pair.split(":", 1) for pair in pairs if ":" in pair)

    @property
    def admin_accounts(self) -> set[str]:
        return {name.strip() for name in self.app_admin_accounts.split(",") if name.strip()}

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@127.0.0.1:{self.postgres_host_port}/{self.postgres_db}"
        )

    @property
    def minio_endpoint(self) -> str:
        if self.minio_endpoint_override:
            return self.minio_endpoint_override
        return f"127.0.0.1:{self.minio_api_port}"

    @property
    def milvus_uri(self) -> str:
        if self.milvus_uri_override:
            return self.milvus_uri_override
        return f"http://127.0.0.1:{self.milvus_port}"


settings = Settings()
