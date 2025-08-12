from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = "<API_Key>"
    openai_api_base: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-2024-08-06"

    # Azure OpenAI用の設定
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_deployment_name: Optional[str] = None
    azure_openai_embedding_deployment_name: Optional[str] = None

    # APIプロバイダーの選択 ("openai" or "azure")
    api_provider: str = "openai"

    model_config = SettingsConfigDict(env_file="./chapter4/.env", extra="ignore")
