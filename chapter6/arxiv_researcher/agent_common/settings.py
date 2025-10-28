import os

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Proxy
    HTTP_PROXY: str = ""
    HTTPS_PROXY: str = ""
    NO_PROXY: str = ""

    # 使用するLLMの種類(AZURE_OPENAI or OLLAMA)
    LLM_TYPE: str = "AZURE_OPENAI"

    # If using Azure OpenAI
    AZURE_OPENAI_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_VERSION: str = ""
    AZURE_OPENAI_DEPLOYMENT_NAME_RAG: str = ""
    AZURE_OPENAI_DEPLOYMENT_NAME_FAST: str = ""

    # If using Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_DEPLOYMENT_NAME: str = "gemma3:1b"

    # If using Ollama Embedding
    OLLAMA_EMBEDDING_MODEL_NAME: str = "bge-m3"

    # Langfuse
    # Langfuseのトレースを無効にするときはFalseに変更する
    LANGFUSE_TRACING_ENABLED: bool = False
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3000"

    def __init__(self, **values):
        super().__init__(**values)
        if self.HTTP_PROXY:
            os.environ["HTTP_PROXY"] = self.HTTP_PROXY
            os.environ["http_proxy"] = self.HTTP_PROXY
        if self.HTTPS_PROXY:
            os.environ["HTTPS_PROXY"] = self.HTTPS_PROXY
            os.environ["https_proxy"] = self.HTTPS_PROXY
        if self.NO_PROXY:
            os.environ["NO_PROXY"] = self.NO_PROXY
            os.environ["no_proxy"] = self.NO_PROXY

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """OSの環境変数と.envに同名の環境変数が定義されている場合に、.envの値を優先するようにする"""
        return dotenv_settings, init_settings, env_settings, file_secret_settings
