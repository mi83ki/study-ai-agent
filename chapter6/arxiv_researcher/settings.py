import os

import cohere
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from arxiv_researcher.agent_common.langchain_llm_factory import (
    LangchainLLMFactory,
    LLMType,
)


class ArxivResearcherSettings(BaseModel):
    """ArxivResearcherエージェントの設定"""

    # 追加情報取得のための最大リトライ回数
    max_evaluation_retry_count: int = 3


class QueryDecomposerSettings(BaseModel):
    """QueryDecomposerエージェントの設定"""

    # タスク分解時の最小タスク数
    min_decomposed_tasks: int = 3
    # タスク分解時の最大タスク数
    max_decomposed_tasks: int = 5


class ArxivSearchAgentSettings(BaseModel):
    """ArxivSearchAgentエージェントの設定"""

    # 検索失敗時の最大リトライ回数
    max_retries: int = 3
    # 1回の検索で取得する最大論文数
    max_search_results: int = 10
    # 詳細分析する最大論文数
    max_papers: int = 3
    # PDFからmarkdown変換時の並列処理数
    max_workers: int = 3


class LangGraphSettings(BaseModel):
    """LangGraph関連の設定"""

    # ノードの最大実行回数制限
    max_recursion_limit: int = 1000


class ModelSettings(BaseModel):
    """モデル関連の設定"""

    # 高速・軽量なタスク用のOpenAIモデル
    openai_fast_model: str = "gpt-4o-mini"
    # 複雑なタスク用のOpenAIモデル
    openai_smart_model: str = "gpt-4o"
    # レポート生成用のAnthropicモデル
    anthropic_model: str = "claude-3-7-sonnet-20250219"
    # 埋め込み用のOpenAIモデル
    openai_embedding_model: str = "text-embedding-3-small"
    # リランキング用のCohereモデル
    cohere_rerank_model: str = "rerank-multilingual-v3.0"
    # 生成時の温度パラメータ
    temperature: float = 0.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # 必須のAPIキー
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    COHERE_API_KEY: str
    JINA_API_KEY: str

    # LangSmith関連の設定
    LANGSMITH_TRACING_V2: str = "false"  # トレーシング機能の有効/無効
    LANGSMITH_ENDPOINT: str = (
        "https://api.smith.langchain.com"  # LangSmithのエンドポイント
    )
    LANGSMITH_API_KEY: str = ""  # LangSmithのAPIキー
    LANGSMITH_PROJECT: str = "arxiv-researcher"  # プロジェクト名

    # 使用するLLMの種類(AZURE_OPENAI or OLLAMA)
    LLM_TYPE: str = "AZURE_OPENAI"
    LLM_MODEL: str = "gpt-4.1-quality"
    FAST_LLM_TYPE: str = "AZURE_OPENAI"
    FAST_LLM_MODEL: str = "gpt-4.1-mini-quality"
    REPORTER_LLM_TYPE: str = "AZURE_OPENAI"
    REPORTER_LLM_MODEL: str = "gpt-4.1-quality"

    # If using Azure OpenAI
    AZURE_OPENAI_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_VERSION: str = ""
    AZURE_OPENAI_DEPLOYMENT_NAME_RAG: str = ""

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

    # デバッグモード設定
    debug: bool = True

    # エージェント毎の設定インスタンス
    arxiv_researcher: ArxivResearcherSettings = ArxivResearcherSettings()
    query_decomposer: QueryDecomposerSettings = QueryDecomposerSettings()
    arxiv_search_agent: ArxivSearchAgentSettings = ArxivSearchAgentSettings()
    langgraph: LangGraphSettings = LangGraphSettings()
    model: ModelSettings = ModelSettings()

    def __init__(self, **values):
        # .envファイルが存在する場合のみ読み込み（主に開発環境用）
        env_path = ".env"
        if os.path.exists(env_path):
            import dotenv

            dotenv.load_dotenv(env_path, override=True)
        super().__init__(**values)

    @property
    def llm(self) -> BaseLanguageModel:
        """複雑なタスク用のLLMインスタンスを返す"""
        llm, _ =LangchainLLMFactory.create_llm(
            llm_type=LLMType[self.LLM_TYPE],
            model_name=self.LLM_MODEL,
            temperature=self.model.temperature,
            enable_default_tracing=False
        )
        return llm

    @property
    def fast_llm(self) -> BaseLanguageModel:
        """高速・軽量なタスク用のLLMインスタンスを返す"""
        llm, _ =LangchainLLMFactory.create_llm(
            llm_type=LLMType[self.FAST_LLM_TYPE],
            model_name=self.FAST_LLM_MODEL,
            temperature=self.model.temperature,
            enable_default_tracing=False
        )
        return llm

    @property
    def reporter_llm(self) -> BaseLanguageModel:
        """レポート生成用のLLMインスタンスを返す"""
        llm, _ =LangchainLLMFactory.create_llm(
            llm_type=LLMType[self.REPORTER_LLM_TYPE],
            model_name=self.REPORTER_LLM_MODEL,
            temperature=self.model.temperature,
            enable_default_tracing=False
        )
        return llm

    @property
    def cohere_client(self) -> cohere.Client:
        """Cohereクライアントインスタンスを返す"""
        return cohere.Client(api_key=self.COHERE_API_KEY)


# グローバルな設定インスタンス
settings = Settings()
