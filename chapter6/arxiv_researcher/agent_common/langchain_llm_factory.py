from enum import Enum

from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_ollama import OllamaLLM
from langchain_openai import AzureChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from loguru import logger
from pydantic import SecretStr

from arxiv_researcher.agent_common.settings import Settings


class LLMType(Enum):
    """LLMの種類を定義するEnumクラス"""

    AZURE_OPENAI = "AZURE_OPENAI"
    OLLAMA = "OLLAMA"


class LangchainLLMFactory:
    # 環境変数を.envから読み込む
    __settings: Settings = Settings()

    @classmethod
    def create_llm(
        cls,
        llm_type: LLMType,
        temperature: float = 0.0,
        enable_default_tracing: bool = True,
    ) -> tuple[BaseLanguageModel, CallbackHandler | None]:
        """LLMを生成するシンプルファクトリ

        Args:
            llm_type (LLMType): LLMの種類を指定するEnumクラス
            temperature (float, optional): temperature. Defaults to 0.0.
            enable_default_tracing (bool, optional): Langfuseのトレーシングをllmインスタンスに適用するするかどうか. Defaults to True.

        Raises:
            ValueError: _description_

        Returns:
            BaseLanguageModel: 生成されたLLM
            CallbackHandler | None: Langfuseのハンドラー
        """
        logger.info(
            {
                "action": "create_llm",
                "llm_type": llm_type,
                "temperature": temperature,
            }
        )

        # Langfuse
        langfuse_handler: CallbackHandler | None = cls.create_langfuse_handler()
        match llm_type:
            case LLMType.AZURE_OPENAI:
                # AzureOpenAIの場合
                llm = AzureChatOpenAI(
                    api_key=SecretStr(cls.__settings.AZURE_OPENAI_KEY),
                    azure_endpoint=cls.__settings.AZURE_OPENAI_ENDPOINT,
                    api_version=cls.__settings.AZURE_OPENAI_VERSION,
                    azure_deployment=cls.__settings.AZURE_OPENAI_DEPLOYMENT_NAME_RAG,
                    temperature=temperature,
                    callbacks=[langfuse_handler]
                    if langfuse_handler and enable_default_tracing
                    else None,
                )
                return llm, langfuse_handler
            case LLMType.OLLAMA:
                # Ollamaの場合
                llm = OllamaLLM(
                    model=cls.__settings.OLLAMA_DEPLOYMENT_NAME,
                    base_url=cls.__settings.OLLAMA_BASE_URL,
                    temperature=temperature,
                    callbacks=[langfuse_handler]
                    if langfuse_handler and enable_default_tracing
                    else None,
                )
                return llm, langfuse_handler
            case _:
                raise ValueError(f"Unsupported LLM type: {llm_type}")

    @classmethod
    def create_langfuse_handler(cls) -> CallbackHandler | None:
        """Langfuseのハンドラーを生成する
        Returns:
            CallbackHandler: Langfuseのハンドラー
        """
        if cls.__settings.LANGFUSE_TRACING_ENABLED:
            try:
                load_dotenv(override=True)
                langfuse = get_client()
                # Verify connection
                if langfuse.auth_check():
                    logger.info(
                        {
                            "action": "create_langfuse_handler",
                            "enable": cls.__settings.LANGFUSE_TRACING_ENABLED,
                        }
                    )
                    return CallbackHandler()
                else:
                    logger.error(
                        {
                            "action": "create_langfuse_handler",
                            "enable": cls.__settings.LANGFUSE_TRACING_ENABLED,
                            "msg": "Authentication failed. Please check your credentials and host.",
                        }
                    )
            except Exception as e:
                logger.exception(
                    {
                        "action": "create_langfuse_handler",
                        "msg": f"Langfuse clientの初期化に失敗しました: {e}",
                    }
                )
        return None


def test():
    """テスト関数"""
    print("LLMのテストを開始します。")
    from langchain_core.prompts import ChatPromptTemplate

    llm, _ = LangchainLLMFactory.create_llm(LLMType.OLLAMA, temperature=0.5)

    prompt = ChatPromptTemplate.from_template("""\
        Question: {question}
        Answer: ステップバイステップで考えてみましょう。""")
    chain = prompt | llm

    print("リクエストを投げます。")
    result = chain.invoke({"question": "美味しいパスタの作り方は?"})
    print(result)


if __name__ == "__main__":
    test()
