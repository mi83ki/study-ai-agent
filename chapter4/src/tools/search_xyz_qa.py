from langchain.tools import tool
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from src.configs import Settings
from src.custom_logger import setup_logger
from src.models import SearchOutput

# 検索結果の最大取得数
MAX_SEARCH_RESULTS = 3

logger = setup_logger(__name__)


class SearchQueryInput(BaseModel):
    query: str = Field(description="検索クエリ")


@tool(args_schema=SearchQueryInput)
def search_xyz_qa(query: str) -> list[SearchOutput]:
    """
    XYZシステムの過去の質問回答ペアを検索する関数。
    """

    logger.info(f"Searching XYZ QA by query: {query}")

    qdrant_client = QdrantClient("http://localhost:6333")

    settings = Settings()
    if settings.api_provider.lower() == "azure":
        openai_client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint or "",
            api_key=settings.azure_openai_api_key or "",
            api_version=settings.azure_openai_api_version or "",
        )
        embedding_model = (
            settings.azure_openai_embedding_deployment_name
            if settings.azure_openai_embedding_deployment_name
            else "text-embedding-3-small"
        )
    else:
        openai_client = OpenAI(api_key=settings.openai_api_key)
        embedding_model = "text-embedding-3-small"

    logger.info("Generating embedding vector from input query")
    query_vector = openai_client.embeddings.create(input=query, model=embedding_model).data[0].embedding

    search_results = qdrant_client.query_points(
        collection_name="documents", query=query_vector, limit=MAX_SEARCH_RESULTS
    ).points

    logger.info(f"Search results: {len(search_results)} hits")
    outputs = []

    for point in search_results:
        outputs.append(SearchOutput.from_point(point))

    logger.info("Finished searching XYZ QA by query")

    return outputs
