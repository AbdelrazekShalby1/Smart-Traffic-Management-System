from typing import List

from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from config.settings import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    QDRANT_DISTANCE,
)


def get_qdrant_client() -> QdrantClient:
    """
    Create and return a Qdrant client.
    """

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
    )

    print("[QDRANT] Connected")

    return client


def _vector_config(collection_info):
    vectors = collection_info.config.params.vectors

    if isinstance(vectors, dict):
        if "" in vectors:
            return vectors[""]
        return next(iter(vectors.values()))

    return vectors


def verify_qdrant_collection(client: QdrantClient):
    """
    Verify that the configured Qdrant collection exists and is
    compatible with the configured embedding model.
    """

    try:
        collection_info = client.get_collection(QDRANT_COLLECTION)
    except Exception as exc:
        message = str(exc)

        if "401" in message or "Unauthorized" in message:
            raise RuntimeError(
                "Qdrant authentication failed. Check QDRANT_API_KEY."
            ) from exc

        if "404" in message or "Not found" in message:
            raise RuntimeError(
                f"Qdrant collection '{QDRANT_COLLECTION}' does not exist."
            ) from exc

        raise RuntimeError(
            f"Could not access Qdrant collection '{QDRANT_COLLECTION}': "
            f"{message}"
        ) from exc

    vector_config = _vector_config(collection_info)

    size = getattr(vector_config, "size", None)
    distance = getattr(vector_config, "distance", None)

    if size != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Qdrant collection '{QDRANT_COLLECTION}' has vector size "
            f"{size}, expected {EMBEDDING_DIMENSION} for {EMBEDDING_MODEL}."
        )

    distance_value = getattr(distance, "value", distance)

    if str(distance_value).lower() != QDRANT_DISTANCE.lower():
        raise RuntimeError(
            f"Qdrant collection '{QDRANT_COLLECTION}' uses distance "
            f"{distance_value}, expected {QDRANT_DISTANCE}."
        )

    return collection_info


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Create the embedding model used for the knowledge base.

    The model is configured through the .env file:

    EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

    This model produces 768-dimensional embeddings.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def get_vector_store() -> QdrantVectorStore:
    """
    Connect to the existing Qdrant collection.
    """

    client = get_qdrant_client()
    verify_qdrant_collection(client)
    embeddings = get_embeddings()

    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )


def get_retriever(k: int = 4):
    """
    Return a retriever that searches the knowledge base.
    """

    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_kwargs={
            "k": k,
        }
    )


def retrieve_knowledge(
    query: str,
    k: int = 4,
) -> List[Document]:
    """
    Retrieve the most relevant documents for a query.
    """

    if not query or not query.strip():
        return []

    print("[RAG] Retrieving knowledge")

    retriever = get_retriever(k=k)

    documents = retriever.invoke(query)

    return documents


def format_retrieved_context(
    documents: List[Document],
) -> str:
    """
    Convert retrieved documents into text
    that can be passed to the LLM.
    """

    if not documents:
        return ""

    context_parts = []

    for i, document in enumerate(documents, start=1):

        source = document.metadata.get(
            "source",
            "unknown",
        )

        content = document.page_content.strip()

        if not content:
            continue

        context_parts.append(
            f"[Source {i}: {source}]\n"
            f"{content}"
        )

    return "\n\n---\n\n".join(context_parts)


def retrieve_context(
    query: str,
    k: int = 4,
) -> str:
    """
    Retrieve relevant knowledge and return
    it as formatted context for the AI agent.
    """

    documents = retrieve_knowledge(
        query=query,
        k=k,
    )

    return format_retrieved_context(documents)
