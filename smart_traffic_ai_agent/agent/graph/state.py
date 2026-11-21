from typing import Annotated, Any, Dict, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state passed between the nodes of the Smart Traffic AI Agent.

    The state contains:
    - Conversation messages
    - Retrieved RAG context
    - Results returned by tools
    """

    # ========================================================
    # Conversation
    # ========================================================

    messages: Annotated[List[BaseMessage], add_messages]

    # ========================================================
    # RAG Context
    # ========================================================

    rag_context: str

    # ========================================================
    # Tool Results
    # ========================================================

    tool_results: Dict[str, Any]
