from typing import Literal

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

from agent.graph.state import AgentState
from agent.graph.tools import tools
from config.settings import (
    GROQ_API_KEY,
    LLM_MODEL,
    validate_agent_settings,
)


# ============================================================
# LLM
# ============================================================

validate_agent_settings()

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=0
)

# Give the LLM access to the Smart Traffic tools
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are the Smart Traffic Intelligence Agent.

You are an AI Agent integrated with a real Smart Traffic
Management Computer Vision System.

The CV system can provide information such as:

- Detected vehicles
- Active vehicles
- Vehicle types
- Vehicle tracking IDs
- Vehicle speeds
- Speeding violations
- Red-light violations
- Traffic density
- Vehicle counting
- Emergency vehicles
- License plates
- Traffic violations
- Historical traffic statistics

Your responsibilities:

1. Answer the user's questions about the traffic system.
2. Use tools whenever the user asks for live, dynamic,
   measured, or database information.
3. Never invent traffic measurements, vehicle counts,
   speeds, violations, or emergency information.
4. Clearly distinguish between:
   - Observed data
   - Historical data
   - Recommendations
   - Predictions
5. Use the knowledge base / RAG context when answering
   questions about traffic laws, regulations, protocols,
   SOPs, or system documentation.
6. If the required information is unavailable, clearly
   say that it is unavailable.
7. Do not claim that an action was performed unless the
   corresponding tool actually confirms that it was performed.
8. Potentially consequential actions such as changing
   traffic signals must require approval.
9. Give concise but useful answers.
10. When useful, explain the reason behind your recommendation.

You are not the CV detector itself.
You are the intelligence and decision-making layer
on top of the traffic management system.
"""


# ============================================================
# Agent Node
# ============================================================

def agent_node(state: AgentState):
    """
    Main AI Agent node.

    The node:
    1. Receives the current conversation state.
    2. Adds the system instructions.
    3. Adds RAG context if available.
    4. Sends everything to the LLM.
    5. The LLM can either:
       - answer directly
       - request one or more tools
    """

    messages = state["messages"]

    # --------------------------------------------------------
    # Build system message
    # --------------------------------------------------------

    system_content = SYSTEM_PROMPT

    # Add RAG context when available
    rag_context = state.get("rag_context", "")

    if rag_context:
        system_content += f"""

Relevant knowledge retrieved from the knowledge base:

---------------- RAG CONTEXT ----------------

{rag_context}

------------------------------------------------

Use this context when it is relevant.
Do not treat RAG information as live traffic data.
"""

    # --------------------------------------------------------
    # Prepare messages
    # --------------------------------------------------------

    model_messages = [
        SystemMessage(content=system_content),
        *messages,
    ]

    # --------------------------------------------------------
    # Call LLM
    # --------------------------------------------------------

    response = llm_with_tools.invoke(model_messages)

    # --------------------------------------------------------
    # Return new message
    # --------------------------------------------------------

    return {
        "messages": [response]
    }


# ============================================================
# Routing Logic
# ============================================================

def should_continue(
    state: AgentState,
) -> Literal["tools", "end"]:
    """
    Decide whether the graph should:

        Agent → Tools

    or:

        Agent → END

    If the LLM requested tool calls, route to the tools node.
    Otherwise, finish the conversation.
    """

    messages = state["messages"]

    if not messages:
        return "end"

    last_message = messages[-1]

    # --------------------------------------------------------
    # Check whether the LLM requested tools
    # --------------------------------------------------------

    if getattr(last_message, "tool_calls", None):
        return "tools"

    # --------------------------------------------------------
    # No tool call → final answer
    # --------------------------------------------------------

    return "end"
