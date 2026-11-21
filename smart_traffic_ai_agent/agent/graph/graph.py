import json

from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END

from agent.graph.state import AgentState
from agent.graph.nodes import (
    agent_node,
    should_continue,
)
from agent.graph.tools import tools

TOOLS_BY_NAME = {
    tool.name: tool
    for tool in tools
}


def tool_node(state: AgentState):
    """
    Execute requested tools and return ToolMessage objects with names.

    Groq's GPT-OSS tool protocol requires named tool messages.
    """

    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in getattr(last_message, "tool_calls", []):
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call["id"]

        tool = TOOLS_BY_NAME.get(tool_name)

        if tool is None:
            result = {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
            }
        else:
            try:
                result = tool.invoke(tool_args)
            except Exception as exc:
                result = {
                    "success": False,
                    "error": str(exc),
                }

        tool_messages.append(
            ToolMessage(
                content=json.dumps(result, default=str),
                name=tool_name,
                tool_call_id=tool_call_id,
            )
        )

    return {
        "messages": tool_messages
    }


def build_agent_graph():
    """
    Build the Smart Traffic AI Agent graph.

    Flow:

        START
          ↓
      agent_node
          ↓
      should_continue
        ↙       ↘
    tools       END
      ↓
    agent_node
    """

    # Create the graph
    graph = StateGraph(AgentState)

    # ---------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------

    # Main AI Agent node
    graph.add_node("agent", agent_node)

    # Tool execution node
    graph.add_node("tools", tool_node)

    # ---------------------------------------------------------
    # Entry point
    # ---------------------------------------------------------

    graph.add_edge(START, "agent")

    # ---------------------------------------------------------
    # Conditional routing
    # ---------------------------------------------------------

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # ---------------------------------------------------------
    # After executing a tool
    # ---------------------------------------------------------

    graph.add_edge("tools", "agent")

    # Compile graph
    return graph.compile()


# -------------------------------------------------------------
# Global compiled graph
# -------------------------------------------------------------

agent_graph = build_agent_graph()
