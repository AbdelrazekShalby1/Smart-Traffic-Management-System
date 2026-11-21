from langchain_core.messages import HumanMessage

from agent.graph.graph import agent_graph
from agent.memory.memory import conversation_memory
from agent.rag.retriever import retrieve_context


RAG_KEYWORDS = {
    "law",
    "laws",
    "regulation",
    "regulations",
    "rule",
    "rules",
    "procedure",
    "procedures",
    "protocol",
    "protocols",
    "sop",
    "documentation",
    "manual",
    "policy",
    "policies",
    "emergency procedure",
    "emergency procedures",
}


def should_use_rag(user_message: str) -> bool:
    """
    Use RAG only for static knowledge questions.
    Live traffic facts must come from tools.
    """

    text = user_message.lower()
    return any(keyword in text for keyword in RAG_KEYWORDS)


def run_agent(user_message: str) -> str:
    """
    Run the Smart Traffic AI Agent
    with conversation memory.
    """

    if not user_message or not user_message.strip():
        return "Please enter a valid question."

    # Add user message to memory
    human_message = HumanMessage(
        content=user_message
    )

    conversation_memory.add_message(
        human_message
    )

    rag_context = ""

    if should_use_rag(user_message):
        try:
            rag_context = retrieve_context(user_message)
        except Exception as exc:
            return f"RAG retrieval failed: {exc}"

    # Get conversation history
    messages = conversation_memory.get_messages()

    # Run LangGraph agent
    result = agent_graph.invoke(
        {
            "messages": messages,
            "rag_context": rag_context,
            "tool_results": {},
        }
    )

    # Get the latest AI response
    response_messages = result.get(
        "messages",
        []
    )

    if not response_messages:
        return "No response was generated."

    final_message = response_messages[-1]

    response_text = final_message.content

    # Save AI response to memory
    conversation_memory.add_message(
        final_message
    )

    return response_text


def main():
    """
    Simple command-line interface
    for testing the agent.
    """

    print("=" * 60)
    print("Smart Traffic AI Agent")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("Goodbye!")
            break

        try:
            response = run_agent(
                user_input
            )

            print("\nAgent:")
            print(response)

        except Exception as e:
            print(
                f"\nAgent Error: {e}"
            )


if __name__ == "__main__":
    main()
