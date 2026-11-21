from typing import List

from langchain_core.messages import BaseMessage


class ConversationMemory:
    """
    Simple in-memory conversation history
    for the Smart Traffic AI Agent.
    """

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: List[BaseMessage] = []

    def add_message(self, message: BaseMessage):
        """
        Add a message to the conversation history.
        """

        self.messages.append(message)

        # Keep only the latest messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def add_messages(self, messages: List[BaseMessage]):
        """
        Add multiple messages.
        """

        self.messages.extend(messages)

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self) -> List[BaseMessage]:
        """
        Return the current conversation history.
        """

        return self.messages.copy()

    def clear(self):
        """
        Clear conversation history.
        """

        self.messages.clear()

    def __len__(self):
        return len(self.messages)


# Global memory instance
conversation_memory = ConversationMemory(
    max_messages=20
)