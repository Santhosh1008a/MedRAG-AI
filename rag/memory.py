from typing import List, Dict

class ConversationMemory:
    """
    Manages simple conversational memory for multi-turn chats.
    Stores chat history as a list of dicts: {"role": "user" | "assistant", "content": str}
    """
    def __init__(self, max_history_turns: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_history_turns = max_history_turns

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message})
        self._trim_history()

    def add_assistant_message(self, message: str):
        self.history.append({"role": "assistant", "content": message})
        self._trim_history()

    def get_history_string(self) -> str:
        """Formats the chat history into a string for the LLM prompt."""
        if not self.history:
            return "No previous history."
            
        history_str = ""
        for turn in self.history:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_str += f"{role}: {turn['content']}\n"
        return history_str.strip()
        
    def get_history_list(self) -> List[Dict[str, str]]:
        return self.history

    def clear(self):
        self.history = []

    def _trim_history(self):
        """Keeps the history within the maximum allowed turns (user + assistant = 1 turn)."""
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            # Remove the oldest messages
            self.history = self.history[-max_messages:]
