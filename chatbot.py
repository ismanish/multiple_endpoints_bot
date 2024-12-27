import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

# LangChain message classes
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# The pipeline orchestrator that determines "RAG", "SQL", or "BOTH",
# and returns the final answer in `result["response"]`.
from pipeline_orchestration import process_question

##############
# File-based Database for persistence
##############
class FileChatDatabase:
    """
    A JSON file-based database to store user conversations persistently.
    Each user_id is a key; the value is a list of messages (dict form).
    Example of self.data structure:
    {
       "123": [
          {"type": "system", "content": "..."},
          {"type": "human", "content": "Hello"},
          {"type": "ai", "content": "Hi there!"},
          ...
       ],
       "default_user": [...],
       ...
    }
    """

    def __init__(self, filepath: str = "chat_data.json"):
        self.filepath = filepath
        self._load_data()

    def _load_data(self):
        """Load the entire data from disk into self.data (dict)."""
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError:
                    self.data = {}
        else:
            self.data = {}

    def _save_data(self):
        """Save self.data (dict) back to disk."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def get_user_messages(self, user_id: str, limit: Optional[int] = None) -> List[BaseMessage]:
        """
        Return up to the last `limit` messages (as BaseMessage objects), 
        or all messages if limit is None.
        """
        if user_id not in self.data:
            return []
        messages_dicts = self.data[user_id]
        
        # If limit is given, slice from the end
        if limit is not None:
            messages_dicts = messages_dicts[-limit:]
        
        # Convert stored dictionaries into actual message objects
        messages = []
        for m in messages_dicts:
            msg_type = m["type"]
            content = m["content"]
            if msg_type == "system":
                messages.append(SystemMessage(content=content))
            elif msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            else:
                # Fallback: treat as base message
                messages.append(BaseMessage(content=content))
        return messages

    def add_message(self, user_id: str, message: BaseMessage):
        """
        Append a message (BaseMessage) to the user's conversation 
        and save to disk.
        """
        if user_id not in self.data:
            self.data[user_id] = []

        # Identify the message type for saving
        msg_type = "base"
        if isinstance(message, SystemMessage):
            msg_type = "system"
        elif isinstance(message, HumanMessage):
            msg_type = "human"
        elif isinstance(message, AIMessage):
            msg_type = "ai"

        self.data[user_id].append({
            "type": msg_type,
            "content": message.content
        })

        self._save_data()


@dataclass
class ChatMemory:
    """
    In-memory chat memory for quick access (mirrors the original code).
    We'll also store the messages in FileChatDatabase for persistence.
    """
    messages: List[BaseMessage] = field(default_factory=list)
    max_messages: int = 5

    def add_message(self, message: BaseMessage):
        self.messages.append(message)
        # Keep pairs of messages (user + assistant) up to max_messages
        if len(self.messages) > self.max_messages * 2:
            self.messages = self.messages[-(self.max_messages * 2):]

    def get_conversation_history(self) -> List[BaseMessage]:
        return self.messages


class MovieChatbot:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.db = FileChatDatabase()  # file-based, persists across runs
        self.memory = ChatMemory()

        # System message describing the chatbot’s role
        system_msg = SystemMessage(content="""You are a helpful movie assistant that can:
1. Answer questions about movie plots, actors, and themes
2. Provide movie rental statistics and popularity information
3. Make personalized movie recommendations
Please be concise and friendly in your responses.""")

        # Load previous messages from JSON file
        previous_messages = self.db.get_user_messages(self.user_id)
        for msg in previous_messages:
            self.memory.add_message(msg)

        # If no messages exist for this user, add system message
        if not previous_messages:
            self.memory.add_message(system_msg)
            self.db.add_message(self.user_id, system_msg)

    def _process_response(self, question: str) -> str:
        """
        Call pipeline_orchestration.process_question(...) to handle the user's question.
        We'll optionally add a bit of context from the last 3 messages.
        """
        # Get the last 3 messages from the file-based DB
        history = self.db.get_user_messages(self.user_id, limit=3)
        context = ""
        if history:
            context = "Previous conversation context:\n"
            for msg in history:
                if isinstance(msg, HumanMessage):
                    context += f"User asked: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    context += f"Assistant answered: {msg.content}\n"

        if context:
            question = f"{context}\nCurrent question: {question}"

        # Send the question to the orchestrator
        try:
            result = process_question(question)
            return result["response"]
        except Exception as e:
            return f"I encountered an error: {str(e)}"

    def chat(self, user_input: str) -> str:
        """
        Handles user input. If asking for conversation history, returns it.
        Otherwise, routes query to pipeline_orchestration and stores the new messages.
        """
        # Check if user wants to see recent conversation
        if any(phrase in user_input.lower() for phrase in ["what did i ask", "what did we talk about", "previous conversation"]):
            full_history = self.db.get_user_messages(self.user_id)
            if not full_history:
                return "We haven't had any previous conversations yet."

            # Show the last 6 messages (3 user + 3 assistant)
            recent_msgs = full_history[-6:]
            lines = []
            for msg in recent_msgs:
                if isinstance(msg, HumanMessage):
                    lines.append(f"You asked: {msg.content}")
                elif isinstance(msg, AIMessage):
                    lines.append(f"I answered: {msg.content}")
            return "Here's our recent conversation:\n" + "\n".join(lines)

        # Normal conversation flow
        user_message = HumanMessage(content=user_input)
        self.memory.add_message(user_message)
        self.db.add_message(self.user_id, user_message)

        response = self._process_response(user_input)

        ai_message = AIMessage(content=response)
        self.memory.add_message(ai_message)
        self.db.add_message(self.user_id, ai_message)

        return response

    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Return all conversation messages (including system) from the file-based DB.
        """
        history = []
        messages = self.db.get_user_messages(self.user_id)
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                history.append({
                    "role": msg.__class__.__name__.replace("Message", "").lower(),
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()
                })
        return history


if __name__ == "__main__":
    user_id_input = input("Enter your user ID (or press Enter for default): ").strip()
    if not user_id_input:
        user_id_input = "default_user"

    chatbot = MovieChatbot(user_id=user_id_input)

    print(f"\nMovie Chatbot: Hello! I'm your movie assistant. User ID: {user_id_input}")
    print("Commands:")
    print("- Type 'exit' to end the conversation")
    print("- Type 'history' to see your chat history")
    print("- Type 'clear' to start a new conversation\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\nMovie Chatbot: Goodbye! Have a great day!")
            break

        if user_input.lower() == "history":
            chat_history = chatbot.get_chat_history()
            print("\nChat History:")
            if not chat_history:
                print("No previous messages found.")
            else:
                for msg in chat_history:
                    # Optionally skip system messages in display
                    if msg["role"] == "system":
                        continue
                    print(f"{msg['role'].title()}: {msg['content']}\n")
            continue

        if user_input.lower() == "clear":
            # Create a brand-new conversation by forging a new user_id
            # with a timestamp, effectively clearing the history
            new_user_id = f"{user_id_input}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            chatbot = MovieChatbot(user_id=new_user_id)
            print("\nMovie Chatbot: Started a new conversation!")
            continue

        response = chatbot.chat(user_input)
        print(f"\nMovie Chatbot: {response}\n")
