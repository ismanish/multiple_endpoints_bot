import uuid
import gradio as gr
from chatbot import MovieChatbot, FileChatDatabase, SystemMessage, HumanMessage, AIMessage

"""
gradio_chatbot.py

To run:
  python gradio_chatbot.py
Then open the printed local URL in your browser.
"""

#######################
# Utility function to convert the stored conversation
# into the "Gradio chatbot" format:
# a list of [("User message", "Bot reply"), ...]
#######################
def load_history_as_chatlog(user_id: str) -> list[tuple[str, str]]:
    """
    Returns the conversation in a format suitable for Gradio's Chatbot component:
    [
      ("User says", "Assistant replies"),
      ...
    ]
    We'll pair them up: if there's an odd number of messages, the last won't have a pair yet.
    """
    db = FileChatDatabase()
    messages = db.get_user_messages(user_id)
    # We'll collect pairs: user -> assistant
    chatlog = []
    buffer_user = None
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            # If we had a pending user message in buffer, flush it (unlikely, but safe)
            if buffer_user is not None:
                # means we never saw an AI reply for the previous user...
                chatlog.append((buffer_user, ""))
            buffer_user = msg.content
        elif isinstance(msg, AIMessage):
            # Pair up with the user message
            if buffer_user is not None:
                chatlog.append((buffer_user, msg.content))
                buffer_user = None
            else:
                # If we somehow have an AI message first, or extra, treat user as empty
                chatlog.append(("", msg.content))
        # We skip system messages when displaying the chat to the user
    
    # If there's an unpaired user message left, pair it with an empty AI
    if buffer_user is not None:
        chatlog.append((buffer_user, ""))
    
    return chatlog


#######################
# Main Gradio Interface
#######################
def start_session(user_id: str, chatbot_history: list[tuple[str, str]]):
    """
    Initializes or loads the MovieChatbot using the provided user_id.
    If user_id is blank, we generate a random one.
    Then we load the conversation from the JSON file (if any).
    """
    if not user_id.strip():
        # Create a random user ID
        user_id = str(uuid.uuid4())[:8]  # short random ID

    # Initialize chatbot (this will load existing conversation if found)
    bot = MovieChatbot(user_id=user_id)

    # Load past conversation into the Gradio Chatbot format
    chatlog = load_history_as_chatlog(user_id)

    if chatlog:
        status_message = f"Welcome back, **{user_id}**! Your conversation has been restored."
    else:
        status_message = f"Welcome, **{user_id}**! A new conversation has been created."

    return user_id, chatlog, status_message


def user_message_submit(user_text: str, chatbot_history: list[tuple[str, str]], user_id: str):
    """
    When the user sends a message, pass it to the MovieChatbot, get a response,
    and update the Chatbot UI.
    """
    if not user_text.strip():
        return chatbot_history, ""  # no change if user enters empty text

    # Re-initialize or load the chatbot for safety
    bot = MovieChatbot(user_id=user_id)

    # We send user_text to the chatbot
    bot_reply = bot.chat(user_text)

    # Append the new user->assistant pair to the conversation
    chatbot_history.append((user_text, bot_reply))
    return chatbot_history, ""


def clear_session(user_id: str, chatbot_history: list[tuple[str, str]]):
    """
    Clears the conversation by forging a brand-new user_id with a timestamp.
    """
    import time
    new_user_id = f"{user_id}_{int(time.time())}"
    # Re-init
    bot = MovieChatbot(user_id=new_user_id)
    # Return empty chatlog
    return new_user_id, [], f"Started a new conversation for user_id={new_user_id}"


def build_interface():
    """
    Creates a fancy Gradio Blocks interface for the movie chatbot.
    """
    with gr.Blocks(
        title="Movie Chatbot",
        css="""
        #title {
            text-align: center;
            font-size: 2rem;
            font-weight: bold;
            margin-top: 20px;
        }
        #subtext {
            text-align: center;
            margin-bottom: 20px;
        }
        .gr-text-input {
            min-height: 80px !important;
        }
        """
    ) as demo:
        # We maintain some states:
        user_id_state = gr.State("")          # current user_id
        chatbot_history_state = gr.State([])  # conversation in Gradio format

        gr.Markdown("<h1 id='title'>🎬 MovieMind AI</h1>")
        gr.Markdown("<div id='subtext'>A helpful assistant for movies, rentals, and more!</div>")

        # Row for user ID input + "Start" button
        with gr.Row():
            user_id_input = gr.Textbox(
                label="User ID",
                placeholder="Enter your user ID or leave blank for a random ID",
                show_label=True
            )
            start_button = gr.Button("Start / Load Chat", variant="primary")

        status_display = gr.Markdown("Please enter your user ID above and click Start.", elem_id="status")

        # Main chatbot display
        chatbot_display = gr.Chatbot(
            label="MovieMind Chat",
            value=[],
            elem_id="chatbot_box"
        )

        # Textbox row for user to type next query
        with gr.Row():
            user_text_input = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=2
            )
            send_button = gr.Button("Send", variant="secondary")

        # Clear conversation button
        clear_button = gr.Button(
            "Clear Conversation",
            variant="stop"
        )

        # 1) Start button sets user_id_state, loads conversation
        start_button.click(
            fn=start_session,
            inputs=[user_id_input, chatbot_history_state],
            outputs=[user_id_state, chatbot_display, status_display]
        )

        # 2) Send user message
        user_text_input.submit(
            fn=user_message_submit,
            inputs=[user_text_input, chatbot_display, user_id_state],
            outputs=[chatbot_display, user_text_input],
        )
        send_button.click(
            fn=user_message_submit,
            inputs=[user_text_input, chatbot_display, user_id_state],
            outputs=[chatbot_display, user_text_input],
        )

        # 3) Clear conversation
        clear_button.click(
            fn=clear_session,
            inputs=[user_id_state, chatbot_display],
            outputs=[user_id_state, chatbot_display, status_display]
        )

    return demo


if __name__ == "__main__":
    interface = build_interface()
    # Launch with share=False (default), or set share=True if you want a public link
    interface.launch(server_name="0.0.0.0", server_port=7860)
