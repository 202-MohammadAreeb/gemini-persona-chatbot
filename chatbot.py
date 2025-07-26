
import os
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv
import pdfplumber

# === Setup ===
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# === Memory ===
chat_history = []
user_memory = {"name": None, "facts": []}
uploaded_doc_text = ""

# === Personas ===
persona_presets = {
    "Friendly Assistant": "You are a friendly and helpful assistant named Nova.",
    "Sarcastic AI": "You are a sarcastic AI who uses dry humor.",
    "Technical Support": "You are a professional technical support assistant.",
    "Fantasy Game Master": "You are a fantasy RPG dungeon master narrating adventures.",
    "Medical Expert": "You are a precise and calm medical assistant."
}

# === PDF Reading ===
def extract_text_from_file(file_obj):
    if file_obj.name.endswith(".pdf"):
        with pdfplumber.open(file_obj.name) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file_obj.name.endswith(".txt"):
        return file_obj.read().decode()
    return ""

# === Core Chat Logic ===
def respond(user_input, persona, file, chat_display):
    global chat_history, user_memory, uploaded_doc_text

    # Load persona prompt
    persona_prompt = persona_presets.get(persona, "You are a helpful assistant.")

    # Handle file upload
    if file:
        uploaded_doc_text = extract_text_from_file(file)
        user_input = f"Based on this document:\n\n{uploaded_doc_text}\n\nQuestion: {user_input}"

    # Memory: Name
    if "my name is" in user_input.lower():
        name = user_input.split("my name is")[-1].strip().split()[0].capitalize()
        user_memory["name"] = name
        reply = f"Nice to meet you, {name}! I'll remember your name during this session."
        chat_display.append({"role": "user", "content": user_input})
        chat_display.append({"role": "assistant", "content": reply})
        return chat_display, ""

    # Memory: Reminder
    if "remind me" in user_input.lower():
        user_memory["facts"].append(user_input)
        reply = "Got it. I've noted that down for you."
        chat_display.append({"role": "user", "content": user_input})
        chat_display.append({"role": "assistant", "content": reply})
        return chat_display, ""

    # Add memory to prompt
    if user_memory["name"]:
        persona_prompt += f" The user's name is {user_memory['name']}."

    # Build prompt from history
    prompt = persona_prompt + "\n\n"
    for msg in chat_display:
        role = msg["role"].capitalize()
        content = msg["content"]
        prompt += f"{role}: {content}\n"
    prompt += f"User: {user_input}\nAssistant:"

    # Get response
    response = model.generate_content(prompt)
    reply = response.text.strip()

    chat_display.append({"role": "user", "content": user_input})
    chat_display.append({"role": "assistant", "content": reply})
    return chat_display, ""



# === Reset Chat ===
def reset_chat(persona):
    global chat_history, user_memory, uploaded_doc_text
    chat_history = []
    user_memory = {"name": None, "facts": []}
    uploaded_doc_text = ""
    return []

# === UI ===
with gr.Blocks(css="""
    .gr-chatbot {
        background-color: #f7f7f8;
        border-radius: 12px;
        padding: 16px;
        font-family: 'Segoe UI', sans-serif;
    }

    .gr-chatbot .message.user {
        background-color: #e5f3ff;
        color: #1a1a1a;
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 6px;
        align-self: flex-end;
        max-width: 80%;
    }

    .gr-chatbot .message.assistant {
        background-color: #ffffff;
        color: #1a1a1a;
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 6px;
        align-self: flex-start;
        max-width: 80%;
    }

    .gr-textbox, .gr-button {
        font-size: 16px;
    }

    footer {
        display: none;
    }
""") as demo:

    # --- Header ---
    gr.Markdown(
        """
        <div style="text-align:center; font-size: 28px; font-weight: bold; margin-bottom: 10px;">
            💬 Gemini Chatbot
        </div>
        <div style="text-align:center; font-size: 16px; color: gray;">
            Powered by Google's Gemini Pro · Persona aware · Document chat ready
        </div>
        """,
        elem_id="title"
    )

    # --- Controls ---
    with gr.Row():
        persona_select = gr.Dropdown(
            label="Choose Persona",
            choices=list(persona_presets.keys()),
            value="Friendly Assistant",
            interactive=True
        )
        clear_btn = gr.Button("🗑️ Reset", size="sm")
        file_upload = gr.File(label="📎 Upload PDF or TXT", file_types=[".pdf", ".txt"])

    # --- Chatbot Display ---
    chatbot_ui = gr.Chatbot(label="", height=480, type="messages")

    # --- Message Input ---
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Message Gemini...",
            show_label=False,
            scale=5
        )
        send_btn = gr.Button("Send", scale=1)

    # --- Hooks ---
    send_btn.click(
        fn=respond,
        inputs=[user_input, persona_select, file_upload, chatbot_ui],
        outputs=[chatbot_ui, user_input]
    )

    user_input.submit(
        fn=respond,
        inputs=[user_input, persona_select, file_upload, chatbot_ui],
        outputs=[chatbot_ui, user_input]
    )

    clear_btn.click(fn=reset_chat, inputs=[persona_select], outputs=[chatbot_ui])

    # --- Footer ---
    gr.Markdown(
        "<div style='text-align: center; font-size: 13px; color: gray;'>Made with ❤️ using Gradio & Gemini Pro</div>"
    )
    
if __name__ == "__main__":
    demo.launch()
