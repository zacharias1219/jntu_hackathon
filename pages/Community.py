import streamlit as st
from utilities.icon import page_icon
import os
from groq import Groq
from typing import Generator

st.set_page_config(
    page_title="Community",
    layout="wide",
    page_icon="💬",
    initial_sidebar_state="collapsed",
)

st.title("💬 Community")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = {
    "gemma-7b-it": {"name": "SARIKA", "tokens": 8192, "developer": "Google"},
    "llama2-70b-4096": {"name": "ROSHNI", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "RICHARD", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "SAKETH", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "RAHUL", "tokens": 32768, "developer": "Mistral"},
}

# Layout for model selection and max_tokens slider

model_option = st.selectbox(
        "Choose a friend:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=4  # Default to mixtral
    )

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option



# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=4096,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})