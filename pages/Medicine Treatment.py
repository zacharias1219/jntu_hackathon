import streamlit as st
from utilities.icon import page_icon
import os
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="Medicine Treatment",
    layout="wide",
    page_icon="💊",
    initial_sidebar_state="collapsed",
)

st.title("💊 Medicine Treatment")

# Function to get Groq completions
def get_groq_completions(user_content):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {
                "role": "system",
                "content": "You are a chat bot in an application where the app provides services for people suffering with auto immune diseases . Your job as a chat bot is too help people find the right medicines for the condition that the patient has and the ones based on doctor recommendation"
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        temperature=0.5,
        max_tokens=5640,
        top_p=1,
        stream=True,
        stop=None,
    )

    result = ""
    for chunk in completion:
        result += chunk.choices[0].delta.content or ""

    return result

# Streamlit interface
def main():
    st.title("🤖 Medicine Recommendation")
    user_content = st.text_input("Help us find the right medicine for you")
    user_data = st.file_uploader(label="Upload doctor report",type=['png','jpg','pdf'])

    if st.button("Enter"):
        if not user_content:
                st.warning("Type something to generate conversation")
                return
        st.info("Helping you.")
        generated_titles = get_groq_completions(user_content)
        st.success("Here you GO!")

        # Display the generated titles
        st.markdown("Conversation:")
        st.text_area("", value=generated_titles, height=200)

if __name__ == "__main__":
    main()