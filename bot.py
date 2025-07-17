import streamlit as st
from utils import write_message
from agent import generate_response


# Page Config
st.set_page_config("Ebert", page_icon=":movie_camera:")

st.title("EduStackBot - Intelligent Student Error Analysis")

st.markdown("""
EduStackBot is a chatbot-powered educational assistant designed to help analyze student performance and misconceptions.
It leverages Neo4j graph queries, GPT-4o language intelligence, and the Newman Error Analysis (NEA) framework to:

- Understand student responses and performance across assessments
- Identify patterns in conceptual, procedural, or transformation errors
- Recommend personalized learning targets and scaffolds
- Provide teachers with actionable insights into student learning stages

Simply ask your question about student data or performance, and EduStackBot will guide you with data-driven answers.
""")


# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the EduStackBot Chatbot!  How can I help you?"},
    ]

# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        # Call the agent
        response = generate_response(message)
        write_message('assistant', response)
        


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    write_message('user', prompt)

    # Generate a response
    handle_submit(prompt)
