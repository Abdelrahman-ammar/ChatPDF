from dotenv import load_dotenv
import os
from utils import ( get_text_chunks,
                   create_knowledge_base,
                   user_question,
                   read_pdf)
import streamlit as st

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")



st.markdown(
    """
    <style>
    .user-message {
        background-color: inherit;
        padding: 10px;
        border-radius: 10px;
        margin: 10px;
        text-align: right;
        width: fit-content;
        margin-left: auto;
        border: 1px solid #d6d6d6;
    }
    .bot-message {
        background-color: inherit;
        padding: 10px;
        border-radius: 10px;
        margin: 10px;
        text-align: left;
        width: fit-content;
        margin-right: auto;
        border: 1px solid #d6d6d6;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None  # To hold the user input temporarily


st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf" )
submit_button = st.sidebar.button("Submit")

if submit_button and uploaded_file is not None:
    with st.spinner("Processing PDF and creating knowledge base..."):
        pdf_text = read_pdf(uploaded_file)
        chunks = get_text_chunks(pdf_text)
        
        create_knowledge_base(chunks)
        st.sidebar.success("PDF uploaded and knowledge base created successfully!")
        st.session_state.pdf_processed = True

st.title("Chat PDF using Gemini")

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for chat in st.session_state.chat_history:
    st.markdown(f"<div class='user-message'><strong>You:</strong> {chat['question']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-message'><strong>Bot:</strong> {chat['response']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)



if st.session_state.pdf_processed:
    
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False
    
    if st.session_state.clear_input:

        user_input = st.text_input("Ask a question:",value="" ,key="user_input")
        st.session_state.clear_input = False
    
    else:
        user_input = st.text_input("Ask a question:",key="user_input")

    if st.button("Send"):
        if user_input:

            st.session_state.pending_question = user_input 
            st.session_state.clear_input = True
            
    # if st.session_state.pending_question:
            bot_response = user_question(st.session_state.pending_question)

            st.session_state.chat_history.append({
                "question": st.session_state.pending_question,
                "response": bot_response
            })
            st.session_state.pending_question = None

            st.rerun()
            
else:
    st.info("Please upload a PDF and submit to create a knowledge base before asking questions.")


if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()    


# texts = "Hello my name is samir , I work as an Ai instructor , I love harry potter"
# chunks = get_text_chunks(texts)
# create_knowledge_base(chunks=chunks)
# print(user_question("what is the name?"))


