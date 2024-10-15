import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
import openai



# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_key"]

st.header("Chat with Anh's Work History.")

# Initialize the chat message history if not already done
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Anh!"},
    ]

# Function to load data and create the index
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        # Load documents from the specified directory
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        
        # Create an LLM instance
        llm = OpenAI(model="gpt-4o-mini", temperature=0.7, system_prompt=(
            "You are a helpful assistant for answering questions about Anh's work and professionalism."
            "Your answers must be helpful and enthusiastic. You must explain how Anh is a great teammate."
            "You must be concise and clear. You must be friendly."
            "You must use humor."
            "You must be exuberant about Anh's work and fun personality."
            "You are an expert on the career and life of Anh."
            "Keep your answers technical and based on "
            "facts – do not hallucinate job dates, company names, or job titles."
        ))

        # Create the index from the documents and pass the LLM directly
        index = VectorStoreIndex.from_documents(docs, llm=llm)
        return index

# Load the data and create the index
index = load_data()

# Create a chat engine from the index
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Get user input and save it to the chat history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If the last message is from the user, generate a response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(st.session_state.messages[-1]["content"])
            st.write(response.response)
            # Add the assistant's response to the message history
            st.session_state.messages.append({"role": "assistant", "content": response.response})