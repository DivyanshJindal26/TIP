import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import os
import json
from langchain.embeddings.base import Embeddings

# Set your OpenRouter API key here
api_key =''

st.markdown("""## RAG CHATBOT """)

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
    
# Extract user string from URL
try:
    user_string = st.query_params['str']
except:
    pass

# Function to get text from a web page
def get_web_page_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure we notice bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from HTML
        text = soup.get_text(separator='\n')
        return text
    except requests.RequestException as e:
        st.error(f"Error fetching the web page: {e}")
        return ""

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save vector store using SentenceTransformers
def get_vector_store(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

# Function to get the conversational chain using OpenAI via OpenRouter
def get_conversational_chain():
    def generate_response(prompt):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = "https://openrouter.ai/api/v1/completions"
        data = {
            "model": "gpt-4",
            "prompt": prompt,
            "temperature": 0.3
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()['choices'][0]['text']
    
    return generate_response

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts)

    def embed_query(self, text):
        return self.model.encode([text])[0]

# Function to handle user input and manage chat history
def user_input(user_question):
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    chain = get_conversational_chain()
    full_prompt = f"Context: {context}\nQuestion: {user_question}"
    response = chain(full_prompt)
    
    chat_entry = {
        "question": user_question,
        "response": response
    }
    
    st.session_state['chat_history'].append(chat_entry)
    save_chat_history(st.session_state['chat_history'])
    
    st.write("Reply: ", response)
    st.markdown(
    f"""
    <style>
    .button {{
        display: inline-block;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        color: white;
        background-color: #4CAF50;
        border: none;
        border-radius: 5px;
        text-align: center;
        text-decoration: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }}

    .button:hover {{
        background-color: #45a049;
    }}
    </style>
    <a href="http://localhost:8501" target="_self">
        <div class="button">Type another question</div>
    </a>
    """,
    unsafe_allow_html=True
)

# Functions to handle chat history
def load_chat_history():
    if os.path.exists('chat_history.json'):
        with open('chat_history.json', 'r') as f:
            return json.load(f)
    return []

def save_chat_history(history):
    with open('chat_history.json', 'w') as f:
        json.dump(history, f, indent=4)

def clear_chat_history():
    if os.path.exists('chat_history.json'):
        os.remove('chat_history.json')
    st.session_state['chat_history'] = []
    
# function to edit the current link
def load_link():
    if os.path.exists('link.json'):
        with open('link.json', 'r') as f:
            return json.load(f)
    return []

def save_link(link):
    with open('link.json','w') as f:
        json.dump(link,f,indent=4)

# Main function
st.header("Ask me Anything....")


with st.sidebar:
    st.title("Menu:")
    link = load_link()[0]
    if link:
        url = st.text_input("Enter the URL of the Web Page", key="url_input", value=link)
    else:
        url = st.text_input("Enter the URL of the Web Page", key="url_input")
    if st.button("Submit & Process", key="process_button") and url and api_key:
        with st.spinner("Processing..."):
            web_text = get_web_page_text(url)
            if web_text:
                text_chunks = get_text_chunks(web_text)
                get_vector_store(text_chunks)
                st.success("Done")
                save_link(url)

with st.spinner("Loading..."):
    # Input for user question with default value from URL if present
    try:
        user_question = st.text_input(
            "Ask a Question from the Web Page Content",
            value=user_string,  # Default value set to user_string
            key="user_question"
        )
    except:
        user_question = st.text_input(
            "Ask a Question from the Web Page Content",
            key="user_question"
            )
    if user_question and api_key:
        user_input(user_question)
        latest_entry = st.session_state['chat_history'][-1]
        
        st.write("---")

    if st.button("Clear Chat History"):
        clear_chat_history()
        st.success("Chat history cleared!")

    if not st.session_state['chat_history']:
        st.session_state['chat_history'] = load_chat_history()

    if st.session_state['chat_history'][:-1]:
        st.subheader("Chat History")
        for entry in reversed(st.session_state['chat_history'][:-1]):
            st.write(f"**Question:** {entry['question']}")
            st.write(f"**Response:** {entry['response']}")
            st.write("---")
