import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import os
import json
from langchain.embeddings.base import Embeddings

# Set up Streamlit page
st.set_page_config(page_title="Document Genie", layout="wide")

# Set your OpenRouter API key here
api_key = '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'

st.markdown("""## RAG CHATBOT """)

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

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
    # Use SentenceTransformers to generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can replace with another model if you prefer
    embeddings = model.encode(text_chunks)
    
    # Use FAISS to store and search embeddings
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

# Function to get the conversational chain using OpenAI via OpenRouter
def get_conversational_chain():
    # Using OpenRouter to interact with OpenAI models
    def generate_response(prompt):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = "https://openrouter.ai/api/v1/completions"
        data = {
            "model": "gpt-4",  # Or use 'gpt-3.5-turbo' if preferred
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
    
# Function to create and save vector store using SentenceTransformer
def get_vector_store(text_chunks):
    # Use the custom embeddings class
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    
    # Use FAISS to store and search embeddings
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

# Function to handle user input and manage chat history
def user_input(user_question):
    # Use the custom embeddings class
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

    # Load the FAISS vector store and search for similar documents
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Combine the relevant document contexts
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Get the conversational chain (ChatGPT via OpenRouter)
    chain = get_conversational_chain()
    
    # Prepare the prompt for the model
    full_prompt = f"Context: {context}\nQuestion: {user_question}"
    response = chain(full_prompt)
    
    # Prepare and save chat entry
    chat_entry = {
        "question": user_question,
        "response": response
    }
    
    # Update chat history in session state
    st.session_state['chat_history'].append(chat_entry)
    save_chat_history(st.session_state['chat_history'])
    
    # Display the response
    st.write("Reply: ", response)

# Functions to handle chat history
def load_chat_history():
    """Load chat history from a JSON file."""
    if os.path.exists('chat_history.json'):
        with open('chat_history.json', 'r') as f:
            return json.load(f)
    return []

def save_chat_history(history):
    """Save chat history to a JSON file."""
    with open('chat_history.json', 'w') as f:
        json.dump(history, f, indent=4)

# Function to clear chat history
def clear_chat_history():
    """Clear the chat history."""
    if os.path.exists('chat_history.json'):
        os.remove('chat_history.json')
    st.session_state['chat_history'] = []

# Main function
def main():
    st.header("Ask me Anything....")

    # Input for user question (Moved above chat history)
    user_question = st.text_input("Ask a Question from the Web Page Content", key="user_question")

    # Process user input after displaying chat history
    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question)
        # Display the response of the latest question right after the input box
        latest_entry = st.session_state['chat_history'][-1]
        
        st.write("---")

    # Clear chat history button above the chat history
    if st.button("Clear Chat History"):
        clear_chat_history()
        st.success("Chat history cleared!")

    # Load existing chat history at the start
    if not st.session_state['chat_history']:
        st.session_state['chat_history'] = load_chat_history()

    # Display chat history (excluding the latest question and response)
    if st.session_state['chat_history'][:-1]:
        st.subheader("Chat History")
        for entry in reversed(st.session_state['chat_history'][:-1]):
            st.write(f"**Question:** {entry['question']}")
            st.write(f"**Response:** {entry['response']}")
            st.write("---")

    with st.sidebar:
        st.title("Menu:")
        url = st.text_input("Enter the URL of the Web Page", key="url_input")
        if st.button("Submit & Process", key="process_button") and url and api_key:  # Check if URL and API key are provided before processing
            with st.spinner("Processing..."):
                web_text = get_web_page_text(url)
                if web_text:
                    text_chunks = get_text_chunks(web_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

if __name__ == "__main__":
    main()
