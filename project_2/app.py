import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import glob
import os

# Set page config
st.set_page_config(page_title="Everstorm Support Bot", page_icon="🛍️", layout="centered")

# System prompt template
SYSTEM_TEMPLATE = """
You are a **Customer Support Chatbot** for Everstorm Outfitters. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, respond with "I'm not sure from the docs."

Rules:
1) Use ONLY the provided <context> to answer.
2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.

CONTEXT:
{context}

USER:
{question}
"""

@st.cache_resource
def load_rag_chain():
    """Load documents, build vector store, and initialize RAG chain"""
    # Load PDF documents
    pdf_paths = glob.glob("data/Everstorm_*.pdf")
    raw_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        raw_docs.extend(loader.load())
    
    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = text_splitter.split_documents(raw_docs)
    
    # Create embeddings and vector store
    embedding_model = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")
    vectordb = FAISS.from_documents(chunks, embedding_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    
    # Initialize LLM
    llm = Ollama(model="gemma3:1b", temperature=0.1, base_url="http://host.docker.internal:11434")
    
    # Create prompt
    prompt = PromptTemplate(template=SYSTEM_TEMPLATE, input_variables=["context", "question"])
    
    # Build RAG chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return chain

# App header
st.title("🛍️ Everstorm Outfitters Support Bot")
st.markdown("Ask me anything about returns, shipping, or customer support!")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load RAG chain
try:
    chain = load_rag_chain()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_question := st.chat_input("How can I help you today?"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = chain.invoke({
                    "question": user_question,
                    "chat_history": st.session_state.chat_history
                })
                response = result["answer"]
                st.markdown(response)
        
        # Update chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append((user_question, response))
    
    # Sidebar with example questions
    with st.sidebar:
        st.header("💡 Example Questions")
        st.markdown("""
        - What is your refund policy?
        - How long does shipping take?
        - How can I track my order?
        - What are your customer support hours?
        - Do you offer international shipping?
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()

except Exception as e:
    st.error(f"⚠️ Error loading chatbot: {str(e)}")
    st.info("Make sure Ollama is running and the model is downloaded. Run `ollama serve` and `ollama pull gemma3:1b`")