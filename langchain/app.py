import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import tempfile

# Load environment variables
#load_dotenv()
#groq_api_key = os.getenv("GROQ_API_KEY")
#hugging_face_api_key = os.getenv("hugging_face_api_key")
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#langchain_api_key = os.getenv("langchain_api_key")

groq_api_key = st.secrets.secrets.GROQ_API_KEY
hugging_face_api_key = st.secrets.secrets.hugging_face_api_key
langchain_api_key = st.secrets.secrets.langchain_api_key

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

st.title("Q&A RAG SYSTEM")
st.sidebar.title("Upload PDF")

uploaded_file = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
        
    try:
        with st.spinner("Processing PDF......"):
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                add_start_index=True
            )
            all_splits = text_splitter.split_documents(documents)
            
            # Create embeddings and vectorstore
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            vector_store = FAISS.from_documents(all_splits, embeddings)
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Initialize LLM
            llm = ChatGroq(
                model="llama3-8b-8192",
                api_key=groq_api_key
            )
            
            # Create prompt template
            prompt = PromptTemplate(
                template="""Use the following pieces of context to answer the question. 
                If the answer is not provided in the context, say 'I cannot find this information in the document.'
                
                Context: {context}
                Question: {question}
                
                Answer: """,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            st.success("PDF processed successfully")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        os.remove(tmp_path)

# Question input and answer display
if st.session_state.qa_chain:
    query = st.text_input("Ask a question about your PDF:")
    
    if query:
        with st.spinner('Finding answer...'):
            result = st.session_state.qa_chain.invoke({"query": query})
            
            st.write("### Answer:")
            st.write(result['result'])
            
            st.write("### Sources:")
            for doc in result['source_documents']:
                st.write(doc.page_content)
                st.write("---")

# Clear button
if st.button('Clear'):
    st.session_state.qa_chain = None
    st.rerun()
