# PDF RAG Q&A System

A Question-Answering system built with LangChain that uses Retrieval Augmented Generation (RAG) to provide accurate answers from PDF documents.

## Features

- PDF document processing and analysis
- Intelligent question answering using RAG architecture
- Real-time document chunking and embedding
- Source attribution for answers
- Clean Streamlit interface
- Memory-efficient document handling

## Tech Stack

- Python 3.8+
- LangChain
- Streamlit
- FAISS for vector storage
- HuggingFace Embeddings
- Groq for language model processing

## Installation

1. Clone the repository
```bash
git clone https://github.com/YourUsername/pdf-rag-qa-system.git
cd pdf-rag-qa-system
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
hugging_face_api_key=your_huggingface_api_key_here
langchain_api_key=your_langchain_api_key_here
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Upload a PDF document using the sidebar
3. Wait for the document to be processed
4. Ask questions about your document in the text input
5. View answers and their source context

## System Architecture

The system works in the following steps:
1. PDF Upload & Processing
2. Text Chunking
3. Embedding Generation
4. Vector Storage
5. Question Processing
6. Context Retrieval
7. Answer Generation


## Limitations

- Currently only supports PDF files
- Maximum file size: 200MB
- Requires active internet connection for API calls
- Processing time depends on document size

## Future Improvements

- Support for multiple document formats
- Image extraction and analysis
- Conversation memory
- Answer confidence scores
- Multi-document querying

## Contributing

Feel free to open issues and pull requests for any improvements you can add.

## License

MIT License - see LICENSE file for details
