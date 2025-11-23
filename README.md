# Bogey Chatbot

A simple Streamlit chatbot that uses Pinecone for retrieval and OpenAI for generation. The app embeds user queries, retrieves relevant passages from a Pinecone index, and responds in Mongolian based solely on the retrieved context.

## Prerequisites

- Python 3.11+
- API keys for OpenAI and Pinecone
- An existing Pinecone index populated with documents (metadata must include a `text` field)

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Export environment variables:

   ```bash
   export OPENAI_API_KEY="<your-openai-key>"
   export PINECONE_API_KEY="<your-pinecone-key>"
   export PINECONE_HOST="<your-pinecone-host>"
   # Optional: override the default index name
   # export PINECONE_INDEX="rag-chat-demo"
   ```

## Run

Start the Streamlit app:

```bash
streamlit run app.py
```

The UI will open in your browser. Type a question in Mongolian; the chatbot will retrieve relevant passages and answer using only that context. If the information is missing, it will note that the knowledge base does not contain the requested data.
