# 🤖 Universal Document RAG Assistant

A Streamlit-based **Retrieval-Augmented Generation (RAG)** application for asking questions about PDF documents using **Cohere embeddings**, **FAISS**, and **Cohere Command**.

The application is designed to work with multiple documents and supports document-level metadata such as document type and category. You can upload PDFs, split them into chunks, create a FAISS vector index, retrieve the most relevant chunks, and generate answers grounded only in the retrieved context.

## ✨ Features

- 📄 Upload multiple PDF documents.
- 🔑 Enter your **Cohere API key directly in the Streamlit sidebar**.
- 🔐 API key field is hidden using a password input.
- 🧩 Configurable chunk size and chunk overlap.
- 📚 Limit the number of pages processed per PDF.
- 🏷️ Add metadata to every document:
  - Document Type
  - Category / Track
  - Source filename
  - PDF page number
- 🔎 Semantic similarity search with FAISS.
- 🎯 Configurable Top-K retrieval.
- 🧠 Configurable maximum context size.
- 🚦 Batched Cohere embedding requests.
- ⏱️ Configurable delay between embedding batches.
- 🔄 Basic retry handling for Cohere rate-limit errors.
- 💬 Chat interface with conversation history.
- 📖 Displays retrieved chunks and their source metadata.
- 🛡️ Prompt instructs the model to answer only from the supplied context.
- 🌱 Can use `.env` as an alternative to entering the API key in the UI.

## 🏗️ Architecture

```text
                     ┌─────────────────────┐
                     │     Streamlit UI    │
                     │      ui.py          │
                     └──────────┬──────────┘
                                │
                    Upload PDFs + metadata
                                │
                                ▼
                     ┌─────────────────────┐
                     │    PyPDFLoader      │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ RecursiveCharacter  │
                     │    Text Splitter    │
                     └──────────┬──────────┘
                                │
                         Document chunks
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Cohere embed-v4.0   │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │       FAISS         │
                     │    Vector Store     │
                     └──────────┬──────────┘
                                │
                         Similarity Search
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Retrieved Context   │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Cohere Command      │
                     │  command-a-03-2025  │
                     └──────────┬──────────┘
                                │
                                ▼
                         Grounded Answer
```

## 📁 Project Structure

```text
project/
│
├── ui.py
├── rag_componet.py
├── README.md
├── requirements.txt
└── .env                 # optional, do NOT commit this file
```

> The original module name in this project is `rag_componet.py`. Keep that filename consistent with the import in `ui.py`.

## 🔧 Technologies

| Technology | Purpose |
|---|---|
| Python | Application language |
| Streamlit | Web interface |
| LangChain | RAG components and model integration |
| PyPDFLoader | PDF text extraction |
| RecursiveCharacterTextSplitter | Document chunking |
| Cohere `embed-v4.0` | Text embeddings |
| Cohere `command-a-03-2025` | Answer generation |
| FAISS | Vector similarity search |
| python-dotenv | Optional `.env` configuration |


## 📸 Project Demo

<p align="center">
  <img src="t1.png" width="45%" />
  <img src="t2.png" width="45%" />
</p>

<p align="center">
  <img src="t3.png" width="45%" />
  <img src="t4.png" width="45%" />
</p>
## 🚀 Installation

### 1. Clone or download the project

Place the project files in the same directory:

```text
ui.py
rag_componet.py
README.md
requirements.txt
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🔑 Cohere API Key

There are two supported ways to provide your Cohere API key.

### Option A — Enter it in Streamlit

Run the application:

```bash
streamlit run ui.py
```

Then use:

```text
Sidebar
   ↓
🔑 Cohere API
   ↓
Cohere API Key
```

Paste your key into the password field.

The UI passes this key to the embedding and generation functions.

### Option B — Use `.env`

Create a `.env` file:

```env
cohere_key=YOUR_COHERE_API_KEY
```

The application also accepts:

```env
COHERE_API_KEY=YOUR_COHERE_API_KEY
```

Do not commit `.env` to Git.

Recommended `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
```

## ▶️ Run the Application

```bash
streamlit run ui.py
```

Streamlit will provide a local URL, normally similar to:

```text
http://localhost:8501
```

## 🧭 How to Use

### Step 1 — Enter the API key

Open the sidebar and enter your Cohere API key.

### Step 2 — Configure RAG settings

The sidebar provides controls for:

- **Chunk Size** — number of characters in each chunk.
- **Chunk Overlap** — shared characters between neighboring chunks.
- **Maximum Pages per PDF** — limits PDF processing.
- **Retrieved Chunks (Top K)** — number of chunks retrieved for a question.
- **Maximum Context Characters** — maximum retrieved text sent to the LLM.
- **Embedding Batch Size** — number of chunks sent in one embedding request.
- **Delay Between Embedding Batches** — delay used to reduce rate-limit problems.

### Step 3 — Upload PDFs

Upload one or more PDF files.

### Step 4 — Add document metadata

For every uploaded document, choose:

- Document Type
- Category / Track

This metadata is attached to every generated chunk.

### Step 5 — Build the knowledge base

Click:

```text
🚀 Build Knowledge Base
```

The application:

1. Saves the uploaded PDF temporarily.
2. Loads it with `PyPDFLoader`.
3. Limits the number of pages if configured.
4. Splits the document into chunks.
5. Adds metadata.
6. Sends chunks to Cohere in batches.
7. Creates a FAISS index.
8. Stores the index and chunks in Streamlit session state.

### Step 6 — Ask questions

Use the chat input:

```text
Ask something about your documents...
```

The application retrieves relevant chunks from FAISS and sends the resulting context to Cohere for answer generation.

## 🔎 Metadata Filtering

Before asking a question, you can filter retrieval by:

- Document Type
- Category / Track
- Document / Source

For example:

```text
Document Type = CV
Category = AI
Document = candidate.pdf
```

This makes the retrieval process more targeted when multiple PDFs are loaded.

## 🧠 RAG Pipeline

The core pipeline is:

```text
PDF
 ↓
PyPDFLoader
 ↓
Documents
 ↓
RecursiveCharacterTextSplitter
 ↓
Chunks + Metadata
 ↓
Cohere Embeddings
 ↓
FAISS
 ↓
Similarity Search
 ↓
Top-K Chunks
 ↓
Context Construction
 ↓
Cohere Chat Model
 ↓
Answer
```

## 📦 Chunking

The application uses:

```python
RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
```

The default values in the UI are:

```text
Chunk Size: 1000
Chunk Overlap: 100
```

These are starting points, not universal optimal values.

For example:

- Smaller chunks can provide more precise retrieval.
- Larger chunks provide more surrounding context.
- Some overlap helps preserve information split across chunk boundaries.

## 🚦 Cohere Rate Limits

Large PDFs can produce hundreds or thousands of chunks. Sending all chunks in one request can cause rate-limit problems.

The project therefore embeds chunks in batches:

```text
Batch 1 → Cohere
wait
Batch 2 → Cohere
wait
Batch 3 → Cohere
wait
...
```

You can control:

```text
Embedding Batch Size
Delay Between Embedding Batches
```

If you receive HTTP `429` or rate-limit errors, reduce the batch size and/or increase the delay.

The implementation also retries a rate-limited batch after waiting.

## 🛡️ Grounded Generation

The generation prompt instructs the model to:

- Use only the supplied document context.
- Avoid inventing information.
- Explicitly state when the answer is not available in the provided document.

This is an important RAG design choice because retrieval should constrain the generation step.

## ⚠️ Important Limitations

### 1. FAISS filtering

The current implementation uses LangChain's FAISS similarity search with metadata filters.

Metadata filtering in this setup is applied through LangChain after/beside the similarity-search workflow rather than being a native database pre-filter.

For very large production systems, a database/vector database with native metadata filtering may be preferable.

### 2. PDFs with scanned images

`PyPDFLoader` works best when the PDF contains selectable text.

Scanned PDFs may require an OCR pipeline before retrieval.

### 3. In-memory session state

The FAISS index is stored in Streamlit session state.

Therefore, the knowledge base is not a permanent database.

Restarting the Streamlit process can require rebuilding the index.

### 4. API usage

Embedding many chunks and generating many answers consumes Cohere API quota.

Monitor your Cohere account usage and rate limits.

## 🔒 Security

The Streamlit API-key field uses:

```python
type="password"
```

The application does not intentionally write the key to a file when it is entered through the UI.

However:

- Do not print the API key.
- Do not commit `.env`.
- Do not put API keys directly in source code.
- Do not share screenshots containing the key.
- For production deployments, prefer Streamlit secrets or a secure secrets manager.

## 🧪 Suggested Experiments

This project is also useful for learning and evaluating RAG.

You can experiment with:

### Chunk size

```text
500
800
1000
1500
2000
```

### Chunk overlap

```text
50
100
150
200
```

### Top-K

```text
3
5
8
10
```

### Context size

Try different context limits and observe how answer quality changes.

## 🛠️ Troubleshooting

### `Cohere API key is required`

Enter the key in the sidebar or create a `.env` file containing:

```env
cohere_key=YOUR_COHERE_API_KEY
```

### HTTP 429

This usually indicates a rate-limit issue.

Try:

- Decreasing embedding batch size.
- Increasing the delay between batches.
- Processing fewer documents at a time.

### No text found

The PDF may be:

- Scanned.
- Image-only.
- Encrypted.
- Structured in a way that the PDF loader cannot extract correctly.

Consider adding OCR for scanned documents.

### Poor answers

Try adjusting:

- Chunk size.
- Chunk overlap.
- Top-K.
- Maximum context size.
- Metadata filters.

Also inspect the **Retrieved Context** section. If the correct information is not retrieved, the problem is primarily retrieval rather than generation.

## 📈 Future Improvements

Possible production improvements include:

- Persistent FAISS indexes.
- PostgreSQL + pgvector.
- Hybrid keyword + vector search.
- Reranking retrieved chunks.
- Query rewriting.
- Multi-query retrieval.
- Better OCR support.
- Document deduplication.
- Persistent document storage.
- User authentication.
- Streaming generated answers.
- RAG evaluation dashboard.
- Precision@K, Recall@K, MAP, and MRR evaluation.
- Citation-aware answers.
- Conversation-aware retrieval.
- Background document processing.
- Better handling of Cohere rate limits and retries.
- Caching embeddings.

## 📄 License

See the `LICENSE` file included with the project.

## 👨‍💻 Project Purpose

This project demonstrates a practical end-to-end RAG system:

```text
Document ingestion
        ↓
Chunking
        ↓
Metadata
        ↓
Embeddings
        ↓
Vector database
        ↓
Retrieval
        ↓
Context construction
        ↓
LLM generation
        ↓
Grounded answer
```

It is suitable as a learning project for understanding how modern document-question-answering systems are built with LangChain, Cohere, FAISS, and Streamlit.

👨‍💻 Author

Made by: Fathy Abderabbo
Email: fathypepo9@gmail.com
