import os
import time

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate


load_dotenv()


# ============================================================
# COHERE CONFIGURATION
# ============================================================

def get_cohere_api_key(api_key=None):
    """
    Resolve the Cohere API key.

    Priority:
    1. API key supplied by the Streamlit UI/function call.
    2. cohere_key from the .env file.
    3. COHERE_API_KEY from the environment.
    """
    key = api_key or os.getenv("cohere_key") or os.getenv("COHERE_API_KEY")

    if not key:
        raise ValueError(
            "Cohere API key is required. Enter it in the Streamlit sidebar "
            "or add cohere_key=YOUR_KEY to your .env file."
        )

    return key.strip()


def create_embeddings(api_key=None):
    """Create the Cohere embedding model for the current API key."""
    return CohereEmbeddings(
        model="embed-v4.0",
        cohere_api_key=get_cohere_api_key(api_key)
    )


def create_llm(api_key=None):
    """Create the Cohere chat model for the current API key."""
    return ChatCohere(
        model="command-a-03-2025",
        temperature=0,
        cohere_api_key=get_cohere_api_key(api_key)
    )


# ============================================================
# PROMPT
# ============================================================

prompt = PromptTemplate(
    template="""
You are a helpful document question-answering assistant.

Use ONLY the provided document context to answer the question.

Context:
{context}

Question:
{question}

Instructions:
- Answer only using the provided context.
- Do not invent information.
- If the answer is not present in the context, say:
  "The information is not available in the provided document."
""",
    input_variables=["context", "question"]
)


# ============================================================
# LOAD + SPLIT PDF
# ============================================================

def retrieval(
    file_path,
    filename,
    document_type,
    category,
    chunk_size=1000,
    chunk_overlap=100,
    max_pages=None
):
    loader = PyPDFLoader(file_path)

    documents = loader.load()

    # Limit pages if requested
    if max_pages is not None:
        documents = documents[:max_pages]

    if not documents:
        return []

    # Prevent invalid configuration
    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    # Add generic metadata
    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["document_type"] = document_type
        chunk.metadata["category"] = category

    return chunks


# ============================================================
# CREATE FAISS WITH BATCHED EMBEDDINGS
# ============================================================

def create_vector_db(
    chunks,
    batch_size=20,
    delay=1.0,
    progress_callback=None,
    cohere_api_key=None
):
    """
    Create a FAISS database without sending all document
    chunks to Cohere in one huge embedding request.

    batch_size:
        Number of chunks embedded per API request.

    delay:
        Delay between batches. Increase this if you hit
        Cohere rate limits.
    """

    if not chunks:
        raise ValueError(
            "No chunks were provided."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be >= 1."
        )

    vector_db = None
    embeddings = create_embeddings(cohere_api_key)

    total = len(chunks)

    for start in range(0, total, batch_size):

        end = min(
            start + batch_size,
            total
        )

        batch = chunks[start:end]

        print(
            f"Embedding chunks "
            f"{start + 1}-{end} / {total}"
        )

        try:

            if vector_db is None:

                vector_db = FAISS.from_documents(
                    batch,
                    embeddings
                )

            else:

                vector_db.add_documents(
                    batch
                )

        except Exception as e:

            error_text = str(e).lower()

            if "429" in error_text or "rate limit" in error_text:

                # Give Cohere some time before retrying
                print(
                    "Cohere rate limit reached. "
                    "Waiting before retry..."
                )

                time.sleep(
                    max(delay, 10)
                )

                # Retry this same batch
                if vector_db is None:

                    vector_db = FAISS.from_documents(
                        batch,
                        embeddings
                    )

                else:

                    vector_db.add_documents(
                        batch
                    )

            else:
                raise

        # Progress callback for Streamlit
        if progress_callback:

            progress_callback(
                end / total
            )

        # Delay between requests
        if end < total:

            time.sleep(delay)

    return vector_db


# ============================================================
# ADD NEW DOCUMENTS TO EXISTING FAISS
# ============================================================

def add_chunks_to_vector_db(
    vector_db,
    chunks,
    batch_size=20,
    delay=1.0,
    progress_callback=None,
    cohere_api_key=None
):
    """
    Add new chunks to an existing FAISS database.
    """

    if not chunks:
        return vector_db

    # The existing FAISS object already contains its embedding function,
    # so the API key is only resolved here to validate that it is available.
    get_cohere_api_key(cohere_api_key)

    total = len(chunks)

    for start in range(0, total, batch_size):

        end = min(
            start + batch_size,
            total
        )

        batch = chunks[start:end]

        try:

            vector_db.add_documents(
                batch
            )

        except Exception as e:

            error_text = str(e).lower()

            if "429" in error_text or "rate limit" in error_text:

                time.sleep(
                    max(delay, 10)
                )

                vector_db.add_documents(
                    batch
                )

            else:
                raise

        if progress_callback:

            progress_callback(
                end / total
            )

        if end < total:

            time.sleep(delay)

    return vector_db


# ============================================================
# RETRIEVAL / AUGMENTATION
# ============================================================

def augmented(
    question,
    vector_db,
    k=5,
    document_type=None,
    category=None,
    source=None,
    max_context_chars=12000
):
    """
    Retrieve relevant chunks using FAISS.

    IMPORTANT:
    FAISS metadata filtering is applied by LangChain after
    similarity search in the standard FAISS implementation.
    Therefore, this is not the same as a native database
    pre-filter before vector search.
    """

    # --------------------------------------------------------
    # Build metadata filter
    # --------------------------------------------------------

    filters = {}

    if document_type is not None:
        filters["document_type"] = document_type

    if category is not None:
        filters["category"] = category

    if source is not None:
        filters["source"] = source

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    if filters:

        retrieved_chunks = vector_db.similarity_search(
            question,
            k=k,
            filter=filters
        )

    else:

        retrieved_chunks = vector_db.similarity_search(
            question,
            k=k
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []
    current_length = 0

    for chunk in retrieved_chunks:

        source_name = chunk.metadata.get(
            "source",
            "Unknown"
        )

        document_type_name = chunk.metadata.get(
            "document_type",
            "Unknown"
        )

        category_name = chunk.metadata.get(
            "category",
            "Unknown"
        )

        page = chunk.metadata.get(
            "page",
            None
        )

        if page is not None:

            metadata_text = (
                f"Source: {source_name}\n"
                f"Type: {document_type_name}\n"
                f"Category: {category_name}\n"
                f"Page: {page + 1}"
            )

        else:

            metadata_text = (
                f"Source: {source_name}\n"
                f"Type: {document_type_name}\n"
                f"Category: {category_name}"
            )

        chunk_text = (
            f"{metadata_text}\n"
            f"{chunk.page_content}"
        )

        # Respect context size
        if (
            current_length + len(chunk_text)
            > max_context_chars
        ):
            break

        context_parts.append(
            chunk_text
        )

        current_length += len(chunk_text)

    context = "\n\n---\n\n".join(
        context_parts
    )

    return context, retrieved_chunks


# ============================================================
# GENERATION
# ============================================================

def generative(
    context,
    question,
    cohere_api_key=None
):
    final_prompt = prompt.format(
        context=context,
        question=question
    )

    llm = create_llm(cohere_api_key)

    response = llm.invoke(
        final_prompt
    )

    return response.content
