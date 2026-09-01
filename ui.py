import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag_componet import (
    retrieval,
    create_vector_db,
    augmented,
    generative
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Universal RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "files" not in st.session_state:
    st.session_state.files = []

if "document_config" not in st.session_state:
    st.session_state.document_config = {}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 Universal RAG")

    st.caption(
        "Chat With Your Documnets"
    )

    st.divider()

    # ========================================================
    # COHERE API KEY
    # ========================================================

    st.subheader("🔑 Cohere API")

    env_api_key = (
        os.getenv("cohere_key")
        or os.getenv("COHERE_API_KEY")
        or ""
    )

    cohere_api_key = st.text_input(
        "Cohere API Key",
        value=env_api_key,
        type="password",
        placeholder="Enter your Cohere API key",
        help=(
            "Your key is used for embeddings and answer generation. "
            "It is kept in Streamlit session state and is not written to disk."
        )
    ).strip()

    if cohere_api_key:
        st.success("Cohere API key is configured.")
    else:
        st.warning("Enter a Cohere API key before building the knowledge base.")

    st.divider()

    # ========================================================
    # RAG SETTINGS
    # ========================================================

    st.subheader("⚙️ RAG Settings")

    chunk_size = st.slider(
        "Chunk Size",
        min_value=300,
        max_value=3000,
        value=1000,
        step=100,
        help=(
            "Approximate number of characters "
            "in each chunk."
        )
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=100,
        step=50,
        help=(
            "Number of characters shared between "
            "neighboring chunks."
        )
    )

    max_pages = st.number_input(
        "Maximum Pages per PDF",
        min_value=1,
        max_value=10000,
        value=500,
        step=50,
        help=(
            "Only this many pages from each PDF "
            "will be processed."
        )
    )

    top_k = st.slider(
        "Retrieved Chunks (Top K)",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    max_context_chars = st.number_input(
        "Maximum Context Characters",
        min_value=2000,
        max_value=100000,
        value=12000,
        step=1000,
        help=(
            "Maximum retrieved text sent to the LLM."
        )
    )

    embedding_batch_size = st.slider(
        "Embedding Batch Size",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help=(
            "Smaller batches reduce the chance of "
            "hitting Cohere's token-rate limit."
        )
    )

    embedding_delay = st.number_input(
        "Delay Between Embedding Batches (seconds)",
        min_value=0.0,
        max_value=30.0,
        value=2.0,
        step=0.5,
        help=(
            "Increase this if Cohere returns HTTP 429."
        )
    )

    st.divider()

    # ========================================================
    # FILE UPLOADER
    # ========================================================

    st.subheader("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    # ========================================================
    # DOCUMENT CONFIGURATION
    # ========================================================

    if uploaded_files:

        st.subheader("🏷️ Document Information")

        document_types = [
            "CV",
            "Book",
            "Paper",
            "Notes",
            "Article",
            "Report",
            "Document",
            "Other"
        ]

        categories = [
            "AI",
            "Backend",
            "Frontend",
            "Mobile",
            "Data Science",
            "Machine Learning",
            "Deep Learning",
            "NLP",
            "Computer Vision",
            "Programming",
            "Software Engineering",
            "Databases",
            "Other"
        ]

        document_config = {}

        for uploaded_file in uploaded_files:

            filename = uploaded_file.name

            with st.expander(
                f"📄 {filename}",
                expanded=False
            ):

                document_type = st.selectbox(
                    "Document Type",
                    document_types,
                    key=f"type_{filename}"
                )

                category = st.selectbox(
                    "Category / Track",
                    categories,
                    key=f"category_{filename}"
                )

                document_config[filename] = {
                    "document_type": document_type,
                    "category": category
                }

        st.divider()

        # ====================================================
        # BUILD KNOWLEDGE BASE
        # ====================================================

        if st.button(
            "🚀 Build Knowledge Base",
            use_container_width=True
        ):

            if not cohere_api_key:
                st.error(
                    "❌ Please enter your Cohere API key in the sidebar first."
                )
                st.stop()

            all_chunks = []

            progress_bar = st.progress(0)

            status_text = st.empty()

            try:

                total_files = len(
                    uploaded_files
                )

                # ============================================
                # PROCESS FILES
                # ============================================

                for file_index, uploaded_file in enumerate(
                    uploaded_files
                ):

                    filename = uploaded_file.name

                    config = document_config[
                        filename
                    ]

                    status_text.markdown(
                        f"### 📖 Processing "
                        f"`{filename}` "
                        f"({file_index + 1}/{total_files})"
                    )

                    # ----------------------------------------
                    # Save temporary PDF
                    # ----------------------------------------

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as tmp:

                        tmp.write(
                            uploaded_file.getvalue()
                        )

                        file_path = tmp.name

                    try:

                        # ------------------------------------
                        # Load + split + metadata
                        # ------------------------------------

                        chunks = retrieval(
                            file_path=file_path,
                            filename=filename,
                            document_type=config[
                                "document_type"
                            ],
                            category=config[
                                "category"
                            ],
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            max_pages=max_pages
                        )

                        if not chunks:
                            st.warning(
                                f"No text found in "
                                f"{filename}."
                            )

                        all_chunks.extend(
                            chunks
                        )

                    finally:

                        if os.path.exists(file_path):
                            os.remove(file_path)

                    # File-level progress
                    file_progress = (
                        (file_index + 1)
                        / total_files
                    )

                    progress_bar.progress(
                        min(
                            file_progress * 0.5,
                            0.5
                        )
                    )

                # ============================================
                # CHECK
                # ============================================

                if not all_chunks:

                    st.error(
                        "No text chunks were created."
                    )

                    st.stop()

                # ============================================
                # CREATE FAISS
                # ============================================

                status_text.markdown(
                    "### 🔎 Creating FAISS index..."
                )

                def update_embedding_progress(value):

                    progress_bar.progress(
                        0.5 + (value * 0.5)
                    )

                vector_db = create_vector_db(
                    chunks=all_chunks,
                    batch_size=embedding_batch_size,
                    delay=embedding_delay,
                    progress_callback=(
                        update_embedding_progress
                    ),
                    cohere_api_key=cohere_api_key
                )

                # ============================================
                # SAVE STATE
                # ============================================

                st.session_state.vector_db = (
                    vector_db
                )

                st.session_state.chunks = (
                    all_chunks
                )

                st.session_state.files = [
                    file.name
                    for file in uploaded_files
                ]

                st.session_state.document_config = (
                    document_config.copy()
                )

                st.session_state.messages = []

                progress_bar.progress(1.0)

                status_text.empty()

                st.success(
                    "✅ Knowledge base created!"
                )

                st.info(
                    f"Processed "
                    f"**{len(uploaded_files)}** documents "
                    f"and created "
                    f"**{len(all_chunks)}** chunks."
                )

            except Exception as e:

                st.error(
                    f"❌ Error while processing documents:\n\n"
                    f"{e}"
                )


    # ========================================================
    # KNOWLEDGE BASE STATUS
    # ========================================================

    st.divider()

    st.subheader("📊 Knowledge Base")

    if st.session_state.vector_db is not None:

        st.success(
            "🟢 Knowledge Base Ready"
        )

        st.metric(
            "Documents",
            len(st.session_state.files)
        )

        st.metric(
            "Chunks",
            len(st.session_state.chunks)
        )

    else:

        st.info(
            "Upload documents and build "
            "the knowledge base."
        )


# ============================================================
# MAIN PAGE
# ============================================================

st.title(
    "🤖 Universal Document RAG Assistant"
)

st.caption(
    "Ask questions about CVs, books, papers, "
    "notes, and other PDF documents."
)


# ============================================================
# SEARCH FILTERS
# ============================================================

if st.session_state.vector_db is not None:

    st.divider()

    st.subheader("🔎 Search Filters")

    col1, col2, col3 = st.columns(3)

    # ========================================================
    # DOCUMENT TYPE
    # ========================================================

    with col1:

        available_types = [
            "All"
        ] + sorted(
            set(
                config["document_type"]
                for config
                in st.session_state.document_config.values()
            )
        )

        selected_type = st.selectbox(
            "📚 Document Type",
            available_types
        )

    # ========================================================
    # CATEGORY
    # ========================================================

    with col2:

        available_categories = [
            "All"
        ] + sorted(
            set(
                config["category"]
                for config
                in st.session_state.document_config.values()
            )
        )

        selected_category = st.selectbox(
            "🏷️ Category / Track",
            available_categories
        )

    # ========================================================
    # SOURCE
    # ========================================================

    with col3:

        selected_source = st.selectbox(
            "📄 Document",
            ["All"] + st.session_state.files
        )

    active_filters = []

    if selected_type != "All":
        active_filters.append(
            f"Type = **{selected_type}**"
        )

    if selected_category != "All":
        active_filters.append(
            f"Category = **{selected_category}**"
        )

    if selected_source != "All":
        active_filters.append(
            f"Document = **{selected_source}**"
        )

    if active_filters:

        st.info(
            "🔎 Active filters: "
            + " | ".join(active_filters)
        )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your documents..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    if not cohere_api_key:
        st.warning(
            "⚠️ Please enter your Cohere API key in the sidebar first."
        )
        st.stop()

    if st.session_state.vector_db is None:

        st.warning(
            "⚠️ Please upload documents and "
            "build the knowledge base first."
        )

        st.stop()

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # --------------------------------------------------------
    # Assistant
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            type_filter = (
                None
                if selected_type == "All"
                else selected_type
            )

            category_filter = (
                None
                if selected_category == "All"
                else selected_category
            )

            source_filter = (
                None
                if selected_source == "All"
                else selected_source
            )

            # ================================================
            # RETRIEVAL
            # ================================================

            with st.spinner(
                "🔍 Searching FAISS..."
            ):

                context, retrieved_chunks = augmented(
                    question=question,
                    vector_db=(
                        st.session_state.vector_db
                    ),
                    k=top_k,
                    document_type=type_filter,
                    category=category_filter,
                    source=source_filter,
                    max_context_chars=max_context_chars
                )

            # ================================================
            # GENERATION
            # ================================================

            if not retrieved_chunks:

                answer = (
                    "I couldn't find relevant information "
                    "in the selected documents."
                )

            else:

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    answer = generative(
                        context=context,
                        question=question,
                        cohere_api_key=cohere_api_key
                    )

            # ================================================
            # DISPLAY ANSWER
            # ================================================

            st.markdown(answer)

            # ================================================
            # RETRIEVED CHUNKS
            # ================================================

            with st.expander(
                "📚 Retrieved Context"
            ):

                if not retrieved_chunks:

                    st.write(
                        "No chunks retrieved."
                    )

                for i, chunk in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    st.markdown(
                        f"### Chunk {i}"
                    )

                    source = chunk.metadata.get(
                        "source",
                        "Unknown"
                    )

                    document_type = (
                        chunk.metadata.get(
                            "document_type",
                            "Unknown"
                        )
                    )

                    category = (
                        chunk.metadata.get(
                            "category",
                            "Unknown"
                        )
                    )

                    page = chunk.metadata.get(
                        "page",
                        None
                    )

                    if page is not None:

                        st.caption(
                            f"📄 {source} | "
                            f"📚 {document_type} | "
                            f"🏷️ {category} | "
                            f"📖 Page {page + 1}"
                        )

                    else:

                        st.caption(
                            f"📄 {source} | "
                            f"📚 {document_type} | "
                            f"🏷️ {category}"
                        )

                    st.write(
                        chunk.page_content
                    )

                    st.divider()

            # ================================================
            # SAVE RESPONSE
            # ================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            st.error(
                f"❌ Error generating answer:\n\n"
                f"{e}"
            )
