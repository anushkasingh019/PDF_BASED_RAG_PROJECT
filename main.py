
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = PROJECT_DIR / "document_loader"

COLLECTION_NAME = "rag_collection"
EMBEDDING_DIMENSION = 384

# Load environment variables
load_dotenv(DEFAULT_PDF_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env")


# ==========================================
# 2. PROMPT TEMPLATE
# ==========================================

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful tutor. Answer using only the "
            "provided context. If the answer is not in the "
            "context, say you don't know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


# ==========================================
# 3. PDF PATH RESOLUTION
# ==========================================

def resolve_pdf_path(
    pdf_path: str | Path | None = None
) -> Path:

    configured_path = pdf_path or os.getenv("RAG_PDF_PATH")

    if configured_path:
        path = Path(configured_path).expanduser()

        if not path.is_absolute():
            path = PROJECT_DIR / path

    else:
        available_pdfs = sorted(
            DEFAULT_PDF_DIR.glob("*.pdf")
        )

        if not available_pdfs:
            raise FileNotFoundError(
                f"No PDF files found in {DEFAULT_PDF_DIR}"
            )

        if len(available_pdfs) > 1:
            names = ", ".join(
                pdf.name for pdf in available_pdfs
            )

            raise ValueError(
                "Multiple PDFs found. Set RAG_PDF_PATH "
                f"to choose one. Available PDFs: {names}"
            )

        path = available_pdfs[0]

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"PDF file does not exist: {path}"
        )

    return path


# ==========================================
# 4. LOAD AND SPLIT PDF
# ==========================================

def load_document_chunks(
    pdf_path: str | Path | None = None
) -> list[Any]:

    path = resolve_pdf_path(pdf_path)

    documents = PyPDFLoader(str(path)).load()

    if not documents:
        raise ValueError(
            f"No pages could be read from PDF: {path}"
        )

    if not any(
        document.page_content.strip()
        for document in documents
    ):
        raise ValueError(
            f"PDF contains no extractable text: {path}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            f"No text chunks could be created from PDF: {path}"
        )

    return chunks


# ==========================================
# 5. CREATE QDRANT RETRIEVER
# ==========================================

def build_retriever(
    pdf_path: str | Path | None = None,
    embeddings: Any | None = None,
) -> Any:

    chunks = load_document_chunks(pdf_path)

    embedding_model = (
        embeddings or FastEmbedEmbeddings()
    )

    client = QdrantClient(location=":memory:")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=Distance.COSINE,
        ),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )

    vectorstore.add_documents(chunks)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever


# ==========================================
# 6. NVIDIA AI MODEL
# ==========================================

def create_llm() -> Any:

    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is missing. "
            "Set it in your .env file."
        )

    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

    except ImportError as exc:
        raise RuntimeError(
            "NVIDIA integration is not installed. "
            "Run: python -m pip install "
            "langchain-nvidia-ai-endpoints"
        ) from exc

    return ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_api_key=api_key,
        temperature=0.2,
        max_completion_tokens=2500,
    )


# ==========================================
# 7. RAG QUESTION ANSWERING
# ==========================================

def query_rag(
    question: str,
    retriever: Any | None = None,
    llm: Any | None = None,
) -> str:

    if not question or not question.strip():
        raise ValueError(
            "Question must not be empty."
        )

    active_retriever = (
        retriever or build_retriever()
    )

    active_llm = llm or create_llm()

    retrieved_docs = active_retriever.invoke(question)

    if not retrieved_docs:
        return (
            "I don't know based on the provided document."
        )

    print(
        f"\nRetrieved {len(retrieved_docs)} "
        "relevant context chunk(s):"
    )

    for index, document in enumerate(
        retrieved_docs, start=1
    ):

        page_num = document.metadata.get("page")

        page_label = (
            page_num + 1
            if isinstance(page_num, int)
            else "Unknown"
        )

        preview = (
            document.page_content
            .strip()
            .replace("\n", " ")[:150]
        )

        print(
            f" [{index}] (Page {page_label}): "
            f"{preview}..."
        )

    context = "\n\n".join(
        document.page_content
        for document in retrieved_docs
        if document.page_content.strip()
    )

    if not context:
        return (
            "I don't know based on the provided document."
        )

    answer = (
        PROMPT | active_llm | StrOutputParser()
    ).invoke(
        {
            "context": context,
            "question": question,
        }
    )

    print("\nAnswer:")
    print(answer)

    return answer


# ==========================================
# 8. CACHE RESOURCES
# ==========================================

@st.cache_resource(show_spinner=False)
def get_cached_retriever():

    return build_retriever()


@st.cache_resource(show_spinner=False)
def get_cached_llm():

    return create_llm()


# ==========================================
# 9. STREAMLIT USER INTERFACE
# ==========================================

def run_app():

    st.set_page_config(
        page_title="PDF RAG Assistant",
        page_icon="📚",
        layout="centered",
    )

    st.title("📚 PDF Question Answering Assistant")

    st.write(
        "Upload your knowledge through a PDF and "
        "ask questions. The RAG system retrieves "
        "relevant information and generates answers "
        "using NVIDIA AI."
    )

    st.divider()

    # Check PDF
    try:
        pdf_path = resolve_pdf_path()

        st.success(
            f"Document detected: {pdf_path.name}"
        )

    except Exception as exc:
        st.error(f"PDF loading problem: {exc}")
        return

    # Application information
    with st.expander("About this application"):

        st.write(
            """
            This application uses:

            - PyPDFLoader for PDF extraction
            - RecursiveCharacterTextSplitter for chunking
            - FastEmbed for text embeddings
            - Qdrant for vector storage and retrieval
            - NVIDIA AI for answer generation
            - Streamlit for the user interface
            """
        )

    st.divider()

    # Question input
    st.subheader("Ask your question")

    question = st.text_area(
        "Enter your question below:",
        placeholder=(
            "Example: What is the Java Virtual Machine "
            "and what does it do?"
        ),
        height=110,
    )

    # Generate answer
    if st.button(
        "Get Answer",
        type="primary",
        use_container_width=True,
    ):

        if not question.strip():
            st.warning(
                "Please enter a question first."
            )
            return

        try:
            with st.spinner(
                "Processing your PDF and preparing search..."
            ):
                retriever = get_cached_retriever()

            with st.spinner(
                "Generating your answer using NVIDIA AI..."
            ):
                llm = get_cached_llm()

                answer = query_rag(
                    question=question,
                    retriever=retriever,
                    llm=llm,
                )

            st.divider()

            st.subheader("Answer")

            st.write(answer)

            st.success(
                "Your answer has been generated successfully!"
            )

        except Exception as exc:
            st.error(
                f"Something went wrong: {exc}"
            )

            with st.expander("View technical details"):
                st.exception(exc)


# ==========================================
# 10. APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":
    run_app()
