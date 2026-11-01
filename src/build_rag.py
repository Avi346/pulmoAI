"""
build_rag.py — RAG Pipeline for Clinical Guidelines
Parses medical PDFs, embeds with HuggingFace, stores in ChromaDB,
and queries using Mistral AI API for report generation.
"""

import os
import sys
import argparse
import threading
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
PDF_DIR = os.path.join(BASE_DIR, "data", "guidelines")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# ── Config ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral-small-latest"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

# ── Prompt Template ────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are an empathetic, highly knowledgeable clinical AI assistant and healthcare consultant.
Answer the user's query in a natural, helpful, human tone based on the medical guidelines and conversation context below.

Critical Guidelines for your tone and output:
1. ADAPTIVE LENGTH: For small or simple factual questions (e.g., definitions, follow-ups, or quick checks), keep your answer brief, direct, and concise (2-4 sentences max). Do NOT generate lengthy reports for simple questions.
2. HOLISTIC PATIENT CARE: If a person asks what to do if they have pneumonia (or general recovery guidance), provide clear, actionable bullet points covering immediate medical steps, diet, rest, and red flags.
3. CONVERSATIONAL MEMORY & TONE:
   - You are provided with the patient's X-Ray Screening Status and Recent Chat History.
   - CRITICAL RULE ON SCREENING RESULTS: Do NOT repeatedly announce or restate the patient context of whether they have pneumonia or not at the start of every single message! Only mention their diagnosis explicitly if they ask "what were my results?" or on their very first question after screening. Otherwise, use their screening diagnosis silently as background medical awareness to tailor your clinical advice.
   - Maintain continuity with Recent Chat History so you naturally understand follow-up questions and pronouns (e.g., "what about dosage?", "can I eat that?").

Patient X-Ray Screening Status:
{patient_context}

Recent Chat History:
{chat_history}

Clinical Guidelines Evidence:
{context}

User Query: {question}

Helpful Response:"""

# ── Build Vector Store ─────────────────────────────────────────────────
def build_vectorstore():
    """Load PDFs, chunk text, embed, and store in ChromaDB."""

    # Check if PDFs exist
    if not os.path.exists(PDF_DIR):
        print(f"ERROR: PDF directory not found: {PDF_DIR}")
        print("Please create it and add medical guideline PDFs.")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"ERROR: No PDF files found in {PDF_DIR}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

    # Load all PDFs
    all_documents = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)
        print(f"  Loaded {pdf_file}: {len(documents)} pages")

    print(f"Total pages loaded: {len(all_documents)}")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Split into {len(chunks)} chunks")

    # Create embeddings and store
    print(f"Embedding with {EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )

    print(f"Vector store saved to {VECTORSTORE_DIR}")
    print("Done! You can now query with: python src/build_rag.py --query \"your question\"")
    return vectorstore

# ── Query RAG Pipeline ─────────────────────────────────────────────────
_cached_chain = None
_chain_lock = threading.Lock()

def get_rag_chain():
    """Build and return the RAG chain (cached singleton)."""
    global _cached_chain
    with _chain_lock:
        if _cached_chain is None:
            if not os.path.exists(VECTORSTORE_DIR):
                print("ERROR: Vector store not found. Run with --build first.")
                sys.exit(1)

            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            # Warm up embedding computation so initial query is instant
            try:
                embeddings.embed_query("warmup")
            except Exception:
                pass

            vectorstore = Chroma(
                persist_directory=VECTORSTORE_DIR,
                embedding_function=embeddings,
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
            llm = ChatMistralAI(model=LLM_MODEL, api_key=MISTRAL_API_KEY)
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            def get_question(input_val):
                if isinstance(input_val, dict):
                    return input_val.get("question", "")
                return str(input_val)

            def get_patient_context(input_val):
                if isinstance(input_val, dict):
                    return input_val.get("patient_context", "No X-ray screened yet.")
                return "No X-ray screened yet."

            def get_chat_history(input_val):
                if isinstance(input_val, dict):
                    return input_val.get("chat_history", "None")
                return "None"

            _cached_chain = (
                {
                    "context": get_question | retriever | format_docs,
                    "question": get_question,
                    "patient_context": get_patient_context,
                    "chat_history": get_chat_history,
                }
                | prompt
                | llm
                | StrOutputParser()
            )
    return _cached_chain

def query_rag(question):
    """Retrieve relevant chunks and generate a report using the LLM."""
    chain = get_rag_chain()
    print(f"Querying: {question}")
    print(f"Using LLM: {LLM_MODEL}")
    print("-" * 50)

    response = chain.invoke(question)
    return response

# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline for Clinical Guidelines")
    parser.add_argument("--build", action="store_true", help="Build the vector store from PDFs")
    parser.add_argument("--query", type=str, help="Query the RAG pipeline")
    args = parser.parse_args()

    if args.build:
        build_vectorstore()
    elif args.query:
        response = query_rag(args.query)
        print(response)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
