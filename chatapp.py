import os
from functools import lru_cache
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

try:
    from openai import NotFoundError as OpenAINotFoundError  # type: ignore
except Exception:  # pragma: no cover
    OpenAINotFoundError = None

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
DEEPSEEK_EMBED_MODEL = os.getenv(
    "DEEPSEEK_EMBEDDING_MODEL", "deepseek-text-embedding"
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "faiss_index"))


def faiss_index_available() -> bool:
    """Return True when a previously saved FAISS index is ready to load."""
    if not INDEX_PATH.exists():
        return False
    if INDEX_PATH.is_dir():
        return any(INDEX_PATH.iterdir())
    return INDEX_PATH.stat().st_size > 0


def ensure_session_state():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "chat_summary" not in st.session_state:
        st.session_state["chat_summary"] = ""
    if "index_ready" not in st.session_state:
        st.session_state["index_ready"] = faiss_index_available()


def format_chat_history(limit: int = 6) -> str:
    ensure_session_state()
    recent = st.session_state["chat_history"][-limit:]
    if not recent:
        return ""
    return "\n".join(f"{entry['role'].capitalize()}: {entry['content']}" for entry in recent)


@lru_cache(maxsize=1)
def get_embedding_model():
    if EMBEDDING_PROVIDER == "huggingface":
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if EMBEDDING_PROVIDER != "deepseek":
        raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")

    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "DEEPSEEK_API_KEY is not configured; the embedding provider is set to DeepSeek and vectors cannot be generated."
        )

    return OpenAIEmbeddings(
        model=DEEPSEEK_EMBED_MODEL,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_API_BASE,
    )


@lru_cache(maxsize=1)
def get_summary_chain():
    if not DEEPSEEK_API_KEY:
        raise ValueError("Conversation summarization requires DEEPSEEK_API_KEY to call the model.")

    summarizer = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.2,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_API_BASE,
    )
    summary_prompt = PromptTemplate(
        template=(
            "You are a conversation-summarizing assistant "
            "who weaves the newest exchange into the existing summary."
            "\n\nCurrent summary:\n{summary}\n\n"
            "New dialogue:\nUser: {question}\Assistant: {answer}\n\n"
            "Please provide an updated, concise summary in English, preserving the key details."
        ),
        input_variables=["summary", "question", "answer"],
    )
    return summary_prompt | summarizer

def get_pdf_text(pdf_docs):
    if not pdf_docs:
        return ""

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text



def get_text_chunks(text):
    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No valid text chunks were produced, unable to build the vector index.")

    embeddings = get_embedding_model()
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    except Exception as err:
        if (
            EMBEDDING_PROVIDER == "deepseek"
            and OpenAINotFoundError
            and isinstance(err, OpenAINotFoundError)
        ):
            raise ValueError(
                "DeepSeek embedding endpoint returned 404. Please confirm your account has access to the embedding model, "
                "or set EMBEDDING_PROVIDER to huggingface to use local embeddings."
            ) from err
        raise
    vector_store.save_local(str(INDEX_PATH))


def update_conversation_memory(question: str, answer: str) -> None:
    ensure_session_state()
    st.session_state["chat_history"].append({"role": "user", "content": question})
    st.session_state["chat_history"].append({"role": "assistant", "content": answer})

    try:
        summary_chain = get_summary_chain()
    except ValueError:
        return

    summary_input = {
        "summary": st.session_state.get("chat_summary", ""),
        "question": question,
        "answer": answer,
    }
    updated_summary = summary_chain.invoke(summary_input)
    if hasattr(updated_summary, "content"):
        updated_summary = updated_summary.content
    st.session_state["chat_summary"] = str(updated_summary).strip()


def get_conversational_chain():

    prompt_template = """
    You are a friendly PDF teaching assistant who answers questions using the summary, recent conversation, and retrieved documents.
    Respond with a warm, concise, and well-structured tone, adding supportive language when appropriate.
    Preserve English nouns from the retrieved content without translating them.
    If the context cannot answer the question, state that clearly and suggest a possible next step.

    Conversation summary:
    {summary}

    Recent chat history:
    {history}

    Document context:
    {context}

    User question: {question}

    Please provide a caring English reply:
    """

    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. Please add it to your environment before asking questions."
        )

    model = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.3,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_API_BASE,
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["summary", "history", "context", "question"],
    )
    return prompt | model



def user_input(user_question):
    embeddings = get_embedding_model()
    
    if not faiss_index_available():
        st.warning("No usable vector index detected. Please upload and process documents before asking questions.")
        return

    try:
        new_db = FAISS.load_local(
            str(INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as err:
        st.error(f"Failed to load the vector index. Please re-upload and process the documents. Details: {err}")
        st.session_state["index_ready"] = False
        return

    docs = new_db.similarity_search(user_question)

    if not docs:
        st.info("No related content was retrieved. Try refining your question or reprocessing the documents.")
        return

    context = "\n\n".join(doc.page_content for doc in docs)
    ensure_session_state()
    summary_text = st.session_state.get("chat_summary") or "No summary yet."
    history_text = format_chat_history() or "No conversation history yet."

    chain = get_conversational_chain()
    raw_response = chain.invoke(
        {
            "summary": summary_text,
            "history": history_text,
            "context": context,
            "question": user_question,
        }
    )
    answer_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
    st.write("Reply: ", answer_text)
    update_conversation_memory(user_question, answer_text)


def main():
    st.set_page_config("Multi PDF Chatbot", page_icon = ":scroll:")
    st.header("Product guide AI search 📚 - Chat Agent 🤖 ")
    ensure_session_state()

    if not DEEPSEEK_API_KEY:
        st.warning("Set DEEPSEEK_API_KEY in your .env file or environment to enable the DeepSeek model.")
    elif st.session_state.get("index_ready"):
        st.info("Existing vector index detected; you can start asking questions.")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        if not DEEPSEEK_API_KEY:
            st.error("DEEPSEEK_API_KEY is missing, so questions cannot be answered yet.")
        else:
            user_input(user_question)

    with st.sidebar:

        st.image("img/Robot.jpg")
        st.write("---")
        
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."): # user friendly message.
                if not pdf_docs:
                    if faiss_index_available():
                        st.session_state["index_ready"] = True
                        st.success("Existing vector index detected; you can start asking questions.")
                    else:
                        st.error("Please upload at least one PDF file.")
                else:
                    raw_text = get_pdf_text(pdf_docs) # get the pdf text
                    text_chunks = get_text_chunks(raw_text) # get the text chunks
                    try:
                        get_vector_store(text_chunks) # create vector store
                    except ValueError as err:
                        st.error(str(err))
                    else:
                        st.session_state["index_ready"] = faiss_index_available()
                        st.success("Documents processed successfully. You can start asking questions.")
        elif st.session_state.get("index_ready"):
            st.info("Existing vector index detected; you can start asking questions.")
        elif faiss_index_available():
            st.session_state["index_ready"] = True
            st.info("Existing vector index detected; you can start asking questions.")
        
        st.write("---")
        st.write("AI App created by @ JasonPlus")  # add this line to display the image

if __name__ == "__main__":
    main()
