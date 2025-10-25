import os
import shutil
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever, SearchType
from langchain_classic.storage import LocalFileStore, create_kv_docstore

try:
    from openai import NotFoundError as OpenAINotFoundError  # type: ignore
except Exception:  # pragma: no cover
    OpenAINotFoundError = None

# 载入本地 .env 配置，优先使用用户在文件中设置的密钥
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
VECTOR_STORE_DIR = INDEX_PATH / "vectorstore"
DOCSTORE_DIR = INDEX_PATH / "docstore"
DEFAULT_PARENT_CHUNK_SIZE = max(CHUNK_SIZE * 2, 2000)
DEFAULT_CHILD_CHUNK_SIZE = max(min(CHUNK_SIZE // 2, 800), 200)
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", str(DEFAULT_PARENT_CHUNK_SIZE)))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", str(max(CHUNK_OVERLAP, 200))))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", str(DEFAULT_CHILD_CHUNK_SIZE)))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "100"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
DOC_ID_KEY = "doc_id"
_langsmith_flag = os.getenv("LANGSMITH_ENABLED", "").lower() in {"1", "true", "yes", "on"}
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "pdf-search-chatbot")
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
LANGSMITH_ENABLED = _langsmith_flag and bool(LANGSMITH_API_KEY)

# LANGSMITH 相关配置：只有在显式开启监控并提供 API Key 时才生效
if LANGSMITH_ENABLED:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    if LANGSMITH_PROJECT:
        os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)


def faiss_index_available() -> bool:
    """Return True when a previously saved FAISS index is ready to load."""
    # 如果向量索引或父文档存储目录不存在，说明尚未构建索引
    if not VECTOR_STORE_DIR.exists() or not DOCSTORE_DIR.exists():
        return False
    try:
        has_vector_data = any(VECTOR_STORE_DIR.iterdir())
    except FileNotFoundError:
        has_vector_data = False
    try:
        has_docstore_data = any(DOCSTORE_DIR.rglob("*"))
    except FileNotFoundError:
        has_docstore_data = False
    return has_vector_data and has_docstore_data


def ensure_session_state():
    """初始化 Streamlit 会话态，避免首次访问时 KeyError。"""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "chat_summary" not in st.session_state:
        st.session_state["chat_summary"] = ""
    if "index_ready" not in st.session_state:
        st.session_state["index_ready"] = faiss_index_available()
    if "langsmith_session_id" not in st.session_state:
        st.session_state["langsmith_session_id"] = str(uuid4())


def format_chat_history(limit: int = 6) -> str:
    """将最近的对话历史格式化为字符串，供提示词引用。"""
    ensure_session_state()
    recent = st.session_state["chat_history"][-limit:]
    if not recent:
        return ""
    return "\n".join(f"{entry['role'].capitalize()}: {entry['content']}" for entry in recent)


@lru_cache(maxsize=1)
def get_embedding_model():
    """根据配置选择嵌入模型，支持 Hugging Face 或 DeepSeek 云端嵌入。"""
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
    """生成对话总结链，用于维护长对话的摘要记忆。"""
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
            "New dialogue:\nUser: {question}\nAssistant: {answer}\n\n"
            "Please provide an updated, concise summary in English, preserving the key details."
        ),
        input_variables=["summary", "question", "answer"],
    )
    return summary_prompt | summarizer

def get_pdf_documents(pdf_docs) -> list[Document]:
    """将上传的 PDF 文件解析为 LangChain Document 对象，每页一条记录。"""
    if not pdf_docs:
        return []

    documents: list[Document] = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            metadata = {
                "source": getattr(pdf, "name", "uploaded.pdf"),
                "page": page_number,
            }
            documents.append(Document(page_content=page_text, metadata=metadata))
    return documents


def build_multi_vector_index(documents: list[Document]) -> None:
    """基于父/子双层切片构建 MultiVector 检索所需的向量索引。"""
    if not documents:
        raise ValueError("No valid document text found. Please upload readable PDF files.")

    if DOCSTORE_DIR.exists():
        shutil.rmtree(DOCSTORE_DIR)
    if VECTOR_STORE_DIR.exists():
        shutil.rmtree(VECTOR_STORE_DIR)

    DOCSTORE_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    parent_docs = parent_splitter.split_documents(documents)
    if not parent_docs:
        raise ValueError("Failed to generate parent chunks from the provided documents.")

    byte_store = LocalFileStore(str(DOCSTORE_DIR))
    docstore = create_kv_docstore(byte_store)

    child_docs: list[Document] = []
    for parent_doc in parent_docs:
        doc_id = str(uuid4())
        docstore.mset([(doc_id, parent_doc)])
        splits = child_splitter.split_documents([parent_doc])
        for chunk in splits:
            chunk.metadata = dict(chunk.metadata)
            chunk.metadata[DOC_ID_KEY] = doc_id
            child_docs.append(chunk)

    if not child_docs:
        raise ValueError("Unable to create child chunks for vector indexing.")

    embeddings = get_embedding_model()
    try:
        vector_store = FAISS.from_documents(child_docs, embedding=embeddings)
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
    vector_store.save_local(str(VECTOR_STORE_DIR))


def load_multi_vector_retriever() -> MultiVectorRetriever:
    """加载本地保存的向量索引与父文档存储，返回 MultiVector 检索器。"""
    embeddings = get_embedding_model()
    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    byte_store = LocalFileStore(str(DOCSTORE_DIR))
    return MultiVectorRetriever(
        vectorstore=vector_store,
        byte_store=byte_store,
        id_key=DOC_ID_KEY,
        search_kwargs={"k": RETRIEVAL_TOP_K},
    )


def build_rag_pipeline(retriever: MultiVectorRetriever):
    """将检索→聚合→重排→生成串联为可追踪的 LangChain Runnable 流水线。"""
    vector_store = retriever.vectorstore
    docstore = retriever.docstore
    search_kwargs = getattr(retriever, "search_kwargs", {}) or {}
    search_type = getattr(retriever, "search_type", SearchType.similarity)

    def _load_index(_: dict, *, config: RunnableConfig | None = None) -> dict:
        """收集索引基本信息，便于在 LangSmith/前端调试观察。"""
        index_size = None
        if hasattr(vector_store, "index"):
            index_obj = getattr(vector_store, "index")
            index_size = getattr(index_obj, "ntotal", None)
        return {
            "vector_store_path": str(VECTOR_STORE_DIR),
            "doc_store_path": str(DOCSTORE_DIR),
            "vector_store_size": index_size,
            "search_type": getattr(search_type, "value", str(search_type)),
            "retrieval_top_k": search_kwargs.get("k", RETRIEVAL_TOP_K),
        }

    def _child_retrieve(inputs: dict, *, config: RunnableConfig | None = None) -> list[Document]:
        """通过向量库召回子片段，并补充相似度分数。"""
        question = inputs["question"]
        try:
            if search_type == SearchType.mmr:
                docs = vector_store.max_marginal_relevance_search(question, **search_kwargs)
            elif search_type == SearchType.similarity_score_threshold:
                raw = vector_store.similarity_search_with_relevance_scores(question, **search_kwargs)  # type: ignore[attr-defined]
                docs = []
                for doc, score in raw:
                    enriched = Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, "retrieval_score": float(score)},
                    )
                    docs.append(enriched)
            else:
                if hasattr(vector_store, "similarity_search_with_relevance_scores"):
                    try:
                        raw = vector_store.similarity_search_with_relevance_scores(question, **search_kwargs)  # type: ignore[attr-defined]
                    except Exception:
                        raw = []
                    if raw:
                        docs = []
                        for doc, score in raw:
                            enriched = Document(
                                page_content=doc.page_content,
                                metadata={**doc.metadata, "retrieval_score": float(score)},
                            )
                            docs.append(enriched)
                        return docs
                docs = vector_store.similarity_search(question, **search_kwargs)
        except Exception as exc:  # pragma: no cover - surface retrieval errors
            raise RuntimeError(f"Vector search failed: {exc}") from exc

        normalized: list[Document] = []
        for doc in docs:
            metadata = dict(getattr(doc, "metadata", {}))
            metadata.setdefault("retrieval_score", None)
            normalized.append(Document(page_content=doc.page_content, metadata=metadata))
        return normalized

    def _parent_lookup(inputs: dict, *, config: RunnableConfig | None = None) -> list[Document]:
        """根据子片段 doc_id 回查父文档，合并长上下文。"""
        child_docs: list[Document] = inputs.get("child_docs") or []
        doc_ids: list[str] = []
        for doc in child_docs:
            doc_id = doc.metadata.get(DOC_ID_KEY)
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)

        if not doc_ids:
            return []

        raw_parents = docstore.mget(doc_ids)
        parent_docs: list[Document] = []
        for doc_id, parent in zip(doc_ids, raw_parents):
            if parent is None:
                continue
            metadata = dict(parent.metadata)
            metadata[DOC_ID_KEY] = doc_id
            parent_docs.append(Document(page_content=parent.page_content, metadata=metadata))
        return parent_docs

    def _group_children(inputs: dict, *, config: RunnableConfig | None = None) -> list[dict]:
        """将每个父文档下的子片段聚类，方便调试与评分。"""
        child_docs: list[Document] = inputs.get("child_docs") or []
        parent_docs: list[Document] = inputs.get("parent_docs") or []
        grouped_children: defaultdict[str, list[Document]] = defaultdict(list)
        for child in child_docs:
            doc_id = child.metadata.get(DOC_ID_KEY)
            if doc_id:
                grouped_children[doc_id].append(child)

        grouped: list[dict] = []
        for parent in parent_docs:
            doc_id = parent.metadata.get(DOC_ID_KEY)
            grouped.append(
                {
                    "doc_id": doc_id,
                    "parent": parent,
                    "children": grouped_children.get(doc_id, []),
                }
            )
        return grouped

    def _rerank(inputs: dict, *, config: RunnableConfig | None = None) -> list[Document]:
        """对父文档做简单的关键字与子片段得分融合重排。"""
        question = inputs["question"].lower()
        keywords = [token for token in question.split() if token]
        docs: list[Document] = inputs.get("parent_docs") or []
        if not docs:
            return []

        children_lookup = {
            entry["doc_id"]: entry["children"] for entry in inputs.get("child_parent_matches", [])
        }

        scored_docs = []
        for position, doc in enumerate(docs):
            text = doc.page_content.lower()
            score = sum(text.count(token) for token in keywords) if keywords else 0.0
            if score == 0:
                related_children = children_lookup.get(doc.metadata.get(DOC_ID_KEY), [])
                score = sum(
                    child.metadata.get("retrieval_score", 0.0) or 0.0 for child in related_children
                )
            if score == 0:
                score = max(0.1, 0.05 * (len(docs) - position))
            enriched = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": float(score)},
            )
            scored_docs.append((score, position, enriched))

        scored_docs.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return [doc for _, _, doc in scored_docs]

    def _build_context(inputs: dict, *, config: RunnableConfig | None = None) -> str:
        docs: list[Document] = inputs.get("reranked_docs") or []
        return "\n\n".join(doc.page_content for doc in docs)

    def _prepare_generation(inputs: dict, *, config: RunnableConfig | None = None) -> dict:
        return {
            "summary": inputs["summary"],
            "history": inputs["history"],
            "context": inputs["context"],
            "question": inputs["question"],
        }

    def _format_answer(result, *, config: RunnableConfig | None = None) -> str:
        if hasattr(result, "content"):
            return str(result.content)
        return str(result)

    rag_chain = get_conversational_chain().with_config(
        run_name="generation",
        tags=["rag", "generation"],
    )

    pipeline = (
        RunnablePassthrough()
        .assign(
            index_stats=RunnableLambda(_load_index).with_config(
                run_name="load_index",
                tags=["rag", "index"],
            )
        )
        .assign(
            child_docs=RunnableLambda(_child_retrieve).with_config(
                run_name="child_retrieval",
                tags=["rag", "retrieval", "child"],
            )
        )
        .assign(
            parent_docs=RunnableLambda(_parent_lookup).with_config(
                run_name="parent_lookup",
                tags=["rag", "retrieval", "parent"],
            )
        )
        .assign(
            child_parent_matches=RunnableLambda(_group_children).with_config(
                run_name="child_parent_alignment",
                tags=["rag", "retrieval", "analysis"],
            )
        )
        .assign(
            reranked_docs=RunnableLambda(_rerank).with_config(
                run_name="rerank",
                tags=["rag", "rerank"],
            )
        )
        .assign(
            context=RunnableLambda(_build_context).with_config(
                run_name="context_assembly",
                tags=["rag", "context"],
            )
        )
        .assign(
            answer=(
                RunnableLambda(_prepare_generation).with_config(
                    run_name="generator_inputs",
                    tags=["rag", "generation"],
                )
                | rag_chain
                | RunnableLambda(_format_answer)
            )
        )
    )
    return pipeline


def build_langsmith_run_config(question: str) -> RunnableConfig | None:
    """为本次问答构建 LangSmith 追踪配置，附带常用元数据。"""
    tracing_flag = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if not tracing_flag or not api_key:
        return None

    ensure_session_state()
    metadata = {
        "question": question,
        "session_id": st.session_state["langsmith_session_id"],
    }
    tags = ["streamlit-chatapp", "pdf-rag"]
    run_name = "chatapp_conversational_chain"
    return RunnableConfig(tags=tags, metadata=metadata, run_name=run_name)


def update_conversation_memory(question: str, answer: str) -> None:
    """将最新问答纳入会话历史，并尝试刷新摘要。"""
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
    run_config = build_langsmith_run_config(question)
    invoke_kwargs = {"config": run_config} if run_config else {}
    updated_summary = summary_chain.invoke(summary_input, **invoke_kwargs)
    if hasattr(updated_summary, "content"):
        updated_summary = updated_summary.content
    st.session_state["chat_summary"] = str(updated_summary).strip()


def get_conversational_chain():
    """返回最终问答链（提示词 + 大模型），供 RAG 生成阶段调用。"""

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
    """处理用户提问：加载索引、执行 RAG 流水线，并展示结果。"""
    if not faiss_index_available():
        st.warning("No usable vector index detected. Please upload and process documents before asking questions.")
        return

    try:
        retriever = load_multi_vector_retriever()
    except Exception as err:
        st.error(f"Failed to load the vector index. Please re-upload and process the documents. Details: {err}")
        st.session_state["index_ready"] = False
        return

    ensure_session_state()
    summary_text = st.session_state.get("chat_summary") or "No summary yet."
    history_text = format_chat_history() or "No conversation history yet."

    run_config = build_langsmith_run_config(user_question)
    invoke_kwargs = {"config": run_config} if run_config else {}

    rag_pipeline = build_rag_pipeline(retriever)
    result = rag_pipeline.invoke(
        {
            "question": user_question,
            "summary": summary_text,
            "history": history_text,
        },
        **invoke_kwargs,
    )

    docs = result.get("reranked_docs") or []
    if not docs:
        st.info("No related content was retrieved. Try refining your question or reprocessing the documents.")
        return

    answer_text = result.get("answer", "")
    if not answer_text:
        st.error("Failed to generate a response. Please try again.")
        return

    st.write("Reply: ", answer_text)
    st.session_state["last_rag_debug"] = result

    # 将检索到的父文档位置信息直接展示给终端用户
    highlighted_sources = []
    for doc in docs[: RETRIEVAL_TOP_K]:
        meta = doc.metadata or {}
        source = meta.get("source", "未命名文件")
        page = meta.get("page")
        rerank_score = meta.get("rerank_score")
        doc_id = meta.get(DOC_ID_KEY, "未标记")
        page_info = f"第 {page} 页" if page is not None else "页码未知"
        highlighted_sources.append(
            f"- 文档：`{source}`（{page_info}，doc_id `{doc_id}`，rerank_score={rerank_score}）"
        )

    if highlighted_sources:
        st.markdown("**参考来源（按相关性排序）**")
        for item in highlighted_sources:
            st.markdown(item)

    # 展开面板提供完整的检索细节，方便排查召回与重排问题
    with st.expander("RAG 调试信息", expanded=False):
        index_stats = result.get("index_stats")
        if index_stats:
            st.markdown("**索引状态**")
            st.json(index_stats)

        child_docs = result.get("child_docs") or []
        if child_docs:
            st.markdown("**子文档检索结果**")
            for idx, doc in enumerate(child_docs, start=1):
                meta = doc.metadata or {}
                score = meta.get("retrieval_score")
                doc_id = meta.get(DOC_ID_KEY, "未标记")
                snippet = doc.page_content.replace("\n", " ")[:300]
                st.markdown(
                    f"{idx}. doc_id: `{doc_id}` · score: {score}\n\n> {snippet}{'...' if len(doc.page_content) > 300 else ''}"
                )

        parent_matches = result.get("child_parent_matches") or []
        if parent_matches:
            st.markdown("**父文档聚合**")
            for entry in parent_matches:
                doc_id = entry.get("doc_id", "未标记")
                parent_doc: Document | None = entry.get("parent")
                child_count = len(entry.get("children") or [])
                meta = parent_doc.metadata if parent_doc else {}
                source = meta.get("source")
                page_no = meta.get("page")
                summary_snippet = (
                    parent_doc.page_content.replace("\n", " ")[:200] if parent_doc else ""
                )
                source_info = f" · source: `{source}`" if source else ""
                page_info = f" · page: {page_no}" if page_no else ""
                st.markdown(
                    f"- doc_id `{doc_id}` · 子片段数: {child_count}{source_info}{page_info}\n"
                    f"  > {summary_snippet}{'...' if parent_doc and len(parent_doc.page_content) > 200 else ''}"
                )

        reranked_docs = result.get("reranked_docs") or []
        if reranked_docs:
            st.markdown("**重排得分**")
            for idx, doc in enumerate(reranked_docs, start=1):
                score = doc.metadata.get("rerank_score")
                doc_id = doc.metadata.get(DOC_ID_KEY, "未标记")
                st.markdown(f"{idx}. doc_id: `{doc_id}` · rerank_score: {score}")

    update_conversation_memory(user_question, answer_text)


def main():
    """Streamlit 入口函数，负责渲染页面元素与交互逻辑。"""
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
                    documents = get_pdf_documents(pdf_docs)  # parse PDFs into documents
                    try:
                        build_multi_vector_index(documents)  # create multi-vector index
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
