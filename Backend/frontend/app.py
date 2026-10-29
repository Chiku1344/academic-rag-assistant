import os
from urllib.parse import urlparse

import streamlit as st

from Backend.frontend.api_client import (
    DEFAULT_BACKEND_URL,
    APIError,
    AcademicRAGAPI,
)
from Backend.frontend.styles import APP_CSS
from Backend.frontend.ui import page_heading, render_saved_answer, render_sources

st.set_page_config(
    page_title="Academic RAG | Research workspace",
    page_icon=":material/school:",
    layout="wide",
)
st.markdown(APP_CSS, unsafe_allow_html=True)

NAV_ITEMS = ("Ask AI", "My Documents", "History", "Settings")
SUGGESTED_QUESTIONS = (
    "Explain database normalization.",
    "What are the ACID properties?",
    "Compare primary keys and foreign keys.",
)


def _initial_backend_url() -> str:
    configured_url = os.getenv("ACADEMIC_RAG_API_URL", DEFAULT_BACKEND_URL).strip().rstrip("/")
    if _valid_backend_url(configured_url):
        return configured_url
    st.session_state["invalid_backend_url"] = True
    return DEFAULT_BACKEND_URL


def _valid_backend_url(value: str) -> bool:
    try:
        parsed = urlparse(value.strip())
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.netloc)
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        return False


def render_sidebar() -> tuple[str, AcademicRAGAPI]:
    st.session_state.setdefault("backend_url", _initial_backend_url())
    st.session_state.setdefault("question_input", "")
    st.session_state.setdefault("conversation", [])

    api = AcademicRAGAPI(st.session_state["backend_url"])
    with st.sidebar:
        st.markdown("## :material/school: Academic RAG")
        st.caption("Academic research workspace")
        st.divider()
        selected_page = st.radio(
            "Workspace",
            NAV_ITEMS,
            key="active_page",
            label_visibility="visible",
        )
        st.divider()
        st.caption("Backend")
        st.code(api.base_url, language=None)
        if st.session_state.pop("invalid_backend_url", False):
            st.warning("ACADEMIC_RAG_API_URL was invalid; using the local default.")
        try:
            api.health()
            st.success("API connected", icon=":material/check_circle:")
        except APIError:
            st.warning("API unavailable. Start FastAPI to use the workspace.")
    return selected_page, api


def _set_question(question: str) -> None:
    st.session_state["question_input"] = question


def render_ask_page(api: AcademicRAGAPI) -> None:
    page_heading(
        "Academic RAG workspace",
        "Your Academic Research Assistant",
        "Ask questions. Explore your notes. Get answers grounded in your study material.",
    )

    document_count = None
    try:
        document_count = api.get_documents()["total_documents"]
        with st.container(border=True):
            st.metric("Processed documents", document_count)
            st.caption("Count includes documents the backend confirmed as indexed.")
    except APIError:
        st.caption("The document count is unavailable while the backend is offline.")

    st.markdown("### Start with a question")
    question_columns = st.columns(3)
    for column, question in zip(question_columns, SUGGESTED_QUESTIONS):
        with column:
            st.button(
                question,
                key=f"suggest_{question}",
                on_click=_set_question,
                args=(question,),
                width="stretch",
            )

    with st.form("ask_form"):
        st.text_area(
            "Your academic question",
            placeholder="Ask a question about material you have uploaded…",
            key="question_input",
            height=100,
        )
        submitted = st.form_submit_button(
            "Ask AI",
            type="primary",
            icon=":material/auto_awesome:",
        )

    if submitted:
        question = st.session_state["question_input"].strip()
        if not question:
            st.warning("Enter a question before submitting.")
        else:
            try:
                with st.spinner("Searching your study material and preparing an answer…"):
                    result = api.ask(question)
                st.session_state["conversation"].append(
                    {
                        "question": result.get("question", question),
                        "answer": result["answer"],
                        "sources": result["sources"],
                    }
                )
                st.rerun()
            except APIError as exc:
                st.error(str(exc))

    if not st.session_state["conversation"]:
        with st.container(border=True):
            if document_count == 0:
                st.markdown("#### Start by adding your study material")
                st.write(
                    "Upload a text-based PDF in **My Documents**. Once ingestion is confirmed, "
                    "ask questions here and review the returned source passages."
                )
            elif document_count is not None:
                st.markdown("#### Your library is ready")
                st.write(
                    "Ask a question to explore your indexed material. Answers include the "
                    "retrieved passages and page references."
                )
            else:
                st.markdown("#### Connect to your research workspace")
                st.write(
                    "Start FastAPI to load your document library and ask questions."
                )
            st.caption("Answers are grounded in passages returned by the connected RAG service.")
    else:
        st.markdown("### Conversation")
        for exchange in st.session_state["conversation"]:
            with st.chat_message("user"):
                st.markdown(exchange["question"])
            with st.chat_message("assistant"):
                st.markdown(exchange["answer"])
                if not exchange["sources"]:
                    st.info("No relevant source passages were returned for this question.")
                else:
                    render_sources(exchange["sources"])


def render_documents_page(api: AcademicRAGAPI) -> None:
    page_heading(
        "Your research library",
        "My Documents",
        "Add academic PDFs to your searchable study collection.",
    )
    st.markdown("### Add a PDF")
    st.caption(
        "PDF only · Up to 25 MB · Text-based PDFs required · "
        "Same-named uploads get a unique filename; content is not deduplicated."
    )
    uploaded_file = st.file_uploader(
        "Choose a PDF to upload",
        type=["pdf"],
        accept_multiple_files=False,
        key="pdf_upload",
    )
    if uploaded_file is not None:
        st.caption(f"Selected: {uploaded_file.name} · {uploaded_file.size / (1024 * 1024):.1f} MB")

    if st.button(
        "Process and index PDF",
        type="primary",
        icon=":material/upload_file:",
        disabled=uploaded_file is None,
        key="process_pdf",
    ):
        try:
            with st.spinner("Extracting pages, creating embeddings, and indexing…"):
                result = api.upload_pdf(uploaded_file.name, uploaded_file.getvalue())
            st.success(
                f"Indexed **{result['filename']}** · {result['total_pages']} pages · "
                f"{result['total_chunks']} chunks."
            )
            if result["filename"] != uploaded_file.name:
                st.info(
                    "A file with that name already exists. The backend kept both files "
                    "by saving this upload under a unique filename."
                )
        except APIError as exc:
            st.error(str(exc))

    st.divider()
    st.markdown("### Indexed documents")
    try:
        documents = api.get_documents()["documents"]
    except APIError as exc:
        st.error(str(exc))
        return

    if not documents:
        with st.container(border=True):
            st.markdown("#### Your library is ready")
            st.write(
                "No successfully indexed documents are recorded yet. Upload a text-based PDF "
                "above; it will appear here only after the backend confirms ingestion."
            )
        return

    for document in documents:
        with st.container(border=True):
            filename, status = st.columns([4, 1])
            filename.markdown(f"**{document.get('filename', 'Untitled PDF')}**")
            status.markdown(":green[Indexed]")
            metrics = []
            if document.get("total_pages") is not None:
                metrics.append(f"{document['total_pages']} pages")
            if document.get("total_chunks") is not None:
                metrics.append(f"{document['total_chunks']} chunks")
            if metrics:
                st.caption(" · ".join(metrics))
            uploaded_at = document.get("uploaded_at")
            if uploaded_at:
                st.caption(f"Added {uploaded_at}")


def render_history_page(api: AcademicRAGAPI) -> None:
    page_heading(
        "Saved conversations",
        "History",
        "Revisit questions and answers saved by the backend.",
    )
    try:
        history = api.get_history()["history"]
    except APIError as exc:
        st.error(str(exc))
        return

    if not history:
        with st.container(border=True):
            st.markdown("#### No saved questions yet")
            st.write("Questions are saved here after the backend successfully answers them.")
        return

    for entry in history:
        question = entry.get("question") or "Question unavailable"
        with st.expander(question):
            render_saved_answer(
                question,
                entry.get("answer") or "No saved answer.",
                entry.get("timestamp"),
            )


def render_settings_page(api: AcademicRAGAPI) -> None:
    page_heading(
        "Workspace preferences",
        "Settings",
        "Configure the local backend connection and review non-sensitive service status.",
    )
    with st.form("backend_settings"):
        backend_url = st.text_input(
            "FastAPI base URL",
            value=api.base_url,
            help="For local development, use http://127.0.0.1:8001.",
        )
        save_url = st.form_submit_button("Save backend URL", type="primary")

    if save_url:
        cleaned_url = backend_url.strip().rstrip("/")
        if not _valid_backend_url(cleaned_url):
            st.error("Enter a valid HTTP or HTTPS URL without embedded credentials.")
        else:
            st.session_state["backend_url"] = cleaned_url
            st.success("Backend URL saved for this Streamlit session.")
            st.rerun()

    st.markdown("### Provider configuration")
    try:
        status = api.get_status()
    except APIError as exc:
        st.error(str(exc))
        return

    pinecone_column, groq_column = st.columns(2)
    with pinecone_column, st.container(border=True):
        st.markdown("**Pinecone**")
        if status["pinecone_configured"]:
            st.success("API key configured")
        else:
            st.warning("Missing PINECONE_API_KEY in the backend environment.")
    with groq_column, st.container(border=True):
        st.markdown("**Groq**")
        if status["groq_configured"]:
            st.success("API key configured")
        else:
            st.warning("Missing GROQ_API_KEY in the backend environment.")
    st.caption("Secret values are never sent to or displayed by the frontend.")


def main() -> None:
    selected_page, api = render_sidebar()
    if selected_page == "Ask AI":
        render_ask_page(api)
    elif selected_page == "My Documents":
        render_documents_page(api)
    elif selected_page == "History":
        render_history_page(api)
    else:
        render_settings_page(api)


main()
