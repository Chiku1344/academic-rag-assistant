from datetime import datetime
from html import escape
from typing import Any

import streamlit as st

from Backend.frontend.api_client import APIError, AcademicRAGClient, default_backend_url

SUGGESTED_QUESTIONS = (
    "Explain database normalization.",
    "What are the ACID properties?",
    "Compare primary keys and foreign keys.",
)


def _show_answer(result: dict[str, Any]) -> None:
    st.markdown("### Answer")
    with st.chat_message("user"):
        st.markdown(result.get("question", ""))
    with st.chat_message("assistant"):
        answer = result.get("answer", "")
        if answer:
            st.markdown(answer)
        else:
            st.info("The API returned an empty answer.")

    sources = result.get("sources", [])
    st.markdown("#### Sources")
    if not sources:
        st.info("No source passages were returned for this answer.")
        return

    for index, source in enumerate(sources, start=1):
        filename = str(source.get("source") or "Unknown document")
        page = source.get("page")
        score = source.get("score")
        snippet = str(source.get("text") or "").strip()
        title = f"{filename} · Page {page if page is not None else 'N/A'}"
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f"**Source {index} · {title}**")
            with right:
                if isinstance(score, (float, int)):
                    st.caption(f"Score {score:.3f}")
            if snippet:
                st.caption(snippet)
            else:
                st.caption("No retrieved text was included in this citation.")


def _request_answer(client: AcademicRAGClient, question: str) -> None:
    try:
        with st.spinner("Searching your study material and preparing an answer…"):
            result = client.ask(question)
        result["question"] = question
        st.session_state["last_answer"] = result
    except APIError as exc:
        st.session_state.pop("last_answer", None)
        st.error(str(exc))


def render_ask_page(client: AcademicRAGClient, session_documents: list[dict[str, Any]]) -> None:
    with st.container(border=True):
        st.markdown('<div class="academic-eyebrow">Academic RAG workspace</div>', unsafe_allow_html=True)
        st.title("Your Academic Research Assistant")
        st.markdown(
            "Ask questions. Explore your notes. Get answers grounded in your study material."
        )

    with st.container(border=True):
        metric_column, description_column = st.columns([1, 3], vertical_alignment="center")
        with metric_column:
            st.metric("Documents", len(session_documents))
        with description_column:
            st.caption(
                "Confirmed uploads this session. Only PDFs with successful backend ingestion responses count. "
                "This is a session count, not a complete server-side document inventory."
            )

    st.markdown("#### Start with a question")
    columns = st.columns(3)
    for column, question in zip(columns, SUGGESTED_QUESTIONS):
        with column:
            if st.button(
                question,
                key=f"suggest_{question}",
                width="stretch",
                wrap=True,
            ):
                st.session_state["question_input"] = question
                st.rerun()

    with st.form("question_form", clear_on_submit=False):
        question = st.text_area(
            "Your academic question",
            placeholder="Ask a question about material you have uploaded…",
            height=110,
            key="question_input",
        )
        submitted = st.form_submit_button("Ask AI", type="primary")

    if submitted:
        cleaned_question = question.strip()
        if not cleaned_question:
            st.warning("Enter a question before submitting.")
        else:
            _request_answer(client, cleaned_question)

    if "last_answer" in st.session_state:
        _show_answer(st.session_state["last_answer"])
    elif not session_documents:
        with st.container(border=True):
            st.markdown("#### Start by adding your study material")
            st.write(
                "Upload a text-based PDF in **My Documents**. Once ingestion is confirmed, "
                "ask questions here and review the returned source passages."
            )
            st.caption("Answers are based on the passages returned by the connected RAG service.")


def _render_session_document(document: dict[str, Any]) -> None:
    filename = escape(str(document.get("filename", "PDF document")))
    pages = document.get("total_pages", 0)
    chunks = document.get("total_chunks", 0)
    processed_at = document.get("processed_at", "")
    with st.container(border=True):
        name_column, status_column = st.columns([4, 1])
        with name_column:
            st.markdown(f"**{filename}**")
            st.caption(f"{pages} pages · {chunks} text chunks")
        with status_column:
            st.success("Indexed")
        if processed_at:
            st.caption(f"Confirmed {processed_at}")


def render_documents_page(client: AcademicRAGClient) -> None:
    st.markdown('<div class="academic-eyebrow">Your materials</div>', unsafe_allow_html=True)
    st.title("My Documents")
    st.write("Upload academic PDFs and index their text for grounded question answering.")

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        max_upload_size=25,
        help="PDF files up to 25 MB are supported.",
        key="academic_pdf_upload",
    )
    st.caption("Supported format: PDF · Maximum size: 25 MB")

    if uploaded_file and st.button("Process and index PDF", type="primary", key="ingest_pdf"):
        try:
            with st.spinner("Extracting pages, creating embeddings, and indexing…"):
                result = client.upload(uploaded_file.name, uploaded_file.getvalue())
            st.session_state.setdefault("session_documents", []).insert(
                0,
                {
                    "filename": result["filename"],
                    "total_pages": result["total_pages"],
                    "total_chunks": result["total_chunks"],
                    "processed_at": datetime.now().astimezone().strftime("%b %d, %Y · %I:%M %p"),
                },
            )
            st.success(
                f"Indexed **{result['filename']}** — "
                f"{result['total_pages']} pages and {result['total_chunks']} chunks."
            )
            if result["filename"] != uploaded_file.name:
                st.info(
                    "A file with that name already existed. The backend stored this upload "
                    f"as **{result['filename']}**."
                )
        except APIError as exc:
            st.error(f"Upload was not confirmed. {exc}")

    documents = st.session_state.get("session_documents", [])
    st.markdown("#### Successfully processed this session")
    st.caption(
        "Only uploads confirmed by FastAPI are shown here. The backend does not currently "
        "provide a persistent document-list endpoint."
    )
    if not documents:
        with st.container(border=True):
            st.markdown("**No confirmed documents yet**")
            st.write(
                "Select a PDF above and choose **Process and index PDF**. "
                "A document appears here only after the backend confirms ingestion."
            )
        return

    st.caption(f"{len(documents)} confirmed upload(s) in this session")
    for document in documents:
        _render_session_document(document)


def render_history_page(client: AcademicRAGClient) -> None:
    st.markdown('<div class="academic-eyebrow">Saved conversations</div>', unsafe_allow_html=True)
    st.title("History")
    st.write("Previously answered questions saved by the backend.")

    try:
        entries = client.history()
    except APIError as exc:
        st.error(str(exc))
        return

    if not entries:
        with st.container(border=True):
            st.markdown("**Your history is empty**")
            st.write("Questions are saved here after the backend successfully generates and logs an answer.")
        return

    selected_id = st.session_state.get("selected_history_id")
    for entry in entries:
        entry_id = entry.get("id")
        question = str(entry.get("question", "Saved question"))
        timestamp = entry.get("timestamp")
        label = f"{question[:90]}{'…' if len(question) > 90 else ''}"
        if timestamp:
            try:
                display_time = datetime.fromisoformat(timestamp).astimezone().strftime(
                    "%b %d, %Y · %I:%M %p"
                )
            except (TypeError, ValueError):
                display_time = str(timestamp)
        else:
            display_time = "Time unavailable"
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f"**{label}**")
                st.caption(display_time)
            with right:
                if st.button(
                    "Open",
                    key=f"history_{entry_id}",
                    disabled=entry_id is None,
                ):
                    st.session_state["selected_history_id"] = entry_id
                    st.session_state["selected_history_entry"] = entry

        if selected_id == entry_id and st.session_state.get("selected_history_entry"):
            selected = st.session_state["selected_history_entry"]
            with st.container(border=True):
                st.markdown("#### Earlier answer")
                st.markdown(selected.get("answer", ""))
                st.caption("Source passages are not stored in query history.")


def render_settings_page(client: AcademicRAGClient) -> None:
    st.markdown('<div class="academic-eyebrow">Workspace configuration</div>', unsafe_allow_html=True)
    st.title("Settings")
    st.write("Configure the API connection and review non-sensitive service readiness.")

    current_url = st.session_state.get("backend_url", default_backend_url())
    with st.form("backend_settings_form"):
        backend_url = st.text_input(
            "FastAPI base URL",
            value=current_url,
            help="The local default is http://127.0.0.1:8001.",
        ).strip().rstrip("/")
        save_url = st.form_submit_button("Save backend URL", type="primary")
    if save_url:
        if not backend_url.startswith(("http://", "https://")):
            st.error("Enter a URL beginning with http:// or https://.")
        else:
            st.session_state["backend_url"] = backend_url
            st.success("Backend URL saved for this Streamlit session.")
            st.rerun()

    st.markdown("#### Service configuration")
    try:
        status = client.status()
    except APIError as exc:
        st.error(str(exc))
        st.caption("Configuration status is unavailable until FastAPI can be reached.")
        return

    _status_row("Database", status.get("database_available"), "SQLite query history")
    _status_row("Pinecone", status.get("pinecone_configured"), "Vector search and document indexing")
    _status_row("Language model", status.get("llm_configured"), "Answer generation")
    st.caption(
        "Only configured/not-configured status is shown. Secret values are never sent to or "
        "displayed by the Streamlit frontend."
    )


def _status_row(label: str, available: Any, description: str) -> None:
    with st.container(border=True):
        status_column, description_column = st.columns([1, 3])
        with status_column:
            if available is True:
                st.success(f"{label} · Ready")
            else:
                st.warning(f"{label} · Needs attention")
        with description_column:
            st.write(description)
            if available is False:
                if label == "Database":
                    st.caption("The backend could not query its configured database.")
                elif label == "Pinecone":
                    st.caption("Set PINECONE_API_KEY in Backend/.env on the backend host.")
                elif label == "Language model":
                    st.caption("Set GROQ_API_KEY in Backend/.env on the backend host.")


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="academic-brand">
                <div class="academic-brand-mark">A</div>
                <div>
                    <div class="academic-brand-name">Academic RAG</div>
                    <div class="academic-brand-caption">Research workspace</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("##### Workspace")
        selected_page = st.radio(
            "Navigate",
            ["Ask AI", "My Documents", "History", "Settings"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"API · {st.session_state.get('backend_url', default_backend_url())}")
        documents = st.session_state.get("session_documents", [])
        st.caption(f"{len(documents)} confirmed upload(s) this session")
    return selected_page
