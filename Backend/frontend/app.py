import streamlit as st

from Backend.frontend.api_client import AcademicRAGClient, default_backend_url
from Backend.frontend.styles import apply_styles
from Backend.frontend.ui import (
    render_ask_page,
    render_documents_page,
    render_history_page,
    render_settings_page,
    render_sidebar,
)

st.set_page_config(
    page_title="Academic RAG | Research workspace",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("backend_url", default_backend_url())
st.session_state.setdefault("session_documents", [])
apply_styles()

page = render_sidebar()
client = AcademicRAGClient(st.session_state["backend_url"])

if page == "Ask AI":
    render_ask_page(client, st.session_state["session_documents"])
elif page == "My Documents":
    render_documents_page(client)
elif page == "History":
    render_history_page(client)
elif page == "Settings":
    render_settings_page(client)
