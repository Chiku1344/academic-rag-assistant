from datetime import datetime
from typing import Any

import streamlit as st


def page_heading(eyebrow: str, title: str, subtitle: str) -> None:
    st.caption(eyebrow.upper())
    st.title(title)
    st.markdown(subtitle)


def render_sources(sources: list[dict[str, Any]]) -> None:
    st.markdown("#### Sources")
    if not sources:
        st.info("No source passages were returned for this answer.")
        return

    for index, source in enumerate(sources, start=1):
        name = source.get("source") or "Unknown document"
        page = source.get("page")
        score = source.get("score")
        excerpt = source.get("text") or "No excerpt was returned."
        page_text = f"Page {page}" if page is not None else "Page unavailable"
        score_text = (
            f"Score {score:.3f}"
            if isinstance(score, (int, float))
            else "Score unavailable"
        )

        with st.container(border=True):
            st.markdown(f"**{index}. {name}**")
            st.caption(f"{page_text}  ·  {score_text}")
            st.markdown(excerpt)


def render_saved_answer(question: str, answer: str, timestamp: str | None = None) -> None:
    if timestamp:
        try:
            parsed = datetime.fromisoformat(timestamp)
            st.caption(parsed.astimezone().strftime("%b %d, %Y · %I:%M %p"))
        except ValueError:
            st.caption(timestamp)
    st.markdown(f"**Question**\n\n{question}")
    st.markdown(answer)