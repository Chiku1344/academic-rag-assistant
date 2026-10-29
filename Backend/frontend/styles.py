import streamlit as st


def apply_styles() -> None:
    st.html(
        """
        <style>
        :root {
            --academic-ink: #17233c;
            --academic-muted: #65728a;
            --academic-primary: #4257b2;
            --academic-border: #e4e8f0;
            --academic-surface: #ffffff;
            --academic-background: #f5f7fb;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--academic-background);
            color: var(--academic-ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background: #fbfcfe;
            border-right: 1px solid var(--academic-border);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--academic-muted);
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 0.65rem;
            padding: 0.38rem 0.58rem;
        }

        h1, h2, h3 {
            color: var(--academic-ink);
            letter-spacing: -0.025em;
        }

        h1 {
            font-weight: 720;
        }

        [data-testid="stCaptionContainer"] {
            color: #59667d;
            opacity: 1;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--academic-surface);
            border-color: var(--academic-border);
            border-radius: 1rem;
        }

        .stButton > button,
        [data-testid="stFormSubmitButton"] > button {
            min-height: 2.65rem;
            border-radius: 0.7rem;
            border: 1px solid #dce2ee;
            font-weight: 620;
            transition: border-color 120ms ease, background 120ms ease;
        }

        .stButton > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            border-color: var(--academic-primary);
            color: var(--academic-primary);
        }

        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input {
            border-radius: 0.7rem;
            border-color: #dce2ee;
        }

        [data-testid="stFileUploader"] section {
            border-radius: 0.9rem;
            border-color: #cbd4e5;
            background: #fafbfe;
        }

        .academic-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0.4rem 0 1.8rem;
        }

        .academic-brand-mark {
            width: 2.45rem;
            height: 2.45rem;
            display: grid;
            place-items: center;
            border-radius: 0.8rem;
            background: #e9edff;
            color: #344a9f;
            font-size: 1.15rem;
            font-weight: 750;
        }

        .academic-brand-name {
            color: var(--academic-ink);
            font-weight: 720;
            line-height: 1.15;
        }

        .academic-brand-caption {
            margin-top: 0.2rem;
            color: var(--academic-muted);
            font-size: 0.76rem;
        }

        .academic-eyebrow {
            color: var(--academic-primary);
            font-size: 0.76rem;
            font-weight: 720;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .academic-muted {
            color: var(--academic-muted);
        }

        .academic-answer {
            padding: 0.25rem 0 0.5rem;
            line-height: 1.75;
        }

        @media (max-width: 720px) {
            [data-testid="stMainBlockContainer"] {
                padding: 1.2rem 1rem 2.5rem;
            }

            h1 {
                font-size: 2rem;
            }
        }
        </style>
        """
    )
