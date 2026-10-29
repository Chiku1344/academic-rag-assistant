APP_CSS = """
<style>
:root {
  color-scheme: light;
  --academic-ink: #17243b;
  --academic-muted: #63718a;
  --academic-border: #e4e9f1;
  --academic-accent: #4658ad;
  --academic-accent-soft: #eef0ff;
}

.stApp {
  background:
    radial-gradient(ellipse at 82% 0%, rgba(89, 107, 191, 0.06), transparent 34rem),
    #f7f8fb;
  color: var(--academic-ink);
}

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stSidebar"] {
  background: #f0f2f7;
  border-right: 1px solid var(--academic-border);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: var(--academic-muted);
}

[data-testid="stSidebar"] [role="radiogroup"] {
  gap: 0.3rem;
}

[data-testid="stSidebar"] [role="radio"] {
  border-radius: 0.7rem;
  padding: 0.45rem 0.65rem;
}

[data-testid="stMainBlockContainer"] {
  max-width: 1120px;
  padding-top: 2.4rem;
  padding-bottom: 4rem;
}

h1, h2, h3, h4 {
  color: var(--academic-ink);
  letter-spacing: -0.035em;
}

[data-testid="stCaptionContainer"] {
  color: var(--academic-muted);
}

[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255, 255, 255, 0.92);
  border-color: var(--academic-border);
  border-radius: 1rem;
  box-shadow: 0 8px 26px rgba(28, 42, 71, 0.035);
}

.stButton > button,
.stFormSubmitButton > button {
  border-radius: 0.7rem;
  font-weight: 600;
  min-height: 2.65rem;
  transition: border-color 120ms ease, background 120ms ease, transform 120ms ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
  border-color: #9aa6d9;
  transform: translateY(-1px);
}

.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
  background: var(--academic-accent);
  border-color: var(--academic-accent);
  color: #fff;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  border-radius: 0.75rem;
  border-color: #d7deea;
}

[data-testid="stFileUploader"] {
  border-radius: 0.9rem;
}

[data-testid="stAlert"] {
  border-radius: 0.8rem;
}

@media (max-width: 700px) {
  [data-testid="stMainBlockContainer"] {
    padding: 1.25rem 1rem 2.5rem;
  }
}
</style>
"""