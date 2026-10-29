import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from Backend.rag.errors import RAGConfigurationError, RAGServiceError
from Backend.rag.prompt import build_academic_prompt

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class AnswerGenerator:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        self.model_name = model_name

    def generate_answer(self, query: str, retrieved_docs: list[dict]) -> str:
        if not retrieved_docs:
            return (
                "I couldn't find sufficient information in the uploaded academic "
                "documents to answer this question."
            )

        if not self.api_key:
            raise RAGConfigurationError("GROQ_API_KEY is not configured.")
        if self.client is None:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as exc:
                raise RAGServiceError("The language model client could not be initialized.") from exc

        prompt = build_academic_prompt(query, retrieved_docs)
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
            )
            answer = completion.choices[0].message.content
        except Exception as exc:
            raise RAGServiceError("The language model could not generate an answer.") from exc

        if not isinstance(answer, str) or not answer.strip():
            raise RAGServiceError("The language model returned an empty answer.")
        return answer.strip()
