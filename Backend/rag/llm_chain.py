from .generator import AnswerGenerator
from .retriever import Retriever


def query_rag_system(user_query: str) -> dict:
    """Run the project's Pinecone/Groq RAG path and return answer citations."""
    documents = Retriever().search(user_query, top_k=4)
    answer = AnswerGenerator().generate_answer(user_query, documents)
    sources = [
        {
            "source": document["source"],
            "page": document["page"],
            "content": document["text"][:200],
        }
        for document in documents
    ]
    return {"answer": answer, "sources": sources}
