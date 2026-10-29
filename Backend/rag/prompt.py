def build_academic_prompt(query: str, contexts: list[dict]) -> str:
    """Combine retrieved chunks with their source and page for grounded answers."""
    formatted_context = "\n\n".join(
        f"[Source {idx}: {ctx.get('source', 'Unknown document')} | "
        f"Page {ctx.get('page', 'N/A')}]\n{ctx['text']}"
        for idx, ctx in enumerate(contexts, start=1)
        if ctx.get("text")
    )

    prompt = f"""You are an expert academic tutor. Answer the student's question based strictly on the provided context.
If the context does not contain enough information to answer, state clearly that the answer is not present in the reference documents.
Do not invent citations. Refer to sources by document name and page when relevant.

Context Information:
---------------------
{formatted_context}
---------------------

Student Question: {query}

Provide a structured, accurate answer:
"""
    return prompt
