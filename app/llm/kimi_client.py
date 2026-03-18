from openai import OpenAI
from app.config import MOONSHOT_API_KEY, KIMI_BASE_URL, CHAT_MODEL

client = OpenAI(
    api_key=MOONSHOT_API_KEY,
    base_url=KIMI_BASE_URL,
)


def ask_kimi_with_context(question: str, context_chunks: list[str]) -> str:
    context_text = "\n\n".join(context_chunks)

    prompt = f"""
You are an AI assistant that answers questions using ONLY the provided document context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."

Context:
{context_text}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You answer questions strictly using the provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content