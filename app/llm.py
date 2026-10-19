
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY not found. Check your .env file."
    )

client = Groq(api_key=api_key)


def generate_answer(question: str, context: str) -> str:
    """
    Answer document-specific questions using retrieved context,
    while also answering general knowledge questions when appropriate.
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": """
You are SupportIQ, a helpful document assistant.

You have access to passages retrieved from the user's uploaded document.
Your job is to answer questions accurately and naturally.

Follow these rules:

1. DOCUMENT-SPECIFIC QUESTIONS
If the user asks about the contents, rules, requirements, facts,
or instructions in the uploaded document, use the supplied context
as your primary evidence.

Do not invent document details.
If the context does not contain enough information, say so clearly.
You may then offer a general explanation if it would help, but
clearly distinguish that explanation from what the document states.

2. GENERAL KNOWLEDGE QUESTIONS
If the user asks a general knowledge question, answer it directly
using your general knowledge, even if the document does not discuss it.

For example, if the user asks whether Mosquitto is an insect,
explain that a mosquito is an insect, whereas Mosquitto is MQTT
broker software in the context of a technical document.

3. AMBIGUOUS QUESTIONS
If a term could have multiple meanings, explain the likely meanings
and use the question and document context to clarify which applies.

4. ACCURACY AND SOURCES
Never claim that general knowledge came from the uploaded document.
Do not invent page numbers, quotations, or source references.
The application displays retrieved document passages separately.

5. STYLE
Use clear, friendly, beginner-friendly language.
Explain technical terms when useful.
Answer the actual question first and avoid unnecessary detail.
""",
            },
            {
                "role": "user",
                "content": (
                    f"User question:\n{question}\n\n"
                    f"Retrieved document context:\n{context}"
                ),
            },
        ],
        temperature=0.2,
        max_completion_tokens=500,
    )

    return response.choices[0].message.content or (
        "Sorry, I couldn't generate an answer. Please try again."
    )
