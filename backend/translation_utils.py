# translation_utils.py

from dotenv import load_dotenv
load_dotenv()
import os
from openai import AzureOpenAI
from langdetect import detect, DetectorFactory

# Seed for consistent detection
DetectorFactory.seed = 0

translator_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# Your Azure chat deployment name
CHAT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

def translate_if_needed(text: str) -> str:
    """
    Detect language with `langdetect`. If non-English, send to Azure with
    a strict “only translate or echo back” system prompt at temperature=0.
    Otherwise return once.
    """
    try:
        lang = detect(text)
    except Exception:
        lang = "en"

    # Only translate if not already English
    if lang != "en":
        try:
            resp = translator_client.chat.completions.create(
                model=CHAT_MODEL,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a translation assistant. "
                            "Translate the user’s text into clear English. "
                            "If the text is a name, username, slug, or otherwise untranslatable, "
                            "do NOT ask questions—just return it verbatim."
                        )
                    },
                    {"role": "user", "content": text}
                ]
            )
            # Return whatever comes back (either the translation or the original echoed)
            return resp.choices[0].message.content.strip()
        except Exception:
            # On any error, just echo
            return text

    # If English detected, just echo
    return text
