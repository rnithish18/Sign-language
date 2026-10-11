"""
translator.py
--------------
Regional-language translation with offline-safe fallback.

Uses deep-translator (GoogleTranslator) to translate English text to Tamil
or Hindi. Translation is cached per (text, target_language) pair to avoid
redundant API calls and to work faster on repeated sentences.

Behaviour when offline or on network error:
  - Returns the original English text unchanged.
  - Sets result["error"] = <message> so the UI can show a notice.
  - Never raises an exception into the caller.

Translation is NEVER called per frame — callers must invoke it only on
sentence-commit events (word-boundary or SPEAK gesture).
"""

import functools
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Simple in-process LRU cache (survives the Streamlit session rerun thanks
# to @st.cache_resource on the caller, but also works in tests/CLI).
_cache: dict = {}


def translate(text: str, target_lang: str) -> dict:
    """
    Translate *text* from English to *target_lang* (ISO 639-1 code, e.g. "ta").

    Returns:
        {
            "original":    str,   # input text
            "translated":  str,   # translated text (== original if English or on error)
            "target_lang": str,
            "cached":      bool,
            "error":       str | None,  # set only on failure
        }
    """
    if not text or not text.strip():
        return {"original": text, "translated": text,
                "target_lang": target_lang, "cached": False, "error": None}

    if target_lang == "en":
        return {"original": text, "translated": text,
                "target_lang": "en", "cached": True, "error": None}

    cache_key = (text.strip(), target_lang)
    if cache_key in _cache:
        return {**_cache[cache_key], "cached": True}

    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="en", target=target_lang).translate(text)
        result = {
            "original": text,
            "translated": translated or text,
            "target_lang": target_lang,
            "cached": False,
            "error": None,
        }
        _cache[cache_key] = result
        return result

    except ImportError:
        msg = "deep-translator not installed. Run: pip install deep-translator"
        logger.warning(msg)
        return {"original": text, "translated": text,
                "target_lang": target_lang, "cached": False, "error": msg}

    except Exception as exc:
        msg = f"Translation unavailable (offline or API error): {exc}"
        logger.warning(msg)
        return {"original": text, "translated": text,
                "target_lang": target_lang, "cached": False, "error": msg}


def clear_cache():
    """Clear the translation cache (useful for testing)."""
    _cache.clear()
