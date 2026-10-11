"""
sign_image_mapper.py
---------------------
Maps letters/characters to sign-language reference image paths.

Used by the Reply mode to show a hearing person what each letter of their
message looks like in sign language.

All logic is kept in src/ so it is fully testable without Streamlit.

Image naming convention  (expected in data/sign_images/<language>/<LABEL>.png):
    A.png, B.png, ... Z.png

If an image is missing for a letter, a text placeholder dict is returned
with  {"type": "placeholder", "label": label}  so the UI can render a
fallback text tile. A warning is logged — the render never fails entirely.
"""

import logging
import os
from typing import List, Union

from src.config import BASE_DIR

logger = logging.getLogger(__name__)

SIGN_IMAGES_DIR = os.path.join(BASE_DIR, "data", "sign_images")


def image_path(language: str, label: str) -> str:
    """Return the expected on-disk path for a sign image."""
    return os.path.join(SIGN_IMAGES_DIR, language.upper(), f"{label.upper()}.png")


def letter_to_frame(language: str, label: str) -> dict:
    """
    Return a dict describing one frame in a reply sign sequence.
    
    Returns:
        {"type": "image", "path": <str>, "label": <str>}  if image exists
        {"type": "placeholder", "label": <str>}           if image is missing
    """
    path = image_path(language, label)
    if os.path.exists(path):
        return {"type": "image", "path": path, "label": label}
    logger.warning("Sign image missing: %s", path)
    return {"type": "placeholder", "label": label}


def text_to_sign_sequence(text: str, language: str) -> List[dict]:
    """
    Convert a string into a sequence of sign-image frames (one per letter).

    - Spaces produce a {"type": "space"} marker (UI shows a gap).
    - Non-alpha characters produce a {"type": "placeholder", "label": ch} frame.
    - Empty input returns [].

    Never raises; missing assets are logged and returned as placeholders.
    """
    if not text:
        return []

    frames: List[dict] = []
    for ch in text.upper():
        if ch == " ":
            frames.append({"type": "space", "label": " "})
        elif ch.isalpha():
            frames.append(letter_to_frame(language, ch))
        else:
            # Punctuation, digits — show as placeholder
            frames.append({"type": "placeholder", "label": ch})
    return frames
