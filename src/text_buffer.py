"""
text_buffer.py
---------------
Turns a stream of committed letter/control-gesture labels into readable,
auto-corrected text.

Responsibilities:
- Append recognized letters to the current word.
- Handle control gestures: SPACE (finish word), BACKSPACE (delete last
  char), CLEAR (wipe everything), SPEAK (signal caller to trigger TTS).
- Auto-correct each finished word using a lightweight spell-checker so
  minor misrecognitions ("HELLP" -> "HELLO") don't ruin the sentence.
"""

from typing import List, Optional

from spellchecker import SpellChecker

from src.config import (
    GESTURE_BACKSPACE,
    GESTURE_CLEAR,
    GESTURE_SPACE,
    GESTURE_SPEAK,
    PHRASE_LABELS,
    PHRASE_GESTURES,
    SOS_PHRASE_LABEL,
)


class TextBuffer:
    def __init__(self, auto_correct: bool = True):
        self.auto_correct_enabled = auto_correct
        self._spell = SpellChecker()
        self.finished_sentence: List[str] = []   # list of corrected words/phrases
        # Tracks which finished_sentence indices are whole phrases (for backspace)
        self._is_phrase: List[bool] = []
        self.current_word: str = ""
        self.suggestions: List[str] = []         # top suggestions for current_word
        self.last_correction: Optional[str] = None
        self.speak_requested = False
        self.sos_triggered = False               # set True when SOS_PHRASE_LABEL committed
        self.last_phrase: Optional[str] = None  # last phrase label committed (for UI highlight)

    # -- main entry point --------------------------------------------------
    def handle_label(self, label: str) -> None:
        """Feed one committed gesture label (a letter, control gesture, or phrase)."""
        if not label:
            return

        # ── Phrase shortcuts ─────────────────────────────────────────────────
        if label in PHRASE_GESTURES:
            self._finish_word()  # flush any in-progress word first
            phrase_text = PHRASE_LABELS[label]
            self.finished_sentence.append(phrase_text)
            self._is_phrase.append(True)
            self.last_phrase = label
            if label == SOS_PHRASE_LABEL:
                self.sos_triggered = True
                self.speak_requested = True  # override speak-aloud toggle for SOS
            self._update_suggestions()
            return

        # ── Control gestures ─────────────────────────────────────────────────
        if label == GESTURE_SPACE:
            self._finish_word()
        elif label == GESTURE_BACKSPACE:
            self._backspace()
        elif label == GESTURE_CLEAR:
            self.clear()
        elif label == GESTURE_SPEAK:
            self.speak_requested = True
        else:
            # Treat anything else as a printable character/letter.
            self.current_word += label

        self._update_suggestions()

    def _update_suggestions(self):
        if not self.auto_correct_enabled or not self.current_word or len(self.current_word) < 2:
            self.suggestions = []
            return
        
        lower = self.current_word.lower()
        candidates = self._spell.candidates(lower)
        if candidates:
            # Keep top 3 suggestions
            self.suggestions = [c.capitalize() if self.current_word.isupper() else c for c in list(candidates)[:3]]
        else:
            self.suggestions = []

    # -- word/sentence management -------------------------------------------
    def _finish_word(self) -> None:
        if not self.current_word:
            return
        corrected = self._autocorrect(self.current_word)
        if corrected.upper() != self.current_word.upper():
            self.last_correction = f"{self.current_word} -> {corrected}"
        else:
            self.last_correction = None
        self.finished_sentence.append(corrected)
        self._is_phrase.append(False)
        self.current_word = ""

    def _backspace(self) -> None:
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.finished_sentence:
            # If last token is a phrase, remove it as one unit.
            # If it's a word, pop it back into the editable buffer (minus last char).
            if self._is_phrase and self._is_phrase[-1]:
                self.finished_sentence.pop()
                self._is_phrase.pop()
            else:
                self.current_word = self.finished_sentence.pop()[:-1]
                if self._is_phrase:
                    self._is_phrase.pop()

    def clear(self) -> None:
        self.finished_sentence = []
        self._is_phrase = []
        self.current_word = ""
        self.last_correction = None
        self.speak_requested = False
        self.sos_triggered = False
        self.last_phrase = None

    def _autocorrect(self, word: str) -> str:
        if not self.auto_correct_enabled or len(word) < 2:
            return word
        lower = word.lower()
        if lower in self._spell:
            return word.capitalize() if word.isupper() and len(word) > 1 else word
        suggestion = self._spell.correction(lower)
        if suggestion and suggestion != lower:
            return suggestion.capitalize() if word[0].isupper() else suggestion
        return word

    # -- output --------------------------------------------------------------
    @property
    def sentence(self) -> str:
        """Full sentence so far, including the word currently being typed."""
        parts = list(self.finished_sentence)
        if self.current_word:
            parts.append(self.current_word)
        return " ".join(parts)

    def consume_speak_request(self) -> bool:
        """Returns True once, then resets, so TTS fires exactly once per gesture."""
        if self.speak_requested:
            self.speak_requested = False
            return True
        return False

    def consume_sos(self) -> bool:
        """Returns True once when an SOS phrase was just committed, then resets."""
        if self.sos_triggered:
            self.sos_triggered = False
            return True
        return False

    def get_final_text(self) -> str:
        """Flushes the in-progress word (auto-corrected) and returns full text."""
        if self.current_word:
            self._finish_word()
        return " ".join(self.finished_sentence)
