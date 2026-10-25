"""
tts_engine.py
-------------
Text-to-speech wrapper offering two backends:

- "offline" (pyttsx3): works with no internet, low-latency, robotic voice.
  Best for real-time "SPEAK" gesture triggers during a conversation.
- "online" (gTTS): natural-sounding Google TTS, requires internet, writes
  an mp3 file. Best for the Streamlit UI where we can just hand back an
  audio file to <audio> / st.audio.

Both are wrapped behind the same `speak()` / `synthesize_to_file()` API so
the rest of the app doesn't care which one is active.
"""

import logging
import os
import tempfile
import threading
from typing import Optional

logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self, backend: str = "offline", rate: int = 170):
        self.backend = backend
        self.rate = rate
        self._offline_engine = None
        if backend == "offline":
            self._init_offline()

    def _init_offline(self):
        try:
            import pyttsx3
            self._offline_engine = pyttsx3.init()
            self._offline_engine.setProperty("rate", self.rate)
        except Exception as exc:
            logger.warning("pyttsx3 init failed (%s), falling back to online backend", exc)
            self.backend = "online"
            self._offline_engine = None

    def speak(self, text: str, blocking: bool = False, lang: str = "en") -> None:
        """Speak text immediately. Non-English always routes through gTTS (online)."""
        if not text.strip():
            return

        # pyttsx3 has no Tamil/Hindi voice on most systems — force gTTS for
        # non-English regardless of the configured backend.
        if lang != "en" or self.backend == "online":
            try:
                path = self.synthesize_to_file(text, lang=lang)
                self._play_file(path)
            except Exception as exc:
                logger.warning("gTTS speak failed (%s): %s", lang, exc)
            return

        if blocking:
            self._offline_engine.say(text)
            self._offline_engine.runAndWait()
        else:
            threading.Thread(
                target=self._speak_offline_async, args=(text,), daemon=True
            ).start()

    def _speak_offline_async(self, text: str):
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.say(text)
        engine.runAndWait()

    def synthesize_to_file(self, text: str, out_path: Optional[str] = None,
                           lang: str = "en") -> str:
        """Render text to an mp3/wav file and return its path (used by Streamlit).
        
        Non-English always uses gTTS — pyttsx3 has no Tamil/Hindi voices.
        Raises on network failure so callers can catch and display a notice.
        """
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        if out_path is None:
            fd, out_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)

        if self.backend == "online" or lang != "en":
            from gtts import gTTS
            tts = gTTS(text=text, lang=lang)
            tts.save(out_path)
        else:
            import pyttsx3
            wav_path = out_path.replace(".mp3", ".wav")
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.save_to_file(text, wav_path)
            engine.runAndWait()
            out_path = wav_path

        return out_path

    def _play_file(self, path: str) -> None:
        """Best-effort local playback (mainly for CLI/testing, not Streamlit)."""
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif os.uname().sysname == "Darwin":
                os.system(f"afplay '{path}' &")
            else:
                os.system(f"xdg-open '{path}' >/dev/null 2>&1 &")
        except Exception:
            pass  # Playback is a nice-to-have; never crash the app over it.
