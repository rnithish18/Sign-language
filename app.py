"""
app.py
------
SignBridge Streamlit application — dark-theme UI matching the design mockup.

Two modes selectable via sidebar toggle:
  1. Translate  - real-time sign -> text -> speech
  2. Learn      - guided lessons with instant feedback and progress tracking

Run:
    streamlit run app.py
"""

import time
from datetime import datetime
from typing import Optional

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from src.config import (
    DEFAULT_LANGUAGE, DEFAULT_OUTPUT_LANGUAGE, PHRASE_LABELS, PHRASE_GESTURES,
    PROGRESS_FILE, SOS_PHRASE_LABEL, SUPPORTED_LANGUAGES, SUPPORTED_OUTPUT_LANGUAGES,
)
from src.landmark_utils import HandLandmarkExtractor
from src.learning_mode import LessonSession, build_lesson_menu
from src.predictor import GesturePredictor, ModelNotFoundError
from src.sign_image_mapper import text_to_sign_sequence
from src.text_buffer import TextBuffer
from src.translator import translate
from src.tts_engine import TTSEngine
from utils.progress_store import ProgressStore
from utils.user_profiles import UserProfileManager, sanitize_username, validate_username

# ---------------------------------------------------------------------------
# Page config + global CSS (dark theme, large text, high contrast)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SignBridge", page_icon="🤟", layout="wide")

st.markdown("""
<style>
/* ── Dark base ──────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #1a1d27; }
[data-testid="stSidebar"] * { color: #e0e0e0; }

/* ── Typography ─────────────────────────────────────────────── */
.big-text { font-size: 2.2rem !important; font-weight: 700;
             color: #ffffff; letter-spacing: .04em; }
.label-sm { font-size: 0.75rem; color: #888; text-transform: uppercase;
             letter-spacing: .08em; margin-bottom: 2px; }

/* ── Status badge ───────────────────────────────────────────── */
.badge-detected { background:#0d6e3c; color:#7fffb9;
                  border-radius:6px; padding:4px 12px;
                  font-weight:700; font-size:.85rem; display:inline-block; }
.badge-none     { background:#3a1a1a; color:#ff8888;
                  border-radius:6px; padding:4px 12px;
                  font-weight:700; font-size:.85rem; display:inline-block; }
.badge-low-conf { background:#3a2e00; color:#ffd966;
                  border-radius:6px; padding:4px 12px;
                  font-weight:700; font-size:.85rem; display:inline-block; }

/* ── Prediction chip ────────────────────────────────────────── */
.pred-chip { background:#1a2a3a; border:1.5px solid #2a5a8a;
             border-radius:8px; padding:8px 18px;
             font-size:1.5rem; font-weight:800; color:#7ec8f7;
             display:inline-block; }

/* ── Text card ──────────────────────────────────────────────── */
.text-card { background:#141820; border:1.5px solid #2a2d3a;
             border-radius:12px; padding:20px 24px; }
.cursor { display:inline-block; width:2px; height:1.4em;
          background:#4a9eff; animation:blink 1s step-end infinite;
          vertical-align:text-bottom; margin-left:2px; }
@keyframes blink { 50% { opacity:0; } }

/* ── Suggestion chips ───────────────────────────────────────── */
.sugg-chip button { background:#1e3a5f !important; color:#7ec8f7 !important;
                    border:none !important; border-radius:20px !important;
                    font-size:.9rem !important; font-weight:600 !important;
                    padding:4px 16px !important; }

/* ── Session history ────────────────────────────────────────── */
.hist-row { display:flex; justify-content:space-between; align-items:center;
            padding:8px 0; border-bottom:1px solid #2a2d3a; }
.hist-text { font-size:1rem; color:#d0d0d0; font-weight:500; }
.hist-time { font-size:.8rem; color:#666; }

/* ── Sidebar toggles ────────────────────────────────────────── */
.profile-avatar { background:#2a5aff; color:#fff; border-radius:50%;
                  width:36px; height:36px; display:inline-flex;
                  align-items:center; justify-content:center;
                  font-weight:700; font-size:.9rem; margin-right:8px; }

/* ── Synthetic banner ───────────────────────────────────────── */
.synth-banner { background:#3a2000; border-left:4px solid #ff9900;
                border-radius:6px; padding:10px 16px;
                color:#ffcc66; font-size:.875rem; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------
@st.cache_resource
def get_progress_store() -> ProgressStore:
    return ProgressStore(PROGRESS_FILE)


@st.cache_resource
def get_profile_manager() -> UserProfileManager:
    return UserProfileManager(PROGRESS_FILE)


def get_predictor(language: str) -> Optional[GesturePredictor]:
    key = f"predictor_{language}"
    if key not in st.session_state:
        try:
            st.session_state[key] = GesturePredictor(language)
        except ModelNotFoundError:
            st.session_state[key] = None
    return st.session_state[key]


def init_session_state():
    st.session_state.setdefault("text_buffer", TextBuffer(auto_correct=True))
    st.session_state.setdefault("tts_engine", TTSEngine(backend="offline"))
    st.session_state.setdefault("mode", "Translate")
    st.session_state.setdefault("language", DEFAULT_LANGUAGE)
    st.session_state.setdefault("username", "guest")
    st.session_state.setdefault("lesson_session", None)
    st.session_state.setdefault("last_committed", "")
    st.session_state.setdefault("last_feedback", "")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("conf_thresh", 0.65)
    st.session_state.setdefault("hold_frames", 12)
    st.session_state.setdefault("speak_aloud", True)
    st.session_state.setdefault("output_language", DEFAULT_OUTPUT_LANGUAGE)
    st.session_state.setdefault("translated_text", "")
    st.session_state.setdefault("translation_error", None)

init_session_state()


# ---------------------------------------------------------------------------
# Video processor (background webrtc thread)
# ---------------------------------------------------------------------------
class SignProcessor:
    """
    streamlit-webrtc calls recv() once per frame in a background thread.
    Landmark extraction + sklearn predict + overlay drawing all happen here.
    Results are polled by the main thread via ctx.video_processor attributes.
    """

    def __init__(self, language: str, mode: str,
                 lesson_session: Optional[LessonSession]):
        self.language = language
        self.mode = mode
        self.lesson_session = lesson_session
        self.extractor = HandLandmarkExtractor()
        try:
            self.predictor: Optional[GesturePredictor] = GesturePredictor(language)
        except ModelNotFoundError:
            self.predictor = None

        self.smoothed_label = None
        self.confidence = 0.0
        self.progress = 0.0
        self.committed_label = None
        self.hand_detected = False
        self.handedness_str = ""      # e.g. "Left", "Right", "Left + right"

    # Gate verbose per-frame debug logging.  Set to True to re-enable.
    DEBUG_RECV = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        import traceback, logging, os

        # ── Debug log setup (writes to signbridge_debug.log beside app.py) ──
        _log = None
        if self.DEBUG_RECV:
            _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signbridge_debug.log")
            _log = logging.getLogger("signbridge.recv")
            if not _log.handlers:
                _h = logging.FileHandler(_log_path, mode="a")
                _h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
                _log.addHandler(_h)
                _log.setLevel(logging.DEBUG)

        try:
            img = frame.to_ndarray(format="bgr24")

            # ── FIX: force contiguous, writable array ──────────────────────
            # av.VideoFrame.to_ndarray() can return non-contiguous or
            # read-only arrays.  The mirror flip below ([:, ::-1, :]) makes
            # it worse by producing a strided view.  OpenCV drawing
            # functions (cv2.rectangle, cv2.putText, mp_drawing.draw_landmarks)
            # require a C-contiguous, writable array and throw a cryptic
            # "Layout of the output array img is incompatible with cv::Mat"
            # error otherwise.  np.ascontiguousarray + writability guard
            # fixes both properties in a single allocation.
            img = np.ascontiguousarray(img)
            if not img.flags.writeable:
                img = img.copy()

            # ── Frame diagnostics for the first 30 frames ──────────────────
            if self.DEBUG_RECV and _log:
                if not hasattr(self, "_diag_count"):
                    self._diag_count = 0
                if self._diag_count < 30:
                    _log.debug(
                        f"FRAME#{self._diag_count}: "
                        f"shape={img.shape} dtype={img.dtype} "
                        f"contiguous={img.flags['C_CONTIGUOUS']} "
                        f"writeable={img.flags.writeable} "
                        f"min={img.min()} max={img.max()} "
                        f"av_format={frame.format.name!r} "
                        f"predictor={'OK' if self.predictor else 'NONE'}"
                    )
                    self._diag_count += 1

            img = img[:, ::-1, :]   # mirror for natural selfie view
            # Mirror creates a non-contiguous view; fix again.
            img = np.ascontiguousarray(img)

            if self.predictor is None:
                if self.DEBUG_RECV and _log:
                    _log.debug("recv: predictor is None, returning raw frame")
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # Single MediaPipe pass on the clean (unannotated) BGR frame.
            mp_raw = self.extractor.process_raw(img)

            if self.DEBUG_RECV and _log:
                if not hasattr(self, "_diag_count"):
                    self._diag_count = 0
                if self._diag_count <= 30:
                    _log.debug(
                        f"FRAME#{self._diag_count - 1}: "
                        f"mp_raw.multi_hand_landmarks="
                        f"{bool(mp_raw and mp_raw.multi_hand_landmarks)}"
                    )

            self._update_handedness(mp_raw)
            frame_result = self.extractor.process_from_raw(img, mp_raw)
            feature_vector = frame_result.feature_vector if frame_result else None
            self.hand_detected = feature_vector is not None

            # Draw landmarks after feature extraction
            img = self.extractor.draw(img, mp_raw)

            result = self.predictor.update(feature_vector)
            self.smoothed_label = result["smoothed_label"]
            self.confidence = result["confidence"]
            self.progress = result["progress"]
            self.committed_label = result["committed_label"]

            self._draw_overlay(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception:
            tb = traceback.format_exc()
            if self.DEBUG_RECV and _log:
                _log.error(f"EXCEPTION in recv():\n{tb}")
            # Return original frame so the stream doesn't hard-stop
            try:
                return av.VideoFrame.from_ndarray(
                    frame.to_ndarray(format="bgr24"), format="bgr24"
                )
            except Exception:
                return frame

    def _update_handedness(self, mp_raw):
        if mp_raw is None or not hasattr(mp_raw, "multi_handedness") \
                or not mp_raw.multi_handedness:
            self.handedness_str = ""
            return
        labels = [h.classification[0].label for h in mp_raw.multi_handedness]
        if len(labels) == 2:
            self.handedness_str = "Left + right"
        elif labels:
            self.handedness_str = labels[0]
        else:
            self.handedness_str = ""

    # ── Public helper so tests can call it directly ───────────────────────
    @staticmethod
    def draw_live_letter_overlay(
        img: np.ndarray,
        smoothed_label,
        confidence: float,
        hand_detected: bool,
        confidence_threshold: float = 0.65,
    ) -> None:
        """
        Draw a large real-time letter overlay in the top-left corner.

        Rules:
        • No hand detected  → nothing drawn (avoids stale letters).
        • Hand detected, confidence ≥ threshold → bold cyan letter (actively
          counting toward commit).
        • Hand detected, confidence < threshold → smaller grey letter (best-
          guess shown, visually muted so the user knows it's uncertain).

        This is intentionally extracted as a static method so unit tests can
        call it without constructing a full SignProcessor / WebRTC context.
        """
        if not hand_detected or smoothed_label is None:
            return

        h, w = img.shape[:2]
        high_conf = confidence >= confidence_threshold

        # Semi-transparent dark background pill so the letter is always readable
        label_str = str(smoothed_label)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 3.2 if high_conf else 2.0
        thickness = 5 if high_conf else 2
        colour = (247, 200, 80) if high_conf else (130, 130, 130)   # gold / grey

        (tw, th), baseline = cv2.getTextSize(label_str, font, font_scale, thickness)
        pad = 14
        x0, y0 = 16, 16
        x1, y1 = x0 + tw + pad * 2, y0 + th + baseline + pad * 2

        # Clamp to frame bounds
        x1 = min(x1, w - 1)
        y1 = min(y1, h - 1)

        # Dark background rectangle (semi-opaque blend)
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        # Letter
        text_x = x0 + pad
        text_y = y0 + th + pad
        cv2.putText(img, label_str, (text_x, text_y), font,
                    font_scale, colour, thickness, cv2.LINE_AA)

        # Thin border: cyan when high-conf, dim-grey when low-conf
        border_colour = (200, 200, 50) if high_conf else (70, 70, 70)
        cv2.rectangle(img, (x0, y0), (x1, y1), border_colour, 2)

    def _draw_overlay(self, img: np.ndarray):
        h, w = img.shape[:2]
        threshold = self.predictor.confidence_threshold if self.predictor else 0.65

        # ── NEW: large real-time letter overlay (top-left) ─────────────────
        SignProcessor.draw_live_letter_overlay(
            img,
            self.smoothed_label,
            self.confidence,
            self.hand_detected,
            threshold,
        )

        # Prediction badge (bottom-left)
        if self.smoothed_label:
            conf_pct = int(self.confidence * 100)
            badge = f"{self.smoothed_label}  {conf_pct}% confident"
            cv2.rectangle(img, (0, h - 60), (300, h), (20, 30, 50), -1)
            cv2.putText(img, badge, (12, h - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (126, 200, 247), 2)

        # Progress bar (bottom-right)
        bar_total = 240
        bar_h = 10
        bar_x = w - bar_total - 12
        bar_y = h - bar_h - 8
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_total, bar_y + bar_h),
                      (50, 50, 50), -1)
        filled = int(bar_total * self.progress)
        if filled > 0:
            cv2.rectangle(img, (bar_x, bar_y),
                          (bar_x + filled, bar_y + bar_h), (50, 205, 50), -1)
        cv2.putText(img, "Hold to confirm",
                    (bar_x, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (160, 160, 160), 1)

        # Low-confidence hint (below the new overlay, clear of it)
        if self.hand_detected and self.confidence < threshold and self.smoothed_label is None:
            cv2.putText(img, "Low confidence — hold steady",
                        (12, h - 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 210, 60), 1)



# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # Logo + profile row
    pm = get_profile_manager()
    username = st.session_state["username"]
    initials = "".join(w[0].upper() for w in username.split()[:2]) or "?"
    st.markdown(
        f'<div style="display:flex;align-items:center;margin-bottom:4px">'
        f'<span class="profile-avatar">{initials}</span>'
        f'<div><b style="font-size:1rem">{username}</b><br>'
        f'<span style="font-size:.75rem;color:#888">Switch profile ↓</span></div></div>',
        unsafe_allow_html=True,
    )

    # Profile name input + validation
    new_name = st.text_input("Name", value=username, label_visibility="collapsed",
                             placeholder="Your name")
    if new_name != username:
        cleaned = sanitize_username(new_name)
        err = validate_username(cleaned)
        if err:
            st.error(err)
        else:
            pm.save_preferences(username, {
                "language": st.session_state["language"],
                "tts_backend": st.session_state["tts_engine"].backend,
                "conf_thresh": st.session_state.get("conf_thresh", 0.65),
                "hold_frames": st.session_state.get("hold_frames", 12),
                "auto_correct": st.session_state["text_buffer"].auto_correct_enabled,
            })
            st.session_state["username"] = cleaned
            prefs = pm.get_preferences(cleaned)
            st.session_state["language"] = prefs["language"]
            st.session_state["conf_thresh"] = prefs["conf_thresh"]
            st.session_state["hold_frames"] = prefs["hold_frames"]
            st.session_state["lesson_session"] = None
            st.rerun()

    known_users = pm.list_users()
    if known_users:
        switch_to = st.selectbox("Switch profile", ["— select —"] + known_users,
                                 index=0, key="profile_switcher",
                                 label_visibility="collapsed")
        if switch_to not in ("— select —", st.session_state["username"]):
            pm.save_preferences(st.session_state["username"], {
                "language": st.session_state["language"],
                "tts_backend": st.session_state["tts_engine"].backend,
                "conf_thresh": st.session_state.get("conf_thresh", 0.65),
                "hold_frames": st.session_state.get("hold_frames", 12),
                "auto_correct": st.session_state["text_buffer"].auto_correct_enabled,
            })
            st.session_state["username"] = switch_to
            prefs = pm.get_preferences(switch_to)
            st.session_state["language"] = prefs["language"]
            st.session_state["conf_thresh"] = prefs["conf_thresh"]
            st.session_state["hold_frames"] = prefs["hold_frames"]
            st.session_state["lesson_session"] = None
            st.rerun()

    with st.expander("⚠️ Danger zone"):
        if st.button("🗑️ Delete profile", type="secondary"):
            pm.delete_user(st.session_state["username"])
            st.session_state["username"] = "guest"
            st.session_state["lesson_session"] = None
            st.session_state["text_buffer"] = TextBuffer(auto_correct=True)
            st.rerun()

    st.divider()

    # Mode toggle
    st.markdown('<p class="label-sm">Mode</p>', unsafe_allow_html=True)
    col_t, col_l, col_r = st.columns(3)
    with col_t:
        if st.button("Translate",
                     type="primary" if st.session_state["mode"] == "Translate" else "secondary",
                     use_container_width=True):
            st.session_state["mode"] = "Translate"
            st.rerun()
    with col_l:
        if st.button("Learn",
                     type="primary" if st.session_state["mode"] == "Learn" else "secondary",
                     use_container_width=True):
            st.session_state["mode"] = "Learn"
            st.rerun()
    with col_r:
        if st.button("Reply",
                     type="primary" if st.session_state["mode"] == "Reply" else "secondary",
                     use_container_width=True):
            st.session_state["mode"] = "Reply"
            st.rerun()

    st.divider()

    # Input Language toggle
    st.markdown('<p class="label-sm">Input language</p>', unsafe_allow_html=True)
    col_isl, col_asl = st.columns(2)
    lang_options = SUPPORTED_LANGUAGES  # ["ASL", "ISL"]
    with col_isl:
        if st.button(lang_options[1] if len(lang_options) > 1 else "ISL",
                     type="primary" if st.session_state["language"] == (lang_options[1] if len(lang_options) > 1 else "ISL") else "secondary",
                     use_container_width=True):
            st.session_state["language"] = lang_options[1] if len(lang_options) > 1 else "ISL"
            st.rerun()
    with col_asl:
        if st.button(lang_options[0],
                     type="primary" if st.session_state["language"] == lang_options[0] else "secondary",
                     use_container_width=True):
            st.session_state["language"] = lang_options[0]
            st.rerun()

    st.divider()

    # Output Language toggle
    st.markdown('<p class="label-sm">Output language</p>', unsafe_allow_html=True)
    out_langs = list(SUPPORTED_OUTPUT_LANGUAGES.keys())
    out_lang = st.selectbox("Output language", out_langs, 
                            index=out_langs.index(st.session_state.get("output_language", DEFAULT_OUTPUT_LANGUAGE)),
                            label_visibility="collapsed")
    if out_lang != st.session_state.get("output_language"):
        st.session_state["output_language"] = out_lang
        # re-translate current sentence if any
        if st.session_state["text_buffer"].finished_sentence:
            tgt = SUPPORTED_OUTPUT_LANGUAGES[out_lang]
            res = translate(st.session_state["text_buffer"].get_final_text(), tgt)
            st.session_state["translated_text"] = res["translated"]
            st.session_state["translation_error"] = res["error"]
        st.rerun()

    st.divider()

    # Sliders
    conf_thresh = st.slider("Confidence %",
                            10, 100, int(st.session_state.get("conf_thresh", 0.65) * 100), 5,
                            format="%d%%")
    hold_frames = st.slider("Hold frames",
                            1, 30, int(st.session_state.get("hold_frames", 12)), 1)
    st.session_state["conf_thresh"] = conf_thresh / 100.0
    st.session_state["hold_frames"] = hold_frames

    # Toggles
    auto_correct = st.checkbox("Auto-correct",
                               value=st.session_state["text_buffer"].auto_correct_enabled)
    st.session_state["text_buffer"].auto_correct_enabled = auto_correct

    speak_aloud = st.checkbox("Speak aloud", value=st.session_state.get("speak_aloud", True))
    st.session_state["speak_aloud"] = speak_aloud

    # TTS backend (collapsed)
    with st.expander("Voice engine"):
        tts_backend = st.radio("Backend", ["offline", "online"],
                               index=0 if st.session_state["tts_engine"].backend == "offline" else 1)
        if st.session_state["tts_engine"].backend != tts_backend:
            st.session_state["tts_engine"] = TTSEngine(backend=tts_backend)

    # Auto-save prefs
    pm.save_preferences(st.session_state["username"], {
        "language": st.session_state["language"],
        "tts_backend": st.session_state["tts_engine"].backend,
        "conf_thresh": st.session_state["conf_thresh"],
        "hold_frames": st.session_state["hold_frames"],
        "auto_correct": auto_correct,
    })

    st.divider()

    # Model status + synthetic banner
    pred = get_predictor(st.session_state["language"])
    if pred is None:
        st.warning(
            f"No model for **{st.session_state['language']}**.\n\n"
            "```\npython -m src.data_collection "
            f"--language {st.session_state['language']}\n"
            f"python -m src.train_model --language {st.session_state['language']}\n```"
        )
    else:
        source = pred.metadata.get("data_source", "synthetic") if pred.metadata else "synthetic"
        is_synth = pred.metadata.get("synthetic", True) if pred.metadata else True
        
        if is_synth or source == "synthetic":
            st.markdown(
                '<div class="synth-banner">⚠️ <b>Prototype Mode</b> — Using synthetic dummy models '
                '(100% accuracy on noise, 0% on real hands). Collect real data for accurate results.</div>',
                unsafe_allow_html=True,
            )
        elif source == "public_dataset":
            st.info(
                "ℹ️ **Public Dataset Model**: Using a model trained on imported public image data. "
                "Performance may vary depending on how similar your webcam setup is to the dataset's images."
            )
        else:
            st.success(f"Model loaded for {st.session_state['language']} ✅ (Trained on your real webcam data!)")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _speak_text(text: str):
    if text.strip() and st.session_state.get("speak_aloud", True):
        # Use target TTS language if translating
        out_lang = st.session_state.get("output_language", DEFAULT_OUTPUT_LANGUAGE)
        lang_code = SUPPORTED_OUTPUT_LANGUAGES.get(out_lang, "en")
        st.session_state["tts_engine"].speak(text, lang=lang_code)


def _add_to_history(text: str):
    if text.strip():
        ts = datetime.now().strftime("%H:%M")
        st.session_state["history"].append({"text": text, "ts": ts})


# ---------------------------------------------------------------------------
# Translate mode
# ---------------------------------------------------------------------------
def render_translate_mode():
    lang = st.session_state["language"]
    mode = st.session_state["mode"]
    conf = st.session_state.get("conf_thresh", 0.65)
    hold = st.session_state.get("hold_frames", 12)
    lesson = st.session_state.get("lesson_session")
    
    def _make_processor_translate():
        p = SignProcessor(language=lang, mode=mode, lesson_session=lesson)
        if p.predictor:
            p.predictor.confidence_threshold = conf
            p.predictor.hold_frames_to_commit = hold
        return p

    pred = get_predictor(lang)

    col_cam, col_right = st.columns([3, 2])

    with col_cam:
        ctx = webrtc_streamer(
            key="signbridge-translate",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=_make_processor_translate,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Status badges below camera
        if ctx.state.playing and ctx.video_processor:
            proc: SignProcessor = ctx.video_processor
            if proc.predictor:
                proc.predictor.confidence_threshold = st.session_state["conf_thresh"]
                proc.predictor.hold_frames_to_commit = st.session_state["hold_frames"]

            c_badge, c_hand, c_conf = st.columns([2, 2, 3])
            with c_badge:
                if proc.hand_detected:
                    st.markdown('<span class="badge-detected">Hand detected</span>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-none">No hand</span>',
                                unsafe_allow_html=True)
            with c_hand:
                if proc.handedness_str:
                    st.caption(proc.handedness_str)
            with c_conf:
                if proc.hand_detected and proc.confidence < st.session_state["conf_thresh"]:
                    st.markdown('<span class="badge-low-conf">Low confidence — hold steady</span>',
                                unsafe_allow_html=True)

            token = id(proc)
            last_seen_key = f"last_committed_id_{token}"
            if proc.committed_label is not None:
                now = time.time()
                if now - st.session_state.get(last_seen_key, 0) > 0.05:
                    st.session_state[last_seen_key] = now
                    # Capture old sentence text to see if a word boundary was crossed
                    old_text = st.session_state["text_buffer"].get_final_text()
                    st.session_state["text_buffer"].handle_label(proc.committed_label)
                    new_text = st.session_state["text_buffer"].get_final_text()
                    
                    # If sentence changed, update translation
                    if new_text and new_text != old_text:
                        out_lang = st.session_state.get("output_language", DEFAULT_OUTPUT_LANGUAGE)
                        if out_lang != DEFAULT_OUTPUT_LANGUAGE:
                            tgt = SUPPORTED_OUTPUT_LANGUAGES[out_lang]
                            res = translate(new_text, tgt)
                            st.session_state["translated_text"] = res["translated"]
                            st.session_state["translation_error"] = res["error"]
                        else:
                            st.session_state["translated_text"] = ""
                            st.session_state["translation_error"] = None

    with col_right:
        buf = st.session_state["text_buffer"]

        # SOS Banner
        if buf.consume_sos():
            st.error("🚨 **I NEED HELP** 🚨", icon="🚨")

        # ── Text card ──────────────────────────────────────────────────
        st.markdown('<p class="label-sm">Recognized text</p>', unsafe_allow_html=True)
        display_text = buf.sentence or ""
        cursor_html = '<span class="cursor"></span>'
        
        # Translation UI
        trans_html = ""
        if st.session_state.get("output_language", DEFAULT_OUTPUT_LANGUAGE) != DEFAULT_OUTPUT_LANGUAGE:
            trans_text = st.session_state.get("translated_text", "")
            err = st.session_state.get("translation_error")
            if err:
                trans_html = f'<div style="color:#ff6b6b;font-size:0.9rem;margin-top:8px;">⚠️ {err}</div>'
            elif trans_text:
                trans_html = f'<div style="color:#81d4fa;font-size:1.5rem;margin-top:8px;font-weight:600;">{trans_text}</div>'

        st.markdown(
            f'<div class="text-card">'
            f'<span class="big-text">{display_text or "&nbsp;"}{cursor_html}</span>'
            f'{trans_html}</div>',
            unsafe_allow_html=True,
        )

        # Autocorrect feedback
        if buf.last_correction:
            st.caption(f"✏️ Auto-corrected: {buf.last_correction}")

        # ── Phrase Shortcuts ────────────────────────────────────────────
        st.markdown('<p class="label-sm" style="margin-top:16px;">Phrases</p>', unsafe_allow_html=True)
        phrase_cols = st.columns(3)
        phrases_list = list(PHRASE_LABELS.items())
        for i, (p_label, p_text) in enumerate(phrases_list[:6]):
            with phrase_cols[i % 3]:
                # Highlight if it was the last phrase committed
                is_active = (buf.last_phrase == p_label)
                btn_type = "primary" if is_active else "secondary"
                if st.button(p_text, key=f"phrase_{p_label}", type=btn_type, use_container_width=True):
                    buf.handle_label(p_label)
                    st.rerun()

        # ── Suggestion chips ────────────────────────────────────────────
        if buf.suggestions:
            st.markdown('<p class="label-sm" style="margin-top:16px;">Did you mean...</p>', unsafe_allow_html=True)
            sugg_cols = st.columns(min(len(buf.suggestions), 4))
            for i, suggestion in enumerate(buf.suggestions[:4]):
                with sugg_cols[i]:
                    if st.button(suggestion, key=f"sugg_{suggestion}",
                                 use_container_width=True):
                        buf.current_word = suggestion
                        buf.handle_label("SPACE")
                        st.rerun()

        st.write("") # spacer

        # ── Action buttons ──────────────────────────────────────────────

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("🔊 Speak", use_container_width=True):
                text = buf.get_final_text()
                _speak_text(text)
                _add_to_history(text)
                st.toast(f"Speaking: {text}")
        with b2:
            if st.button("⌫ Backspace", use_container_width=True):
                buf.handle_label("BACKSPACE")
                st.rerun()
        with b3:
            if st.button("🗑️ Clear", use_container_width=True):
                buf.clear()
                st.rerun()

        # Auto-speak via SPEAK gesture
        if buf.consume_speak_request():
            text = buf.get_final_text()
            _speak_text(text)
            _add_to_history(text)
            st.toast(f"Speaking: {text}")

        st.divider()

        # ── Session history ─────────────────────────────────────────────
        st.markdown('<p class="label-sm">Session history</p>', unsafe_allow_html=True)
        history = st.session_state.get("history", [])
        if not history:
            st.caption("No sentences yet.")
        else:
            for entry in reversed(history[-8:]):
                text_val = entry["text"] if isinstance(entry, dict) else entry
                ts_val = entry.get("ts", "") if isinstance(entry, dict) else ""
                st.markdown(
                    f'<div class="hist-row">'
                    f'<span class="hist-text">{text_val}</span>'
                    f'<span class="hist-time">{ts_val}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Learn mode
# ---------------------------------------------------------------------------
def render_learn_mode():
    st.header("📚 Learn — Guided Sign Language Lessons")
    language = st.session_state["language"]
    username = st.session_state["username"] or "guest"
    store = get_progress_store()

    lesson_menu = build_lesson_menu(language, username, store)
    overall_acc = store.overall_accuracy(username, language)

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.metric("Overall accuracy", f"{overall_acc}%")
    with col_meta2:
        completed = sum(1 for l in lesson_menu if l.get("completed") and l["id"] != "practice_mistakes")
        total = sum(1 for l in lesson_menu if l["id"] != "practice_mistakes")
        st.metric("Lessons done", f"{completed}/{total}")

    if not lesson_menu:
        st.info("No lessons available for this language yet.")
        return

    lesson_titles = [
        f"{'✅ ' if l['completed'] else ''}{l['title']}" for l in lesson_menu
    ]
    choice_idx = st.selectbox("Choose a lesson", range(len(lesson_menu)),
                              format_func=lambda i: lesson_titles[i])
    chosen = lesson_menu[choice_idx]
    st.caption(chosen["description"])

    if st.button("▶️ Start / Restart lesson"):
        st.session_state["lesson_session"] = LessonSession(
            language=language, lesson_id=chosen["id"],
            username=username, progress_store=store,
        )

    session: Optional[LessonSession] = st.session_state.get("lesson_session")

    lang = st.session_state["language"]
    mode = st.session_state["mode"]
    conf = st.session_state.get("conf_thresh", 0.65)
    hold = st.session_state.get("hold_frames", 12)
    current_lesson = session
    
    def _make_processor_learn():
        p = SignProcessor(language=lang, mode=mode, lesson_session=current_lesson)
        if p.predictor:
            p.predictor.confidence_threshold = conf
            p.predictor.hold_frames_to_commit = hold
        return p

    col_video, col_status = st.columns([3, 2])
    with col_video:
        ctx = webrtc_streamer(
            key="signbridge-learn",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=_make_processor_learn,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_status:
        if session is None or session.lesson_id != chosen["id"]:
            st.info("Press **Start / Restart lesson** to begin.")
            return

        if session.finished:
            summary = session.summary()
            st.success(
                f"🎉 Lesson complete! "
                f"Score: {summary['correct']}/{summary['total']} ({summary['score_pct']}%)"
            )
            return

        # ── Progress indicator ──────────────────────────────────────────
        st.progress(session.progress_fraction,
                    text=f"{session.current_index}/{len(session.targets)} signs done")

        target = session.current_target
        if target:
            st.markdown(f"### Sign this: **{target}**")

            # Reference description (simple text — no image assets yet)
            from src.learning_content import get_lesson_by_id
            try:
                lesson_data = get_lesson_by_id(language, chosen["id"])
                descriptions = lesson_data.get("descriptions", {})
                if target in descriptions:
                    st.info(f"💡 {descriptions[target]}")
            except Exception:
                pass

        # ── Live camera feedback ────────────────────────────────────────
        if ctx.state.playing and ctx.video_processor:
            proc: SignProcessor = ctx.video_processor
            if proc.smoothed_label:
                st.caption(f"Detecting: {proc.smoothed_label} ({int(proc.confidence * 100)}%)")
            if proc.committed_label and session:
                now = time.time()
                debounce_key = "learn_last_commit_time"
                if now - st.session_state.get(debounce_key, 0) > 0.3:
                    st.session_state[debounce_key] = now
                    correct = session.submit_prediction(proc.committed_label)
                    st.session_state["last_feedback"] = (
                        f"✅ Correct: **{proc.committed_label}**" if correct
                        else f"❌ Try again — detected **{proc.committed_label}**"
                    )

        if st.session_state.get("last_feedback"):
            if "✅" in st.session_state["last_feedback"]:
                st.success(st.session_state["last_feedback"])
            else:
                st.error(st.session_state["last_feedback"])


# ---------------------------------------------------------------------------
# Reply mode
# ---------------------------------------------------------------------------
def render_reply_mode():
    st.header("💬 Reply Mode")
    st.write("Type or speak a message to translate it into a sign language sequence.")
    
    col_input, col_mic = st.columns([8, 1])
    with col_input:
        reply_text = st.text_input("Message", placeholder="Type a message...", label_visibility="collapsed")
    
    with col_mic:
        mic_clicked = st.button("🎤", help="Speak to type")
        
    if mic_clicked:
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Listening...")
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
            try:
                spoken = r.recognize_google(audio)
                st.session_state["reply_text_buf"] = spoken
                st.success(f"Heard: {spoken}")
                st.rerun()
            except sr.UnknownValueError:
                st.error("Could not understand audio")
            except sr.RequestError as e:
                st.error(f"Speech service unavailable: {e}")
        except ImportError:
            st.warning("SpeechRecognition not installed. Falling back to typing.")
        except Exception as e:
            st.error(f"Mic error: {e}")

    # Use the session buffer if set by mic, else use text input
    if "reply_text_buf" in st.session_state and not reply_text:
        reply_text = st.session_state["reply_text_buf"]
    elif reply_text:
        st.session_state["reply_text_buf"] = reply_text
        
    if not reply_text:
        st.info("Enter text to see the sign sequence.")
        return
        
    lang = st.session_state["language"]
    seq = text_to_sign_sequence(reply_text, lang)
    
    if not seq:
        return
        
    # Sequence Player state
    if "reply_idx" not in st.session_state:
        st.session_state["reply_idx"] = 0
    if "reply_playing" not in st.session_state:
        st.session_state["reply_playing"] = False
        
    # Controls
    c_prev, c_play, c_next, c_speed = st.columns([1, 1, 1, 3])
    with c_prev:
        if st.button("⏮ Prev") and st.session_state["reply_idx"] > 0:
            st.session_state["reply_idx"] -= 1
            st.session_state["reply_playing"] = False
    with c_play:
        play_label = "⏸ Pause" if st.session_state["reply_playing"] else "▶️ Play"
        if st.button(play_label):
            st.session_state["reply_playing"] = not st.session_state["reply_playing"]
            # Reset index if playing from the end
            if st.session_state["reply_playing"] and st.session_state["reply_idx"] >= len(seq) - 1:
                st.session_state["reply_idx"] = 0
            st.rerun()
    with c_next:
        if st.button("Next ⏭") and st.session_state["reply_idx"] < len(seq) - 1:
            st.session_state["reply_idx"] += 1
            st.session_state["reply_playing"] = False
    with c_speed:
        speed = st.slider("Speed (seconds per sign)", 0.2, 2.0, 1.0, 0.1)
        
    idx = st.session_state["reply_idx"]
    st.progress((idx + 1) / len(seq), text=f"Sign {idx + 1} of {len(seq)}")
    
    # Render current frame
    frame = seq[idx]
    st.markdown('<div style="display:flex;justify-content:center;margin-top:20px;">', unsafe_allow_html=True)
    if frame["type"] == "space":
        st.markdown('<div style="width:200px;height:200px;border:2px dashed #444;display:flex;align-items:center;justify-content:center;"><span style="font-size:2rem;color:#666;">(Space)</span></div>', unsafe_allow_html=True)
    elif frame["type"] == "image":
        st.image(frame["path"], caption=frame["label"], width=300)
    else: # placeholder
        st.markdown(f'<div style="width:200px;height:200px;background:#333;display:flex;align-items:center;justify-content:center;border-radius:10px;"><span style="font-size:4rem;color:#eee;">{frame["label"]}</span></div>', unsafe_allow_html=True)
        st.caption("No image available for this character.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-play loop
    if st.session_state["reply_playing"]:
        if idx < len(seq) - 1:
            time.sleep(speed)
            st.session_state["reply_idx"] += 1
            st.rerun()
        else:
            st.session_state["reply_playing"] = False
            st.rerun()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if st.session_state["mode"] == "Translate":
    render_translate_mode()
elif st.session_state["mode"] == "Learn":
    render_learn_mode()
elif st.session_state["mode"] == "Reply":
    render_reply_mode()

st.divider()
st.caption(
    "SignBridge — Real-time sign recognition (MediaPipe + scikit-learn), "
    "auto-corrected text, control gestures, and interactive learning."
)
