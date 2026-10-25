# app.py
# ------
# SignBridge Streamlit application — dark-theme UI matching the design mockup.
#
# Two modes selectable via sidebar toggle:
#   1. Translate  - real-time sign -> text -> speech
#   2. Learn      - guided lessons with instant feedback and progress tracking
#
# Run:
#     streamlit run app.py

import streamlit as st

st.set_page_config(page_title="SignBridge", page_icon="🤟", layout="wide")

import os
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional

import av
import cv2
import numpy as np
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from src.auto_typer import AutoTyper
from src.config import (
    DEFAULT_LANGUAGE, DEFAULT_OUTPUT_LANGUAGE, PHRASE_LABELS, PHRASE_GESTURES,
    PROGRESS_FILE, SOS_PHRASE_LABEL, SUPPORTED_LANGUAGES, SUPPORTED_OUTPUT_LANGUAGES,
    THEME_NAVY, THEME_NAVY_CARD, THEME_NAVY_BORDER, THEME_PURPLE, THEME_ORANGE,
    THEME_BLUE, THEME_CYAN, THEME_GREEN, THEME_RED, THEME_TEXT_PRIMARY, THEME_TEXT_MUTED,
    PHRASE_ICONS, HOLD_SECONDS, MIN_CONF, REPEAT_DELAY_SECONDS,
)
from src.landmark_utils import HandLandmarkExtractor, extract_feature_vector
from src.learning_mode import LessonSession, build_lesson_menu
from src.predictor import GesturePredictor, ModelNotFoundError
from src.sign_image_mapper import text_to_sign_sequence
from src.text_buffer import TextBuffer
from src.translator import translate
from src.tts_engine import TTSEngine
from src.ui_helpers import (
    build_analyzing_dots,
    build_hand_tracking_pill,
    build_stream_status_pill,
    format_phrase_chip_label,
    is_phrase_calibrated,
    render_html_pill,
)
from utils.progress_store import ProgressStore
from utils.user_profiles import UserProfileManager, sanitize_username, validate_username

# ---------------------------------------------------------------------------
# Page config + global CSS (dark theme, large text, high contrast)
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Dark base & Global layout ────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 50% 0%, #151a2d 0%, #0b0f19 100%) !important;
    color: #e2e8f0;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1440px !important;
}

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0d111d !important;
    border-right: 1px solid #1a233a !important;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1;
}

/* ── Header Bar ───────────────────────────────────────────── */
.sb-header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0 16px 0;
    border-bottom: 1px solid #1b233a;
    margin-bottom: 18px;
}
.sb-header-left {
    display: flex;
    align-items: center;
    gap: 14px;
}
.sb-header-logo-badge {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 0 16px rgba(139, 92, 246, 0.4);
}
.sb-header-title {
    font-size: 1.65rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.sb-header-tagline {
    font-size: 0.78rem;
    color: #94a3b8;
    letter-spacing: 0.04em;
    margin-top: 2px;
}

/* ── Sidebar Cards & Elements ─────────────────────────────── */
.profile-card {
    background: #14192b;
    border: 1px solid #222b44;
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.profile-avatar-circle {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    color: #ffffff;
    font-size: 0.95rem;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.4);
    flex-shrink: 0;
}
.sb-divider {
    height: 1px;
    background: #1a233a;
    margin: 14px 0;
}

/* ── Typography & Headings ────────────────────────────────── */
.label-header {
    font-size: 0.82rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
}
.big-text {
    font-size: 2.2rem !important;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.02em;
    line-height: 1.3;
}
.blinking-cursor {
    display: inline-block;
    width: 3px;
    height: 1.3em;
    background: #38bdf8;
    animation: blink 1s step-end infinite;
    vertical-align: text-bottom;
    margin-left: 4px;
    box-shadow: 0 0 8px #38bdf8;
}
@keyframes blink { 50% { opacity: 0; } }

/* ── Mode & Language Buttons ──────────────────────────────── */
.mode-btn-container button[kind="primary"],
.input-lang-container button[kind="primary"] {
    background: linear-gradient(135deg, #f97316 0%, #ec4899 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(249, 115, 22, 0.4) !important;
}
.mode-btn-container button[kind="secondary"],
.input-lang-container button[kind="secondary"] {
    background: #161c2d !important;
    border: 1px solid #252e47 !important;
    color: #94a3b8 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}
.mode-btn-container button[kind="secondary"]:hover,
.input-lang-container button[kind="secondary"]:hover {
    background: #1f273d !important;
    color: #ffffff !important;
    border-color: #3b4870 !important;
}

/* ── Camera Card ──────────────────────────────────────────── */
.cam-card-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
div[data-testid="stWebRtcStreamer"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1.5px solid rgba(139, 92, 246, 0.3);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4), 0 0 20px rgba(139, 92, 246, 0.12);
}
div[data-testid="stWebRtcStreamer"] video {
    border-radius: 14px !important;
}
div[data-testid="stWebRtcStreamer"] button {
    background: linear-gradient(135deg, #ef4444 0%, #f43f5e 100%) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 8px 24px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35) !important;
}
.status-strip-box {
    display: inline-flex;
    align-items: center;
    padding: 8px 14px;
    background: #131728;
    border: 1px solid #232c48;
    border-radius: 12px;
    font-size: 0.85rem;
    margin-top: 8px;
}
.gesture-trans-card {
    background: linear-gradient(135deg, rgba(249, 115, 22, 0.12) 0%, rgba(139, 92, 246, 0.15) 50%, rgba(56, 189, 248, 0.12) 100%);
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-radius: 16px;
    padding: 16px 20px;
    margin-top: 14px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

/* ── Recognized Text Card ─────────────────────────────────── */
.text-display-card {
    background: #090c15;
    border: 1.5px solid rgba(139, 92, 246, 0.45);
    border-radius: 16px;
    padding: 20px 24px;
    min-height: 120px;
    box-shadow: 0 0 25px rgba(139, 92, 246, 0.12);
}

/* ── Phrases Grid ─────────────────────────────────────────── */
div:has(> .phrases-grid-marker) + div button,
div:has(> .phrases-grid-marker) + div + div button {
    background: #141a2e !important;
    border: 1px solid #232d4b !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
    min-height: 58px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    white-space: pre-line !important;
}
div:has(> .phrases-grid-marker) + div button:hover,
div:has(> .phrases-grid-marker) + div + div button:hover {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 12px rgba(139, 92, 246, 0.25) !important;
}
/* Active phrase highlight */
div:has(> .phrases-grid-marker) + div button[kind="primary"],
div:has(> .phrases-grid-marker) + div + div button[kind="primary"] {
    background: linear-gradient(135deg, #f97316 0%, #a855f7 100%) !important;
    border: 1.5px solid #f97316 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 18px rgba(249, 115, 22, 0.4) !important;
}
/* SOS Phrase (1st column in row 1) highlight */
div:has(> .phrases-grid-marker) + div [data-testid="column"]:nth-child(1) button[kind="secondary"] {
    background: linear-gradient(135deg, rgba(249, 115, 22, 0.25) 0%, rgba(168, 85, 247, 0.25) 100%) !important;
    border: 1.5px solid rgba(249, 115, 22, 0.6) !important;
    color: #ffedd5 !important;
}

/* ── Suggestion Chips ─────────────────────────────────────── */
div:has(> .sugg-chips-marker) + div button {
    background: #1b2640 !important;
    border: 1px solid #33446b !important;
    color: #7ec8f7 !important;
    border-radius: 20px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 4px 14px !important;
}

/* ── Quick Actions ────────────────────────────────────────── */
div:has(> .quick-actions-marker) + div [data-testid="column"]:nth-child(1) button {
    background: linear-gradient(135deg, #2563eb 0%, #38bdf8 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    min-height: 48px !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
}
div:has(> .quick-actions-marker) + div [data-testid="column"]:nth-child(2) button,
div:has(> .quick-actions-marker) + div [data-testid="column"]:nth-child(3) button {
    background: #141a2e !important;
    border: 1px solid #252e47 !important;
    color: #cbd5e1 !important;
    border-radius: 12px !important;
    min-height: 48px !important;
    font-weight: 600 !important;
}
div:has(> .quick-actions-marker) + div [data-testid="column"] button:hover {
    transform: translateY(-1px);
}

/* ── History & Alerts ─────────────────────────────────────── */
.hist-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #1a2238;
}
.hist-text {
    font-size: 0.95rem;
    color: #cbd5e1;
    font-weight: 500;
}
.hist-time {
    font-size: 0.75rem;
    color: #64748b;
}
.synth-banner {
    background: rgba(245, 158, 11, 0.1);
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 10px 16px;
    color: #fde68a;
    font-size: 0.875rem;
    margin-bottom: 12px;
}
.sos-banner {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
    color: #ffffff;
    font-size: 1.5rem;
    font-weight: 800;
    text-align: center;
    padding: 14px 20px;
    border-radius: 12px;
    margin-bottom: 16px;
    box-shadow: 0 0 25px rgba(220, 38, 38, 0.6);
    letter-spacing: 0.05em;
    animation: sosPulse 1.5s infinite;
}
@keyframes sosPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.92; transform: scale(0.995); }
}
</style>
""", unsafe_allow_html=True)

def _get_rtc_configuration() -> RTCConfiguration:
    """Return WebRTC configuration with TURN support if secrets are configured, or Google STUN fallback."""
    secrets = dict(os.environ)
    try:
        secrets.update(dict(st.secrets))
    except Exception:
        pass

    # 1. Direct Metered credentials (username & password)
    try:
        if "METERED_USERNAME" in secrets and "METERED_PASSWORD" in secrets:
            u = str(secrets["METERED_USERNAME"]).strip()
            p = str(secrets["METERED_PASSWORD"]).strip()
            return RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.relay.metered.ca:80"]},
                    {
                        "urls": [
                            "turn:global.relay.metered.ca:80",
                            "turn:global.relay.metered.ca:443",
                            "turn:global.relay.metered.ca:443?transport=tcp",
                        ],
                        "username": u,
                        "credential": p,
                    },
                ]
            })
    except Exception:
        pass

    # 1. Direct ice_servers list in st.secrets
    try:
        if "ice_servers" in secrets:
            return RTCConfiguration({"iceServers": secrets["ice_servers"]})
    except Exception:
        pass

    # 2. Metered.ca TURN (Free 50 GB/month)
    try:
        if "METERED_API_KEY" in secrets:
            import urllib.request, json
            app_name = str(secrets.get("METERED_APP_NAME", "")).strip()
            api_key = str(secrets["METERED_API_KEY"]).strip()
            url = (
                f"https://{app_name}.metered.live/api/v1/turn/credentials?apiKey={api_key}"
                if app_name
                else f"https://metered.live/api/v1/turn/credentials?apiKey={api_key}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "SignBridge"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                servers = json.loads(resp.read().decode())
                if isinstance(servers, list) and servers:
                    return RTCConfiguration({"iceServers": servers})
    except Exception as exc:
        logging.warning("Metered TURN credentials fetch failed: %s", exc)

    # 3. Twilio TURN
    try:
        if "TWILIO_ACCOUNT_SID" in secrets and "TWILIO_AUTH_TOKEN" in secrets:
            from twilio.rest import Client
            client = Client(secrets["TWILIO_ACCOUNT_SID"], secrets["TWILIO_AUTH_TOKEN"])
            token = client.tokens.create()
            if token and token.ice_servers:
                return RTCConfiguration({"iceServers": token.ice_servers})
    except Exception as exc:
        logging.warning("Twilio TURN credentials fetch failed: %s", exc)

    # 4. Fallback: Google public STUN
    return RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )


RTC_CONFIGURATION = _get_rtc_configuration()


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

    All mutable state that the main thread reads is protected by _lock and
    exposed only through get_state().  The main thread must never read instance
    attributes directly; use proc.get_state() instead to avoid torn reads.

    Frame pipeline:
        av.VideoFrame  →  to_ndarray(bgr24)  →  extract_feature_vector()
        (mirror + process_raw + process_from_raw, all in one call)
        →  draw()  →  _draw_overlay()  →  av.VideoFrame

    extract_feature_vector is the single canonical function shared by
    data_collection.py, cli_app.py and app.py so all three paths apply the
    same mirror + BGR→RGB + slot-ordering logic.
    """

    # Gate verbose per-frame debug logging.  Set to True to re-enable.
    DEBUG_RECV = False

    def __init__(self, language: str, mode: str,
                 lesson_session: Optional[LessonSession] = None):
        self.language = language
        self.mode = mode
        self.lesson_session = lesson_session
        self.extractor = HandLandmarkExtractor()
        try:
            self.predictor: Optional[GesturePredictor] = GesturePredictor(language)
        except ModelNotFoundError:
            self.predictor = None

        # ── Lock-protected state (written by background thread, read by main thread) ──
        self._lock = threading.Lock()
        self._committed_queue: deque = deque()
        needs_cal = (
            set(self.predictor.metadata.get("needs_calibration", []))
            if self.predictor and self.predictor.metadata
            else set()
        )
        self.auto_typer = AutoTyper(
            hold_seconds=HOLD_SECONDS,
            min_confidence=MIN_CONF,
            repeat_delay_seconds=REPEAT_DELAY_SECONDS,
            needs_calibration=needs_cal,
        )
        self._state: dict = {
            "smoothed_label": None,
            "confidence": 0.0,
            "progress": 0.0,
            "committed_label": None,
            "hand_detected": False,
            "handedness_str": "",
        }

    # ── Convenience read-only properties for callers that still use dot-access ──
    # These take the lock so reads are always consistent.
    @property
    def smoothed_label(self):
        with self._lock:
            return self._state["smoothed_label"]

    @property
    def confidence(self):
        with self._lock:
            return self._state["confidence"]

    @property
    def progress(self):
        with self._lock:
            return self._state["progress"]

    @property
    def committed_label(self):
        with self._lock:
            return self._state["committed_label"]

    @property
    def hand_detected(self):
        with self._lock:
            return self._state["hand_detected"]

    @property
    def handedness_str(self):
        with self._lock:
            return self._state["handedness_str"]

    def get_state(self) -> dict:
        """Return an atomic snapshot of processor state for the main thread."""
        with self._lock:
            return dict(self._state)

    def _write_state(self, **kwargs):
        """Write one or more state fields atomically from the background thread."""
        with self._lock:
            self._state.update(kwargs)

    def pop_committed_labels(self) -> list:
        """Pop all committed labels queued by auto_typer since last check."""
        with self._lock:
            items = list(self._committed_queue)
            self._committed_queue.clear()
            return items

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        import traceback, logging, os

        # ── Debug log setup ─────────────────────────────────────────────────
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
            img = np.ascontiguousarray(img)
            if not img.flags.writeable:
                img = img.copy()

            # ── Canonical pipeline (mirror=True matches data_collection.py) ─
            # extract_feature_vector handles: mirror flip, BGR→RGB, process_raw,
            # process_from_raw.  The raw MediaPipe result is returned alongside
            # the FrameResult so we can draw the skeleton separately.
            frame_result = extract_feature_vector(img, self.extractor, mirror=True)
            feature_vector = frame_result.feature_vector if frame_result else None

            hand_detected = feature_vector is not None

            # Rebuild handedness string before taking the lock
            handedness_str = ""
            if frame_result and frame_result.hands:
                labels = [h.handedness for h in frame_result.hands if h.handedness not in ("", "Unknown")]
                if len(labels) >= 2:
                    handedness_str = "Left + right"
                elif labels:
                    handedness_str = labels[0]

            # ── Draw skeleton on mirrored copy ───────────────────────────────
            # We need the raw mp_result for drawing.  Re-run process_raw on the
            # already-mirrored frame (extract_feature_vector has already flipped
            # it internally, so we call process_raw on the *mirrored* version).
            img_mirrored = np.ascontiguousarray(img[:, ::-1, :])
            mp_raw_for_draw = self.extractor.process_raw(img_mirrored)
            img_draw = self.extractor.draw(img_mirrored, mp_raw_for_draw)

            if self.predictor is None:
                self._write_state(
                    hand_detected=hand_detected,
                    handedness_str=handedness_str,
                )
                out_frame = av.VideoFrame.from_ndarray(img_draw, format="bgr24")
                out_frame.pts = frame.pts
                out_frame.time_base = frame.time_base
                return out_frame

            result = self.predictor.update(feature_vector)

            # Auto-typer integration: processes predictions with 0.8s hold and repeat lockout
            auto_res = self.auto_typer.process(
                label=result["smoothed_label"],
                confidence=result["confidence"],
            )
            if auto_res["committed_label"]:
                with self._lock:
                    self._committed_queue.append(auto_res["committed_label"])

            self._write_state(
                smoothed_label=result["smoothed_label"],
                confidence=result["confidence"],
                progress=auto_res["progress"],
                committed_label=auto_res["committed_label"],
                hand_detected=hand_detected,
                handedness_str=handedness_str,
            )

            self._draw_overlay(img_draw)
            out_frame = av.VideoFrame.from_ndarray(img_draw, format="bgr24")
            out_frame.pts = frame.pts
            out_frame.time_base = frame.time_base
            return out_frame

        except Exception:
            tb = traceback.format_exc()
            print(f"[SignProcessor.recv EXCEPTION] {tb}")
            return frame

    def _update_handedness(self, mp_raw):
        if mp_raw is None or not hasattr(mp_raw, "multi_handedness") \
                or not mp_raw.multi_handedness:
            return ""
        labels = [h.classification[0].label for h in mp_raw.multi_handedness]
        if len(labels) == 2:
            return "Left + right"
        elif labels:
            return labels[0]
        return ""


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
    # ── Profile card ───────────────────────────────────────────────
    pm = get_profile_manager()
    username = st.session_state["username"]
    initials = "".join(w[0].upper() for w in username.split()[:2]) or "?"
    st.markdown(
        f'<div class="profile-card">'
        f'<div class="profile-avatar-circle">{initials}</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-weight:700;font-size:1.05rem;color:#ffffff;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{username}</div>'
        f'<div style="font-size:0.75rem;color:#94a3b8;">Switch profile ↓</div>'
        f'</div>'
        f'</div>',
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

    # ── Select Sign Language ───────────────────────────────────────
    st.markdown('<div class="label-header">🔤 Select Sign Language</div>', unsafe_allow_html=True)
    lang_options = SUPPORTED_LANGUAGES  # ["ASL", "ISL"]
    curr_lang_idx = lang_options.index(st.session_state["language"]) if st.session_state["language"] in lang_options else 0
    selected_lang = st.selectbox(
        "Sign Language",
        lang_options,
        index=curr_lang_idx,
        format_func=lambda l: "✋ ASL (American Sign Language)" if l == "ASL" else "👐 ISL (Indian Sign Language)",
        label_visibility="collapsed",
    )
    if selected_lang != st.session_state["language"]:
        st.session_state["language"] = selected_lang
        st.session_state["lesson_session"] = None
        st.rerun()

    # ── Safety Zone / Danger Zone ──────────────────────────────────
    with st.expander("⚠️ Safety Zone"):
        st.markdown('<div style="font-size:0.8rem;color:#f87171;font-weight:600;margin-bottom:6px;">Danger Zone</div>', unsafe_allow_html=True)
        if st.button("🗑️ Delete profile", type="secondary", use_container_width=True):
            pm.delete_user(st.session_state["username"])
            st.session_state["username"] = "guest"
            st.session_state["lesson_session"] = None
            st.session_state["text_buffer"] = TextBuffer(auto_correct=True)
            st.rerun()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Mode Selector ──────────────────────────────────────────────
    st.markdown('<div class="label-header">⚡ Mode</div><div class="mode-btn-container">', unsafe_allow_html=True)
    col_t, col_l, col_r = st.columns(3)
    with col_t:
        if st.button("🌐 Translate",
                     type="primary" if st.session_state["mode"] == "Translate" else "secondary",
                     use_container_width=True):
            st.session_state["mode"] = "Translate"
            st.rerun()
    with col_l:
        if st.button("📖 Learn",
                     type="primary" if st.session_state["mode"] == "Learn" else "secondary",
                     use_container_width=True):
            st.session_state["mode"] = "Learn"
            st.rerun()
    with col_r:
        if st.button("💬 Reply",
                     type="primary" if st.session_state["mode"] == "Reply" else "secondary",
                     use_container_width=True):
            st.session_state["mode"] = "Reply"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Input Language Toggle ──────────────────────────────────────
    st.markdown('<div class="label-header" style="margin-top:14px;">🌐 Input language</div><div class="input-lang-container">', unsafe_allow_html=True)
    col_isl, col_asl = st.columns(2)
    with col_isl:
        if st.button(lang_options[1] if len(lang_options) > 1 else "ISL",
                     type="primary" if st.session_state["language"] == (lang_options[1] if len(lang_options) > 1 else "ISL") else "secondary",
                     use_container_width=True):
            st.session_state["language"] = lang_options[1] if len(lang_options) > 1 else "ISL"
            st.session_state["lesson_session"] = None
            st.rerun()
    with col_asl:
        if st.button(lang_options[0],
                     type="primary" if st.session_state["language"] == lang_options[0] else "secondary",
                     use_container_width=True):
            st.session_state["language"] = lang_options[0]
            st.session_state["lesson_session"] = None
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Output Language Toggle ─────────────────────────────────────
    st.markdown('<div class="label-header" style="margin-top:14px;">🗣️ Output language</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Sliders & Settings ─────────────────────────────────────────
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

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Model status + banners ─────────────────────────────────────
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
        needs_cal = pred.metadata.get("needs_calibration", []) if pred.metadata else []
        acc_real = pred.metadata.get("validation_accuracy_real_labels_only") if pred.metadata else None

        if is_synth and source not in ("public_dataset", "webcam", "mixed"):
            # Fully synthetic model (e.g. ISL placeholder)
            st.markdown(
                f'<div class="synth-banner">⚠️ <b>Prototype Mode ({st.session_state["language"]})</b> — '
                f'Model trained entirely on synthetic dummy data (100% accuracy on noise, 0% on real hands). '
                f'Live recognition is blocked until real data is collected and retrained.</div>',
                unsafe_allow_html=True,
            )
        elif needs_cal:
            # Calibrated model with missing gestures
            acc_str = f" | Real-label val-accuracy: {acc_real*100:.1f}%" if acc_real is not None else ""
            st.warning(
                f"⚠️ **Calibrated Model** ({source}){acc_str}. "
                f"**{len(needs_cal)} gesture(s) need calibration** (no real data collected yet): "
                f"`{'`, `'.join(needs_cal)}`"
            )
        elif source == "public_dataset" or source == "mixed":
            acc_str = f" | Val-accuracy on real labels: {acc_real*100:.1f}%" if acc_real is not None else ""
            st.info(
                f"ℹ️ **Public Dataset Model**{acc_str}: Trained on imported Kaggle data. "
                "Performance depends on how closely your webcam setup matches the dataset images."
            )
        else:
            st.success(f"Model loaded for {st.session_state['language']} ✅ (Trained on your real webcam data!)")

    # ── Sidebar Footer Wave ────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:20px 0 8px 0;font-size:0.75rem;color:#64748b;letter-spacing:0.04em;">'
        '<div style="width:100%;height:1px;background:linear-gradient(90deg,transparent,#252e47,transparent);margin-bottom:12px;"></div>'
        'Sign Language for a better tomorrow ♡'
        '</div>',
        unsafe_allow_html=True,
    )


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
# Top Header Bar (Wordmark + Tagline + Live/Offline status pill)
# ---------------------------------------------------------------------------
is_stream_playing = bool(st.session_state.get("stream_playing", False))
header_pill_data = build_stream_status_pill(is_stream_playing)
header_pill_html = render_html_pill(header_pill_data)

st.markdown(
    f'<div class="sb-header-bar">'
    f'<div class="sb-header-left">'
    f'<div class="sb-header-logo-badge">🤝</div>'
    f'<div>'
    f'<div class="sb-header-title">SignBridge</div>'
    f'<div class="sb-header-tagline">Bridging Hands • Connecting Hearts</div>'
    f'</div>'
    f'</div>'
    f'<div class="sb-header-right">{header_pill_html}</div>'
    f'</div>',
    unsafe_allow_html=True,
)


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
    if pred and pred.metadata and pred.metadata.get("synthetic", True) and pred.metadata.get("data_source") not in ("public_dataset", "webcam", "mixed"):
        st.error(
            f"🛑 **{lang} is disabled for live recognition**\n\n"
            f"The {lang} model was trained on 100% synthetic dummy data (random Gaussian noise, 0 real hand samples) "
            "and cannot recognize physical gestures.\n\n"
            "Live recognition is blocked to avoid invalid predictions until real data is collected and retrained.\n\n"
            f"**To collect real data:**\n"
            f"```\npython -m src.data_collection --language {lang} --auto --labels ...\n```\n"
            f"**To retrain:**\n"
            f"```\npython -m src.train_model --language {lang} --split-strategy block\n```"
        )
        return

    buf = st.session_state["text_buffer"]

    # ── Full-width SOS Banner ──────────────────────────────────────
    if buf.consume_sos():
        st.markdown(
            '<div class="sos-banner">🚨 <b>I NEED HELP</b> 🚨</div>',
            unsafe_allow_html=True,
        )

    col_cam, col_right = st.columns([1.15, 1], gap="medium")

    with col_cam:
        cam_top_spot = st.empty()

        ctx = webrtc_streamer(
            key="signbridge-translate",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=_make_processor_translate,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        is_playing = bool(ctx and ctx.state.playing)
        st.session_state["stream_playing"] = is_playing

        proc: Optional[SignProcessor] = ctx.video_processor if (is_playing and ctx.video_processor) else None
        hand_detected = bool(proc and proc.hand_detected)

        # Placeholders for live camera card topbar, status strip, and gesture card
        hand_pill_data = build_hand_tracking_pill(hand_detected if is_playing else False)
        hand_pill_html = render_html_pill(hand_pill_data)

        cam_top_spot.markdown(
            f'<div class="cam-card-topbar">'
            f'<div class="label-header" style="margin-bottom:0;">📹 Live Camera</div>'
            f'<div>{hand_pill_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        cam_status_spot = st.empty()
        if not is_playing:
            cam_status_spot.markdown(
                '<div class="status-strip-box">'
                '<span style="color:#64748b;">⚪ Camera offline — press START to begin</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            cam_status_spot.markdown(
                '<div class="status-strip-box">'
                '<span style="color:#94a3b8;">⚪ Stand by... show hand in frame</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        cam_gesture_card_spot = st.empty()

    with col_right:
        # Recognized text card
        live_pill = render_html_pill({
            "label": "Live Update",
            "dot_color": THEME_GREEN,
            "text_color": THEME_GREEN,
            "bg_color": "rgba(16, 185, 129, 0.12)",
            "border_color": "rgba(16, 185, 129, 0.3)",
        })
        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
            f'<div class="label-header" style="margin-bottom:0;">✨ Recognized Text</div>'
            f'<div>{live_pill}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        text_spot = st.empty()
        display_text = buf.sentence or ""
        cursor_html = '<span class="blinking-cursor"></span>'
        trans_html = ""
        if st.session_state.get("output_language", DEFAULT_OUTPUT_LANGUAGE) != DEFAULT_OUTPUT_LANGUAGE:
            trans_text = st.session_state.get("translated_text", "")
            err = st.session_state.get("translation_error")
            if err:
                trans_html = f'<div style="color:#ef4444;font-size:0.9rem;margin-top:10px;">⚠️ {err}</div>'
            elif trans_text:
                trans_html = (
                    f'<div style="color:#38bdf8;font-size:1.35rem;margin-top:10px;font-weight:600;'
                    f'border-top:1px solid #1f273f;padding-top:8px;">'
                    f'<span style="font-size:0.8rem;color:#94a3b8;font-weight:400;display:block;">'
                    f'{st.session_state.get("output_language")} translation:</span>{trans_text}</div>'
                )

        text_spot.markdown(
            f'<div class="text-display-card">'
            f'<span class="big-text">{display_text or "&nbsp;"}{cursor_html}</span>'
            f'{trans_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        correction_spot = st.empty()
        if buf.last_correction:
            correction_spot.caption(f"✏️ Auto-corrected: {buf.last_correction}")

        # Phrases Grid
        st.markdown(
            '<div class="label-header" style="margin-top:16px;margin-bottom:8px;">💬 Phrases</div>'
            '<div class="phrases-grid-marker"></div>',
            unsafe_allow_html=True,
        )

        p_row1 = st.columns(3)
        p_row2 = st.columns(3)
        phrase_slots = p_row1 + p_row2

        phrases_list = list(PHRASE_LABELS.items())
        for i, (p_label, p_text) in enumerate(phrases_list[:6]):
            with phrase_slots[i]:
                is_cal = is_phrase_calibrated(p_label, pred.metadata if pred else None)
                btn_label = format_phrase_chip_label(p_label, p_text, is_cal)
                is_active = (buf.last_phrase == p_label)
                btn_type = "primary" if is_active else "secondary"
                if st.button(btn_label, key=f"phrase_{p_label}", type=btn_type, use_container_width=True):
                    buf.handle_label(p_label)
                    st.rerun()

        # Suggestion chips
        if buf.suggestions:
            st.markdown(
                '<div class="label-header" style="margin-top:14px;margin-bottom:8px;">Did you mean...</div>'
                '<div class="sugg-chips-marker"></div>',
                unsafe_allow_html=True,
            )
            sugg_cols = st.columns(min(len(buf.suggestions), 4))
            for i, suggestion in enumerate(buf.suggestions[:4]):
                with sugg_cols[i]:
                    if st.button(suggestion, key=f"sugg_{suggestion}", use_container_width=True):
                        buf.current_word = suggestion
                        buf.handle_label("SPACE")
                        st.rerun()

        # Quick Actions
        st.markdown(
            '<div class="label-header" style="margin-top:16px;margin-bottom:8px;">⚡ Quick Actions</div>'
            '<div class="quick-actions-marker"></div>',
            unsafe_allow_html=True,
        )
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("🔊 Speak", use_container_width=True):
                text = buf.get_final_text()
                _speak_text(text)
                _add_to_history(text)
                st.toast(f"Speaking: {text}")
        with b2:
            if st.button("⌫ Backspace", use_container_width=True):
                buf.handle_label("DEL")
                st.rerun()
        with b3:
            if st.button("🗑️ Clear", use_container_width=True):
                buf.clear()
                st.rerun()

        if buf.consume_speak_request():
            text = buf.get_final_text()
            _speak_text(text)
            _add_to_history(text)
            st.toast(f"Speaking: {text}")

        # Session history
        st.markdown(
            '<div class="label-header" style="margin-top:16px;margin-bottom:8px;">📜 Session history</div>',
            unsafe_allow_html=True,
        )
        history = st.session_state.get("history", [])
        if not history:
            st.caption("No sentences recorded yet.")
        else:
            for entry in reversed(history[-5:]):
                text_val = entry["text"] if isinstance(entry, dict) else entry
                ts_val = entry.get("ts", "") if isinstance(entry, dict) else ""
                st.markdown(
                    f'<div class="hist-row">'
                    f'<span class="hist-text">{text_val}</span>'
                    f'<span class="hist-time">{ts_val}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<div style="text-align:center;padding:16px 0 4px 0;font-size:0.75rem;color:#64748b;letter-spacing:0.04em;">'
            '♡ Smaller gestures ✧ Bigger connections</div>',
            unsafe_allow_html=True,
        )

    # ── Live Polling Video Loop ─────────────────────────────────────
    if is_playing and proc:
        last_rendered_text = None
        last_rendered_hand = None
        last_rendered_status = None
        last_rendered_gesture = None

        while ctx.state.playing:
            # 1. Drain any newly committed labels from the processor
            new_labels = proc.pop_committed_labels()
            text_changed = False
            for lbl in new_labels:
                if lbl == "SPEAK":
                    t_text = buf.get_final_text()
                    if t_text.strip() and st.session_state.get("speak_aloud", True):
                        _speak_text(t_text)
                        _add_to_history(t_text)
                else:
                    buf.handle_label(lbl)
                text_changed = True

            # 2. Update translation if text changed
            if text_changed:
                out_lang = st.session_state.get("output_language", DEFAULT_OUTPUT_LANGUAGE)
                if out_lang != DEFAULT_OUTPUT_LANGUAGE:
                    tgt = SUPPORTED_OUTPUT_LANGUAGES[out_lang]
                    res = translate(buf.sentence, tgt)
                    st.session_state["translated_text"] = res["translated"]
                    st.session_state["translation_error"] = res["error"]
                else:
                    st.session_state["translated_text"] = ""
                    st.session_state["translation_error"] = None

            # 3. Read atomic snapshot from processor
            state = proc.get_state()
            h_detected = state["hand_detected"]
            s_label = state["smoothed_label"]
            conf = state["confidence"]
            prog = state["progress"]
            h_str = state["handedness_str"]

            # 4. Update status chip ONLY when hand_detected changes
            if h_detected != last_rendered_hand:
                last_rendered_hand = h_detected
                h_pill = build_hand_tracking_pill(h_detected)
                cam_top_spot.markdown(
                    f'<div class="cam-card-topbar">'
                    f'<div class="label-header" style="margin-bottom:0;">📹 Live Camera</div>'
                    f'<div>{render_html_pill(h_pill)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # 5. Update status strip ONLY when status state changes
            status_key = (prog > 0, int(prog * 8), h_detected, h_str)
            if status_key != last_rendered_status:
                last_rendered_status = status_key
                if prog > 0:
                    dots = build_analyzing_dots(prog, 12)
                    cam_status_spot.markdown(
                        f'<div class="status-strip-box">'
                        f'<span style="color:#f87171;font-weight:600;">🔴 Analyzing gestures...</span>'
                        f'<span style="color:#38bdf8;letter-spacing:2px;font-size:0.85rem;margin-left:8px;">{dots}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif h_detected:
                    h_info = f" • {h_str}" if h_str else ""
                    cam_status_spot.markdown(
                        f'<div class="status-strip-box">'
                        f'<span style="color:#10b981;font-weight:600;">🟢 Tracking...</span>'
                        f'<span style="color:#94a3b8;font-size:0.85rem;margin-left:6px;">{h_info}</span>'
                        f'<span style="color:#a855f7;margin-left:6px;font-size:0.85rem;">〰️</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    cam_status_spot.markdown(
                        '<div class="status-strip-box">'
                        '<span style="color:#94a3b8;">⚪ Stand by... show hand in frame</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

            # Update gesture card ONLY when gesture changes
            gesture_key = (h_detected, s_label, int(conf * 10) if conf else 0)
            if gesture_key != last_rendered_gesture:
                last_rendered_gesture = gesture_key
                if h_detected and s_label:
                    conf_pct = int(conf * 100)
                    hand_desc = h_str or "Hand tracked"
                    cam_gesture_card_spot.markdown(
                        f'<div class="gesture-trans-card">'
                        f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">'
                        f'<div>'
                        f'<div style="font-weight:700;color:#f8fafc;font-size:1.05rem;display:flex;align-items:center;gap:8px;">'
                        f'<span style="color:#f59e0b;">✨</span> Your gesture is being translated...</div>'
                        f'<div style="font-size:0.85rem;color:#cbd5e1;margin-top:4px;">'
                        f'Active sign: <b style="color:#38bdf8;font-size:1.2rem;">{s_label}</b> ({conf_pct}% confidence)'
                        f' • {hand_desc}</div>'
                        f'</div>'
                        f'<div>'
                        f'<svg width="105" height="34" viewBox="0 0 105 34" fill="none">'
                        f'<path d="M2 17C12 17 16 6 24 6C32 6 36 28 44 28C52 28 56 9 64 9C72 9 76 24 84 24C92 24 96 17 103 17" stroke="url(#wave_grad_live)" stroke-width="2.5" stroke-linecap="round"/>'
                        f'<defs><linearGradient id="wave_grad_live" x1="2" y1="17" x2="103" y2="17" gradientUnits="userSpaceOnUse">'
                        f'<stop stop-color="#38bdf8"/><stop offset="0.5" stop-color="#a855f7"/><stop offset="1" stop-color="#ec4899"/>'
                        f'</linearGradient></defs></svg>'
                        f'</div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    cam_gesture_card_spot.empty()

            # 6. Update Recognized Text box ONLY if text or translation changed
            current_text_state = (buf.sentence, st.session_state.get("translated_text", ""), buf.last_correction)
            if current_text_state != last_rendered_text:
                last_rendered_text = current_text_state
                display_text = buf.sentence or ""
                cursor_html = '<span class="blinking-cursor"></span>'
                trans_html = ""
                if st.session_state.get("output_language", DEFAULT_OUTPUT_LANGUAGE) != DEFAULT_OUTPUT_LANGUAGE:
                    trans_text = st.session_state.get("translated_text", "")
                    err = st.session_state.get("translation_error")
                    if err:
                        trans_html = f'<div style="color:#ef4444;font-size:0.9rem;margin-top:10px;">⚠️ {err}</div>'
                    elif trans_text:
                        trans_html = (
                            f'<div style="color:#38bdf8;font-size:1.35rem;margin-top:10px;font-weight:600;'
                            f'border-top:1px solid #1f273f;padding-top:8px;">'
                            f'<span style="font-size:0.8rem;color:#94a3b8;font-weight:400;display:block;">'
                            f'{st.session_state.get("output_language")} translation:</span>{trans_text}</div>'
                        )

                text_spot.markdown(
                    f'<div class="text-display-card">'
                    f'<span class="big-text">{display_text or "&nbsp;"}{cursor_html}</span>'
                    f'{trans_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if buf.last_correction:
                    correction_spot.caption(f"✏️ Auto-corrected: {buf.last_correction}")
                else:
                    correction_spot.empty()

            time.sleep(0.1)



# ---------------------------------------------------------------------------
# Learn mode
# ---------------------------------------------------------------------------
def render_learn_mode():
    st.header("📚 Learn — Guided Sign Language Lessons")
    language = st.session_state["language"]
    pred = get_predictor(language)
    if pred and pred.metadata and pred.metadata.get("synthetic", True) and pred.metadata.get("data_source") not in ("public_dataset", "webcam", "mixed"):
        st.error(
            f"🛑 **{language} lessons are disabled**\n\n"
            f"The {language} model was trained on 100% synthetic dummy data (random Gaussian noise, 0 real hand samples). "
            "Interactive learning feedback is blocked until real training data exists."
        )
        return

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
