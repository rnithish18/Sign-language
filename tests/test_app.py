import sys
from unittest.mock import MagicMock
mock_webrtc = MagicMock()
mock_webrtc.webrtc_streamer.return_value.video_processor = None
sys.modules["streamlit_webrtc"] = mock_webrtc


import streamlit as st
import pytest
from app import init_session_state

def test_session_state_initialization():
    """
    Tests that the centralized session-state initialization block creates all required keys.
    """
    # Clear session state if anything is lingering
    st.session_state.clear()
    
    # Run the initialization
    init_session_state()
    
    # Assert all critical keys exist
    expected_keys = [
        "text_buffer", "tts_engine", "mode", "language", "username", 
        "lesson_session", "last_committed", "last_feedback", "history", 
        "conf_thresh", "hold_frames", "speak_aloud", "output_language",
        "translated_text", "translation_error"
    ]
    
    for key in expected_keys:
        assert key in st.session_state, f"Missing key in session_state: {key}"
        
    assert st.session_state["language"] == "ASL"
    assert st.session_state["mode"] == "Translate"
    assert st.session_state["username"] == "guest"
