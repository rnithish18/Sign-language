"""
config.py
---------
Central configuration for SignBridge. Keeping every tunable constant here
means the rest of the codebase never hard-codes paths, thresholds, or
labels, and swapping languages / models only requires touching this file.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")
PROGRESS_FILE = os.path.join(DATA_DIR, "learner_progress.json")

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# MediaPipe Hands settings
# ---------------------------------------------------------------------------
MAX_NUM_HANDS = 2                 # 1 for finger-spelling; set to 2 for two-hand signs
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6
NUM_LANDMARKS = 21                # MediaPipe Hands always returns 21 landmarks/hand
# 1 presence bit + (21 * 3) features = 64 per hand. Total for 2 hands = 128
FEATURE_LENGTH = (1 + NUM_LANDMARKS * 3) * 2

# ---------------------------------------------------------------------------
# Recognition behaviour
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.65        # predictions below this are shown as "..."
PREDICTION_SMOOTHING_WINDOW = 8    # majority-vote over last N frames
HOLD_FRAMES_TO_COMMIT = 12         # frames a sign must be held before it is typed
COOLDOWN_FRAMES_AFTER_COMMIT = 10  # frames to ignore after committing a letter
HOLD_SECONDS = 0.8                 # seconds a sign must be held to auto-type
MIN_CONF = 0.80                    # minimum confidence required to accumulate hold
HAND_LOST_TIMEOUT_SECONDS = 0.5    # seconds hand must be away before repeating same letter
REPEAT_DELAY_SECONDS = 0.5         # alias for HAND_LOST_TIMEOUT_SECONDS

# ---------------------------------------------------------------------------
# Control gesture labels (must also exist as trainable classes)
# ---------------------------------------------------------------------------
GESTURE_SPACE = "SPACE"
GESTURE_DEL = "DEL"
GESTURE_BACKSPACE = "BACKSPACE"  # Alias kept for backward-compatibility with UI/tests
GESTURE_CLEAR = "CLEAR"
GESTURE_SPEAK = "SPEAK"
# BACKSPACE is dropped as a separate label because DEL has real public data and maps
# to backspace in text_buffer.py.
CONTROL_GESTURES = {GESTURE_SPACE, GESTURE_CLEAR, GESTURE_SPEAK}

# ---------------------------------------------------------------------------
# Phrase shortcuts — whole-hand gestures that commit full phrases as one unit.
# Keys must be distinct from letter signs and control gestures.
# SOS_PHRASE_LABEL is the gesture that triggers an immediate "I NEED HELP" alert.
# ---------------------------------------------------------------------------
PHRASE_LABELS: dict = {
    "PHRASE_HELP":     "I need help",
    "PHRASE_THANK_YOU": "Thank you",
    "PHRASE_WATER":    "I need water",
    "PHRASE_WASHROOM": "Where is the washroom?",
    "PHRASE_YES":      "Yes",
    "PHRASE_NO":       "No",
}
SOS_PHRASE_LABEL = "PHRASE_HELP"
PHRASE_GESTURES = set(PHRASE_LABELS.keys())

# ---------------------------------------------------------------------------
# Regional output languages
# ---------------------------------------------------------------------------
SUPPORTED_OUTPUT_LANGUAGES = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
}
DEFAULT_OUTPUT_LANGUAGE = "English"

# ---------------------------------------------------------------------------
# Supported sign languages and their default alphabets
# (Learning-mode lesson content lives in learning_content.py)
# ---------------------------------------------------------------------------
LANGUAGE_ASL = "ASL"
LANGUAGE_ISL = "ISL"
SUPPORTED_LANGUAGES = [LANGUAGE_ASL, LANGUAGE_ISL]

ASL_ALPHABET = [chr(c) for c in range(ord("A"), ord("Z") + 1)]  # A-Z (no J/Z motion handled statically)
ISL_ALPHABET = [chr(c) for c in range(ord("A"), ord("Z") + 1)]  # ISL also A-Z; many are two-handed

DEFAULT_LANGUAGE = LANGUAGE_ASL

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def model_path(language: str) -> str:
    """Return the on-disk path for a given language's trained model."""
    return os.path.join(MODELS_DIR, f"signbridge_{language.lower()}_model.joblib")


def label_encoder_path(language: str) -> str:
    return os.path.join(MODELS_DIR, f"signbridge_{language.lower()}_labels.joblib")


def dataset_csv_path(language: str) -> str:
    return os.path.join(PROCESSED_DATA_DIR, f"{language.lower()}_landmarks.csv")


def metadata_path(language: str) -> str:
    """JSON sidecar written by train_model.py alongside the .joblib model."""
    return os.path.join(MODELS_DIR, f"signbridge_{language.lower()}_metadata.json")


# ---------------------------------------------------------------------------
# UI Theme Colors (Mockup Palette)
# ---------------------------------------------------------------------------
THEME_NAVY = "#0b0f19"
THEME_NAVY_CARD = "#121626"
THEME_NAVY_CARD_SECONDARY = "#161c2d"
THEME_NAVY_BORDER = "#1f273f"
THEME_PURPLE = "#8b5cf6"
THEME_PURPLE_DARK = "#6d28d9"
THEME_PURPLE_GRADIENT = "linear-gradient(135deg, #7c3aed 0%, #a855f7 100%)"
THEME_ORANGE = "#f97316"
THEME_ORANGE_GRADIENT = "linear-gradient(135deg, #f97316 0%, #ec4899 100%)"
THEME_BLUE = "#3b82f6"
THEME_BLUE_GRADIENT = "linear-gradient(135deg, #2563eb 0%, #38bdf8 100%)"
THEME_CYAN = "#06b6d4"
THEME_GREEN = "#10b981"
THEME_RED = "#ef4444"
THEME_TEXT_PRIMARY = "#ffffff"
THEME_TEXT_MUTED = "#94a3b8"

# Phrase icons for UI chips
PHRASE_ICONS = {
    "PHRASE_HELP": "✋",
    "PHRASE_THANK_YOU": "😊",
    "PHRASE_WATER": "💧",
    "PHRASE_WASHROOM": "🚻",
    "PHRASE_YES": "✔️",
    "PHRASE_NO": "✖️",
}

