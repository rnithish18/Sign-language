# ─── Stage 1: build ──────────────────────────────────────────────────────────
FROM python:3.9-slim AS builder

# System libs needed to compile opencv & mediapipe wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps in a virtualenv so they're easy to copy in stage 2
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# pyttsx3 uses platform speech engines unavailable in Linux containers;
# swap it for a no-op stub or just install it (it will fall back gracefully).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        # opencv-headless is smaller and has no GUI deps (we never show windows in Docker)
        opencv-python-headless==4.10.0.84 \
        mediapipe==0.10.14 \
        numpy==1.26.4 \
        pandas==2.2.2 \
        "scikit-learn==1.5.1" \
        joblib==1.4.2 \
        pyttsx3==2.90 \
        "gTTS==2.5.1" \
        "pyspellchecker==0.8.1" \
        "streamlit==1.37.0" \
        "streamlit-webrtc==0.47.7" \
        "av==12.3.0" \
        "Pillow==10.4.0"

# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.9-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source
COPY . .

# Streamlit config: disable browser auto-open, bind to all interfaces
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.fileWatcherType=none"]
