# ============================================================
# Dockerfile — RAG Voice Agent
# ============================================================
#
# A Dockerfile is a recipe. Docker reads it top-to-bottom
# and builds a self-contained box (called an "image") that
# has everything the app needs to run — Python, libraries,
# your code — all bundled together.
#
# Build the image:
#   docker build -t rag-agent .
#
# Run a container from it:
#   docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... rag-agent
# ============================================================


# ── Step 1: Choose a base image ──────────────────────────────
#
# Think of a base image as a starting point — a pre-built OS
# with some software already installed.
#
# python:3.11-slim  →  Debian Linux + Python 3.11, stripped
#                       of anything not needed (smaller size).
#
FROM python:3.11-slim


# ── Step 2: Install system-level packages ────────────────────
#
# Some Python libraries need C libraries to work.
# apt-get is Debian's package manager (like pip but for the OS).
#
# libgomp1      — needed by some PDF libraries internally
# ffmpeg        — audio conversion; gTTS produces MP3, ffmpeg
#                 ensures playback tools are available
# curl          — useful for health-check scripts
#
# --no-install-recommends  →  keeps the image smaller
# rm -rf /var/lib/apt/lists/*  →  delete apt cache after install
#
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        portaudio19-dev \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*


# ── Step 3: Set the working directory ────────────────────────
#
# WORKDIR creates a folder inside the container and makes it
# the "current directory" for all following commands.
# Think of it as: cd /app  (but it also creates the folder).
#
WORKDIR /app


# ── Step 4: Copy requirements first, then install ────────────
#
# Why copy requirements.txt BEFORE copying your code?
# Docker caches each step. If you copy requirements.txt first
# and it hasn't changed, Docker skips the pip install step on
# the next build — saving minutes every time you rebuild.
#
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# ── Step 5: Copy the rest of the application code ────────────
#
# The  .  on the left = everything in your project folder.
# The  .  on the right = the WORKDIR (/app) inside the container.
# .dockerignore (see that file) tells Docker what to skip.
#
COPY . .


# ── Step 6: Declare the port ──────────────────────────────────
#
# EXPOSE tells Docker "this container listens on port 8000".
# This is documentation only — you still need -p 8000:8000
# when running to actually map it to your machine.
#
EXPOSE 8000


# ── Step 7: The command that starts the app ───────────────────
#
# CMD is what runs when you do  docker run ...
# Here we start uvicorn (the FastAPI server):
#   --host 0.0.0.0  →  listen on all network interfaces inside
#                       the container (required — default 127.0.0.1
#                       would be unreachable from outside)
#   --port 8000     →  the port the app listens on
#   main:app        →  file main.py, object named app
#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
