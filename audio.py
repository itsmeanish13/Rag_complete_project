"""
audio.py — Microphone capture, Groq Whisper transcription, and gTTS playback.

Pipeline:
  record_question()  →  bytes (WAV in memory)
      ↓
  transcribe(audio)  →  str  (via Groq Whisper API)
      ↓
  [LLM + RAG tool calls — handled by voice_agent.py]
      ↓
  speak(text)        →  plays MP3 via platform audio command
"""

from __future__ import annotations

import io
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
import time

# ── Optional heavy imports (caught at call-time for clear error messages) ─────
try:
    import pyaudio
    _PYAUDIO_OK = True
except ImportError:
    _PYAUDIO_OK = False

try:
    from gtts import gTTS
    _GTTS_OK = True
except ImportError:
    _GTTS_OK = False

from groq import Groq


# ── Recording constants ───────────────────────────────────────────────────────

SAMPLE_RATE    = 16_000   # Hz — Whisper works best at 16 kHz
CHANNELS       = 1        # mono
SAMPLE_WIDTH   = 2        # bytes (int16)
CHUNK_FRAMES   = 1_024    # frames per PyAudio read
SILENCE_THRESH = 300      # RMS below this → silence
SILENCE_SEC    = 2.0      # seconds of silence before auto-stop
MAX_SEC        = 30.0     # hard maximum recording length


# ── STT — Groq Whisper ────────────────────────────────────────────────────────

def transcribe(wav_bytes: bytes, client: Groq, language: str = "en") -> str:
    """
    Send raw WAV bytes to Groq Whisper and return the transcribed text.

    Groq supports: whisper-large-v3, whisper-large-v3-turbo, distil-whisper-large-v3-en
    whisper-large-v3-turbo is the best speed/accuracy tradeoff on the free tier.
    """
    wav_file = io.BytesIO(wav_bytes)
    wav_file.name = "question.wav"   # Groq inspects the filename for MIME type

    transcription = client.audio.transcriptions.create(
        file=wav_file,
        model="whisper-large-v3-turbo",
        language=language,
        response_format="text",
    )
    # Groq returns a plain string when response_format="text"
    return str(transcription).strip()


# ── TTS — gTTS + platform playback ───────────────────────────────────────────

def speak(text: str, lang: str = "en", slow: bool = False) -> None:
    """
    Convert *text* to speech with gTTS and play it back.

    gTTS writes an MP3; we play it with whichever CLI player is available:
      macOS   → afplay  (built-in)
      Linux   → mpg123 or ffplay or aplay (after converting via ffmpeg)
      Windows → start (built-in, opens default media player)
    """
    if not _GTTS_OK:
        raise ImportError("gTTS is not installed. Run:  pip install gtts")

    # Sanitise text: strip markdown emphasis that sounds bad when read aloud
    clean = _strip_markdown(text)

    tts = gTTS(text=clean, lang=lang, slow=slow)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name
    try:
        tts.save(tmp_path)
        _play_audio_file(tmp_path)
        # On Windows the default player is launched asynchronously via
        # the shell 'start' command; give it a moment to open the file
        # before we unlink it to avoid "file not found" errors.
        if platform.system() == "Windows":
            time.sleep(2)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _play_audio_file(path: str) -> None:
    """Play an audio file using the best available CLI tool."""
    system = platform.system()

    if system == "Darwin":
        _run(["afplay", path])

    elif system == "Windows":
        # 'start' is a shell builtin
        subprocess.run(f'start "" "{path}"', shell=True, check=False)

    else:  # Linux / BSD
        if shutil.which("mpg123"):
            _run(["mpg123", "-q", path])
        elif shutil.which("ffplay"):
            _run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path])
        elif shutil.which("vlc"):
            _run(["vlc", "--intf", "dummy", "--play-and-exit", path])
        else:
            print(
                "  Warning: no audio player found.\n"
                "  Install one:  sudo apt install mpg123  (or ffmpeg)",
                file=sys.stderr,
            )


def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        print(f"  Warning: player not found: {cmd[0]}", file=sys.stderr)
    except subprocess.CalledProcessError:
        pass  # non-zero exit is usually harmless (end of file)


# ── Microphone recording ──────────────────────────────────────────────────────

def record_question(prompt: str = "🎙  Listening… (speak now)") -> bytes:
    """
    Record from the default microphone until silence is detected or the
    maximum duration is reached. Returns raw WAV bytes (in memory).
    """
    if not _PYAUDIO_OK:
        raise ImportError(
            "PyAudio is not installed.\n"
            "  macOS:   brew install portaudio && pip install pyaudio\n"
            "  Linux:   sudo apt install python3-pyaudio portaudio19-dev\n"
            "  Windows: pip install pyaudio"
        )

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_FRAMES,
    )

    print(f"\n{prompt}")
    frames: list[bytes] = []
    silent_chunks = 0
    max_chunks    = int(MAX_SEC * SAMPLE_RATE / CHUNK_FRAMES)
    silence_limit = int(SILENCE_SEC * SAMPLE_RATE / CHUNK_FRAMES)

    try:
        for _ in range(max_chunks):
            chunk = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            frames.append(chunk)
            rms = _rms(chunk)
            if rms < SILENCE_THRESH:
                silent_chunks += 1
                if silent_chunks >= silence_limit and len(frames) > silence_limit:
                    break
            else:
                silent_chunks = 0
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    print("  ✓ Recording complete.")
    return _frames_to_wav(frames)


def _rms(chunk: bytes) -> float:
    """Root-mean-square amplitude of a raw int16 PCM chunk."""
    import struct
    n = len(chunk) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", chunk)
    return (sum(s * s for s in samples) / n) ** 0.5


def _frames_to_wav(frames: list[bytes]) -> bytes:
    """Pack raw PCM frames into an in-memory WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


# ── Text cleanup for TTS ──────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove markdown symbols that read badly aloud."""
    import re
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)   # bold / italic
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)          # code spans
    text = re.sub(r"#{1,6}\s*", "", text)                   # headings
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # links
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.M)   # bullets
    text = re.sub(r"\n{2,}", ". ", text)                    # paragraph breaks
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()