export GROQ_API_KEY="gsk_..."   # free at console.groq.com/keys

# Phase 1

python chat.py sample_corpus.txt

# Phase 2

python voice_agent.py sample_corpus.txt

# Phase 3

uvicorn main:app --reload --port 8000
# open http://localhost:8000
