# Docker Guide — RAG Voice Agent

Run this project on **any machine** that has Docker installed —
Windows, macOS, or Linux — without installing Python, pip, or
any library manually.

---

## What is Docker and why use it?

Without Docker:
> "It works on my machine" — then you send it to a friend and
> they spend an hour installing the right Python version, the
> right packages, the right system libraries.

With Docker:
> You ship a **container** — a self-contained box that includes
> your code, the right Python version, every library, and the
> OS layer it needs. It runs identically everywhere.

**One analogy:** A container is like a food delivery box. The
restaurant (you) packs everything — food, utensils, sauce — into
one sealed box. The customer (any machine) just opens it and eats.
They don't need a kitchen.

---

## Files added to this project

```
rag_webapp/
├── Dockerfile            ← recipe to build the container image
├── .dockerignore         ← files to exclude from the image
├── docker-compose.yml    ← easy way to run with one command
├── .env.example          ← template for your secret API key
└── ... (existing files)
```

---

## Step 0 — Install Docker

Go to **https://docs.docker.com/get-docker/** and install
Docker Desktop for your OS.

Verify it works:
```bash
docker --version
# Docker version 24.x.x ...

docker-compose --version
# Docker Compose version 2.x.x ...
```

---

## Step 1 — Set up your API key

```bash
# Copy the example file
cp .env.example .env

# Open .env in any text editor and replace the placeholder:
#   GROQ_API_KEY=gsk_your_key_here
# with your real key from https://console.groq.com/keys
```

Your `.env` file should look like:
```
GROQ_API_KEY=gsk_abc123yourrealkey
```

> **Never commit `.env` to git.** It is already listed in
> `.dockerignore` and you should add it to `.gitignore` too.

---

## Step 2 — Build and run (the easy way)

```bash
docker-compose up --build
```

What this does, step by step:
1. Reads `docker-compose.yml`
2. Reads `Dockerfile` and builds the image (downloads Python,
   installs all packages — takes 1-3 minutes the first time)
3. Starts a container from that image
4. Maps port 8000 on your machine to port 8000 in the container
5. Loads your `GROQ_API_KEY` from `.env`

Open your browser: **http://localhost:8000**

---

## Step 3 — Stop it

```bash
# If running in the foreground (you see logs):
Ctrl + C

# Then remove the container:
docker-compose down

# Or stop and remove in one step:
docker-compose down
```

---

## Everyday commands

```bash
# Start in the background (no log output)
docker-compose up -d

# See if it's running
docker-compose ps

# Watch the logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild after you change any code or requirements.txt
docker-compose up --build

# Open a shell inside the running container (for debugging)
docker exec -it rag-voice-agent bash
```

---

## How the Dockerfile works (plain English)

```
FROM python:3.11-slim
```
Start from a base image that has Python 3.11 on Debian Linux.
"slim" means it strips out anything not essential — smaller image.

```
RUN apt-get install -y libgomp1 ffmpeg curl
```
Install OS-level packages the Python libraries need.
`apt-get` is Debian's package manager. `RUN` runs a shell command
during the build.

```
WORKDIR /app
```
Create a folder `/app` inside the container and make it the
current directory. Like running `mkdir /app && cd /app`.

```
COPY requirements.txt .
RUN pip install -r requirements.txt
```
Copy just the requirements file first and install.
Why separately? Docker caches this step. If `requirements.txt`
hasn't changed, Docker skips reinstalling on the next build.

```
COPY . .
```
Copy all your project files into `/app` inside the container.
(`.dockerignore` controls what gets skipped.)

```
EXPOSE 8000
```
Document that this container uses port 8000.
Does not actually open the port — `-p 8000:8000` does that.

```
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
The command that runs when the container starts.
`--host 0.0.0.0` means "accept connections from outside the
container" — without this, the browser can't reach it.

---

## How docker-compose.yml works (plain English)

```yaml
build: .
```
Build the image from the Dockerfile in the current directory.

```yaml
ports:
  - "8000:8000"
```
`HOST:CONTAINER` — connect port 8000 on your laptop to port 8000
inside the container. Change the left number to use a different
port on your machine (e.g. `"9000:8000"` → visit localhost:9000).

```yaml
environment:
  - GROQ_API_KEY=${GROQ_API_KEY}
```
Pass an environment variable into the container.
`${GROQ_API_KEY}` reads the value from your `.env` file or shell.

```yaml
volumes:
  - ./uploads:/app/uploads
```
Link a folder on your machine (`./uploads`) to a folder inside
the container (`/app/uploads`). Files saved here survive a
container restart. Without this, uploaded files disappear when
the container stops.

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/api/status"]
```
Every 30 seconds Docker pings `/api/status`. If it fails 3 times,
the container is marked unhealthy. Useful for monitoring.

---

## Sharing with someone else (no Docker Hub needed)

```bash
# Save the image to a file
docker save rag-agent -o rag-agent.tar

# On the other machine, load it
docker load -i rag-agent.tar

# Run it (they still need their own .env with their API key)
docker-compose up
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `port 8000 already in use` | Something else is on 8000. Change `"8000:8000"` to `"8001:8000"` in compose, visit localhost:8001 |
| `GROQ_API_KEY not set` | Check `.env` exists and has the key. Run `docker-compose down` then `up` again |
| `Cannot connect to Docker daemon` | Docker Desktop is not running — open it first |
| Changes to code not showing | Run `docker-compose up --build` to rebuild the image |
| `pip install` fails mid-build | Network issue. Run `docker-compose build --no-cache` to retry from scratch |
| Container exits immediately | Run `docker-compose logs` to see the error message |

---

## What Docker does NOT solve

- You still need a valid `GROQ_API_KEY` — Docker just makes
  it easy to pass it in without touching the code.
- The container needs internet access to call the Groq API
  and gTTS (Google TTS).
- This setup is for development and personal use. For
  production, you'd add HTTPS, proper secret management,
  and a process manager.