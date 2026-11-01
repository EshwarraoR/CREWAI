# CrewAI Assignment Backend

FastAPI service exposing 10 CrewAI projects (4 Crews, 6 Flows) using `gemini/gemini-3.1-flash-lite-preview`.

## Run locally
```
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
uvicorn app:app --reload
```

## API
Every route is `POST` with JSON `{"input": "..."}` and returns `{"output": "..."}`.

| Route | Project | Input text |
|---|---|---|
| /agent1 | Blog post crew | topic |
| /agent2 | Competitor analysis crew | competitors, e.g. `Notion vs Coda` |
| /agent3 | Meeting notes crew | transcript |
| /agent4 | Data report crew | CSV text |
| /agent5 | Customer support flow | customer message |
| /agent6 | Resume screening flow | resume text (optional `Requirements: ...`) |
| /agent7 | Bug triage flow | issue title and body |
| /agent8 | Content repurposing flow | article text |
| /agent9 | Lead qualification flow | `Company name, https://site.com` |
| /agent10 | Approval report flow | topic (optional `Feedback: ...` to trigger revision) |

## Deploy (Render)
Build: `pip install -r requirements.txt`
Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
Env var: `GEMINI_API_KEY`

## Keeping the free Render instance awake
Point an uptime monitor (e.g. UptimeRobot, cron-job.org) at `GET /health` every 5 minutes.
