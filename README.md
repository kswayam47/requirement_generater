# Barclay Requirement Generator

A FastAPI-based toolkit for automatically analysing raw requirements, generating user stories and epics for Jira, and producing Software Requirement Specification (SRS) documents in multiple formats (Word, Excel, Markdown, PDF).

---

## Features

* **Natural-language requirements analysis** – Uses Large-Language-Model (LLM) powered NLP to break down unstructured text into clear, testable requirements.
* **Jira integration** – Converts analysed requirements into stories / tasks / bugs and optionally groups them under an epic.
* **Document generation**
  * Word `.docx` (templated)
  * Excel `.xlsx`
  * Markdown `.md` / PDF via *pandoc*
* **SRS generator** – Turn analysed requirements into a full-fledged SRS automatically.
* **Real-time WebSocket streaming** – Long-running operations stream their progress to the browser.
* **CLI & REST API** – Drive everything from the command line or via HTTP endpoints.

---

## Project layout

```
├── app.py                  # Lightweight FastAPI server (port 8002)
├── main.py                 # Full-featured FastAPI server (port 8000 by default)
├── requirement_analyzer.py # Core requirements-analysis logic
├── nlp.py                  # LLM wrapper (Gemini / openAI etc.)
├── srs_generator_v2.py     # Markdown / Word SRS builder
├── static/                 # Front-end files (HTML/JS/CSS)
├── requirements.txt        # Python dependencies
└── README.md               # You are here
```

---

## Getting started

1. **Clone & install**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Environment variables**
   * `GEMINI_API_KEY` – API key for Google Generative AI (or whichever provider you wire up in `nlp.py`).
   * Optional variables for Jira if you extend integration.
   You can place them in a `.env` file:
   ```ini
   GEMINI_API_KEY=your_key_here
   ```
3. **Run the API**
   ```bash
   uvicorn main:app --reload  # default 0.0.0.0:8000
   ```
   or, for the lightweight variant
   ```bash
   uvicorn app:app --port 8002
   ```
4. **Open** `http://localhost:8000` (or `:8002`) in the browser. Static assets are served from the `static` folder.

---

## Key endpoints (main.py)

| Method | Path | Purpose |
| ------ | ---- | ------- |
| POST | `/analyze_requirements` | Return structured requirements from free-form text. |
| POST | `/normalize_text` | Clean up pasted text (remove erratic line-breaks, etc.). |
| GET  | `/check_pandoc` | Verify *pandoc* installation (used for PDF/MD conversions). |
| WebSocket | `/ws` | Stream progress logs & interactively supply input during analysis/SRS generation. |

The **app.py** file exposes additional helper endpoints such as `/download`, `/convert_srs`, and `/download_requirement_docx`.

---

## SRS workflow (typical)

1. POST to `/analyze_requirements` with raw description.
2. Save the resulting `requirements_answers.txt`.
3. Trigger SRS build via WebSocket message `{ "type": "generate_srs" }` or run:
   ```bash
   python srs_generator_v2.py
   ```
4. Download `system_srs.md`, `system_srs.docx`, or `system_srs.pdf`.

---

## Deployment

The repository contains a **Procfile** (`web: uvicorn main:app --host=0.0.0.0 --port=$PORT`) and **runtime.txt** (Python version) so it can be deployed directly to Heroku or any container platform.

---

## Development tips

* Logs are emitted with `logging` – set `LOG_LEVEL` env var or tweak the config in the source.
* Long-running analyser tasks run in background threads; progress is pushed through `AnalyzerManager`.
* Word template is found in `word_template.docx` – customise styles there.
* For PDF export ensure *pandoc* is installed (`apt install pandoc` / `brew install pandoc`).

---

## Contributing

Pull requests are welcome! Please create an issue first if you plan to make significant changes.

---

## License

MIT License © 2025 Barclay Requirement Generator contributors.
