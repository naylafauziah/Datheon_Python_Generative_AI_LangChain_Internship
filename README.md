# Datheon Python + Generative AI & LangChain Internship — Tasks 1–4

> **Status:** Ready-to-run project source. Unit tests can be executed locally; live Gemini/Tavily integration requires API keys, network access, and available API quota. The bundled PDF/TXT files are **fictional demonstration materials** and do not represent official Datheon policy.

## Architecture

- **Task 1:** `task1_extract.py` — Google Gemini structured JSON response + Pydantic v2 schema with extra fields rejected and consistency checks.
- **Task 2:** `task2_rag.py` / `src/rag.py` — PDF/TXT -> LangChain loaders -> recursive chunks -> Gemini vector embeddings -> persistent Chroma -> retrieved context -> Gemini -> page citations.
- **Task 3:** `task3_agent.py` — LangChain `create_agent` with Tavily search, secure AST calculator, and sandboxed local file reading; observable tool action and result log.
- **Task 4:** `task4_app.py` — Streamlit interactive document Q&A with uploads, persistent RAG vector store, streaming LLM tokens, and per-browser-session chat memory.

> The old `RetrievalQA` and `AgentExecutor` patterns have changed in current LangChain versions. This project uses explicit retrieval and `create_agent` for more maintainable modern implementations.

## 1. Prepare a Python environment

Recommended: **Python 3.11 or 3.12**. Run the following from the `datheon_ai_internship` folder in VS Code Terminal.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If your machine only has Python 3.12, replace `py -3.11` with `py -3.12`. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal only and retry.

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## 2. Configure API keys

1. Create a Gemini API key at **https://aistudio.google.com/apikey**.
2. Create a Tavily API key at **https://app.tavily.com/** (required only for Task 3).
3. Edit `.env`:

```dotenv
GOOGLE_API_KEY=your_real_google_key_here
TAVILY_API_KEY=your_real_tavily_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
EMBEDDING_MODEL=gemini-embedding-001
RAG_TOP_K=4
```

Never commit `.env` to GitHub. Gemini and embedding quotas/pricing depend on your account. The exact model IDs may need adjustment if unavailable in your region/account; reindex RAG after changing embedding models.

## 3. Task 1 — JSON Extractor

### Run

```bash
python task1_extract.py --type invoice --input data/samples/invoice.txt
python task1_extract.py --type resume --input data/samples/resume.txt
```

Outputs are written to `outputs/invoice_extracted.json` and `outputs/resume_extracted.json`.

### How it works

1. Read a raw UTF-8 invoice or CV (`.txt`, `.md`, or digital `.pdf`).
2. Add domain-specific prompt instructions and two miniature few-shot interpretations.
3. Ask Gemini to generate constrained JSON using the Pydantic schema.
4. Explicitly validate the response using `model_validate_json()`.
5. On schema validation errors, retry once. Fail clearly rather than saving invalid data.
6. Save standardized JSON. Missing optional data is `null`, missing arrays are `[]`.

### What to test

- Correct vendor, invoice number, invoice date, line items, tax and total. Test a wrong `quantity * unit_price` value too.
- Correct resume name, email, work experience, education and skills.
- An invoice whose `subtotal + tax != total` must fail validation.
- An invalid email or unknown schema field must fail validation.

**Limit:** Strict Pydantic validation proves the JSON shape and implemented consistency constraints, **not** that every extracted fact is semantically correct. Compare extracted values to the source document.

## 4. Task 2 — PDF RAG Q&A

The project includes two fictional demo documents:
- `data/docs/kebijakan_magang.txt`
- `data/docs/panduan_proyek_demo.pdf` (2 pages; tests page citations)

Add your own **digital/text-based** PDF or TXT files to `data/docs`, then run:

```bash
python task2_rag.py index
python task2_rag.py ask "Berapa bobot penilaian kualitas kode?"
python task2_rag.py ask "Di halaman berapa kriteria keberhasilan proyek dijelaskan?"
python task2_rag.py ask "Siapa presiden pertama Mars?"
```

Expected behavior: the first question finds `40%` in the fictional TXT; the second should find the relevant PDF page; the unsupported question should produce **`Tidak ditemukan dalam dokumen.`**. Exact wording depends on the model and retrieval.

### Pipeline explanation

1. `PyPDFLoader` extracts PDF text page by page; `TextLoader` reads TXT.
2. `RecursiveCharacterTextSplitter` creates chunks (900 chars, 150 chars overlap).
3. `GoogleGenerativeAIEmbeddings` converts chunks into semantic vectors.
4. `Chroma` stores vectors persistently in `chroma_db/`.
5. Similarity search retrieves the top K chunks for each question.
6. Gemini receives **only those retrieved passages as evidence** with IDs `[S1]`, `[S2]`, etc.
7. Answers must cite retrieved IDs; code rejects missing or nonexistent citation IDs.
8. CLI prints retrieved PDF page numbers (converted from 0-based to 1-based).

**Important limitations:** The retrieval step and source-ID validation reduce unsupported statements, but cannot mathematically guarantee factual faithfulness. PDF scans require OCR, which this baseline does not implement. A missing PDF may be caused by poor extraction or imperfect retrieval, and the embedding/index step uses API quota.

## 5. Task 3 — Autonomous AI Agent

```bash
python task3_agent.py --query "Read kebijakan_magang.txt, list the evaluation weights, then calculate 40 percent of 250. Also search the web for the latest official LangChain agents documentation URL."
```

Three required tools and one helper tool:

- `web_search`: Tavily search with recent results and source URLs.
- `math_calculator`: safe arithmetic through Python `ast`; it never uses `eval()` or arbitrary Python execution.
- `read_local_file`: reads TXT/PDF within `data/docs` only.
- `list_local_files`: helps the agent find available documents.

The `create_agent` tool-calling loop determines which tools to invoke. Console trace logs **ACTION**, **OBSERVATION**, then **FINAL ANSWER**. It intentionally does not expose model-private chain-of-thought (a raw “THOUGHT” trace is neither required for debugging nor reliably available from hosted LLMs).

**Safety:** `read_local_file` blocks directory traversal, math is limited to arithmetic, and web results are treated as untrusted data. The agent is an educational baseline, not a production secure browsing environment.

## 6. Task 4 — Streamlit Chatbot

First index documents using Task 2, or press **Index / Re-index documents** in the UI. Then start:

```bash
streamlit run task4_app.py
```

Open the local address shown by Streamlit (usually **http://localhost:8501**).

How to use:

1. Use the sidebar to upload new PDF/TXT files.
2. Press **Index / Re-index documents**. Uploads get written to `data/docs` and all documents are indexed.
3. Ask a question in the chat input.
4. Watch the answer stream to the UI while the model generates it.
5. Expand **Retrieved references** to inspect source file, page and snippet.
6. Ask a follow-up question; prior turns are available in `st.session_state.messages`.
7. Press **Clear chat history** to reset the current browser session.

**Memory scope:** History lives in Streamlit session state for that browser session. It does not persist across all browser resets or server restarts. Chroma vectors do persist on local disk.

**Deployment caution:** This is intended for **local demonstration**. File uploads and the Chroma collection are shared by all users of one server process. Add authentication, per-user storage, size limits, and data governance before any public deployment, especially for private CVs/invoices.

## 7. Run automated tests

```bash
python -m pytest -q
python -m compileall -q .
```

`test_schemas.py`: Pydantic acceptance/rejection, total consistency and email checks.
`test_math.py`: arithmetic, security-oriented invalid AST cases, divide-by-zero.
`test_rag_helpers.py`: citation validation and preservation of source/page metadata.

Unit tests **do not use live API keys**. Integration smoke tests require working Gemini and Tavily API keys and a network connection.

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `GOOGLE_API_KEY belum diatur` | `.env` missing or placeholder | Create `.env` and use real key |
| `ModuleNotFoundError` | Wrong interpreter or missing install | Activate `.venv`, `pip install -r requirements.txt` |
| Gemini `404 model not found` | Model not accessible in account | Change `GEMINI_MODEL`; check Google model list |
| Gemini `429` | Quota/rate limit exceeded | Reduce requests or check quota/billing |
| No document answer | PDF scanned, unindexed or retrieval miss | Use text-based PDF; reindex; add more targeted question |
| `index` empty | No readable text in `data/docs` | Add a digital PDF or UTF-8 TXT |
| Chroma dimension mismatch | Embedding model changed | Run `python task2_rag.py index` |
| Tavily error | Missing key or exhausted quota | Set/refresh `TAVILY_API_KEY` |
| Streamlit displays citation warning | Model did not cite retrieved IDs | Rephrase with a more targeted question |

## 9. Suggested submission evidence

Take screenshots for each task:

1. **Task 1:** input TXT + console structured JSON + saved JSON in `outputs`.
2. **Task 2:** `index` logs, a grounded answer with `[S1]` and a PDF citation, unsupported question fallback.
3. **Task 3:** ACTION and OBSERVATION traces for three tools, then FINAL ANSWER.
4. **Task 4:** full chat interface, uploaded files, streamed output, cited source/page, and conversational follow-up.
5. Include `pytest` output and a sanitized README/GitHub repository URL. **Do not include API keys.**

## References (official docs)

- Gemini structured response + SDK: https://googleapis.github.io/python-genai/
- LangChain Gemini chat and embeddings: https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai
- LangChain Chroma: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
- LangChain agents: https://docs.langchain.com/oss/python/langchain/agents
- Tavily Python SDK: https://github.com/tavily-ai/tavily-python
- Streamlit chat: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
