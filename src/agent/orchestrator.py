"""
orchestrator.py
===============

Reason-Action (ReAct) agent loop for autonomous geospatial analysis of the
Kamrup region (Assam, India).

Architecture
------------
1. **RAG Retrieval** — Queries the ChromaDB vector index to discover
   relevant local dataset schemas (filenames, columns, CRS, geometry
   types).
2. **System Prompt Injection** — Injects the retrieved metadata into a
   structured system prompt that forces the LLM to emit a ``<thought>``
   block *before* any code.
3. **LLM Call** — Sends the prompt to Gemini, OpenAI, or Anthropic
   (auto-detected from environment).
4. **Sandboxed Execution** — Runs the generated code inside a
   restricted ``exec()`` with only the approved geoprocessing tools
   and standard data-science imports available.
5. **Self-Correction** — If ``exec()`` raises an exception, the
   traceback is fed back to the LLM for up to **2 retry attempts**.

Usage
-----
>>> from src.agent.orchestrator import run_agent
>>> result = run_agent("Buffer all rivers by 500 m and show the result.")
>>> print(result.thought)
>>> print(result.code)
>>> print(result.output)
"""

from __future__ import annotations

import io
import logging
import os
import pathlib
import re
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.rag.vector_index import (
    build_kamrup_index,
    get_kamrup_context,
    format_context_for_prompt,
)
from src.tools import geoprocessing  # exposed to the exec() sandbox

logger = logging.getLogger(__name__)

# Load .env once at import time (no-op if the file doesn't exist).
load_dotenv()

# ---------------------------------------------------------------------------
# Project paths (used inside the sandbox so LLM-generated code can
# reference data files with relative paths).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_SUPPORTED_PROVIDERS = ("groq", "gemini", "openai", "anthropic")

# Hard cap on self-correction retries.  Prevents infinite loops that
# burn through API quota when the LLM generates persistently buggy code.
_MAX_RETRIES = 2

# Maximum character length for the RAG metadata context injected into
# the system prompt.  Prevents context-window bloat if the metadata
# catalogue grows large.  ~4000 chars ≈ ~1000 tokens.
_MAX_CONTEXT_CHARS = 4000

# ---------------------------------------------------------------------------
# System prompt — the "brain" of the ReAct agent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an advanced GIS Agent analyzing the **Kamrup region** (Assam,
India).  You have access to a curated set of local spatial datasets and
a Python geoprocessing toolkit.  Follow the ReAct (Reason → Action)
pattern strictly.

═══════════════════════════════════════════════════════════════════
STEP 1 — RAG CONTEXT (already retrieved for you)
═══════════════════════════════════════════════════════════════════
The following metadata was retrieved from the vector database.  Use it
to discover **exact filenames, file paths, column names, CRS, and
geometry types** for the datasets relevant to the user's query.

{metadata_context}

═══════════════════════════════════════════════════════════════════
STEP 2 — EMIT A <thought> BLOCK
═══════════════════════════════════════════════════════════════════
Before writing ANY code, output a `<thought>` block explaining:
- Which dataset file(s) you will load and why.
- The exact column names you will reference.
- Which geoprocessing function(s) you will call and in what order.
- What CRS transformations are needed (all Kamrup data is EPSG:4326;
  metric operations require EPSG:32646 via `project_to_kamrup_utm`).
- Any edge cases or assumptions.

═══════════════════════════════════════════════════════════════════
STEP 3 — WRITE EXECUTABLE PYTHON CODE
═══════════════════════════════════════════════════════════════════
Write code inside a fenced ```python block.  You may ONLY use these
pre-imported symbols:

**Data Loading (pre-imported):**
- `gpd` (geopandas) — load data with `gpd.read_file(PROJECT_ROOT / "path")`
- `pd` (pandas) — tabular operations
- `json`, `pathlib.Path`
- `PROJECT_ROOT` — pathlib.Path pointing to the project root directory

**Kamrup Geoprocessing Toolkit (all in `geoprocessing.*`):**
- `geoprocessing.project_to_kamrup_utm(gdf)`
  Reprojects any GeoDataFrame to EPSG:32646 (UTM Zone 46N).
  **Always call this before any distance / buffer / area operation.**

- `geoprocessing.create_buffer(gdf, distance_meters)`
  Auto-projects to EPSG:32646, then buffers by the given metres.
  Returns a GeoDataFrame in EPSG:32646.

- `geoprocessing.intersect_features(gdf1, gdf2)`
  Spatial intersection.  Both GDFs **must** be in EPSG:32646.
  Raises `CRSMismatchError` if they are not.

- `geoprocessing.calculate_area(gdf)`
  Adds an `area_sq_km` column.  Auto-projects if needed.

- `geoprocessing.reproject_layer(gdf, target_epsg)`
  Reprojects to any EPSG code (e.g. 4326 for map display).

- `geoprocessing.buffer_vector(input_gdf, distance)`
  Low-level buffer.  Requires an already-projected CRS.

- `geoprocessing.intersect_layers(gdf_a, gdf_b)`
  Low-level intersection.  Requires matching CRS on both inputs.

**Standard workflow:**
1. Load data:  `rivers = gpd.read_file(PROJECT_ROOT / "data/kamrup_synthetic/vector/rivers_kamrup.geojson")`
2. Project:    `rivers_utm = geoprocessing.project_to_kamrup_utm(rivers)`
3. Analyse:    `buffered = geoprocessing.create_buffer(rivers, 500)`
4. Intersect:  `result = geoprocessing.intersect_features(buffered, other_utm)`
5. Area:       `result = geoprocessing.calculate_area(result)`
6. Display:    `result_4326 = geoprocessing.reproject_layer(result, 4326)`
7. Output:     `print(result_4326.to_json())`

**CRITICAL — Always print your final result as GeoJSON:**
`print(result_gdf.to_json())`

**NEVER DO THESE:**
- Do NOT read or print raw spatial coordinates or full GeoDataFrame rows
  in your `<thought>` block.  Only reference column names and schemas.
- Do NOT embed raw GeoJSON, coordinate arrays, or WKT strings in your
  reasoning.  The execution sandbox handles all data locally.
- Keep your code concise.  Avoid loading data you do not need.
"""

_CORRECTION_PROMPT = """\
The code you generated raised an error during execution.

**Traceback:**
```
{traceback}
```

**Previous code that failed:**
```python
{failed_code}
```

Analyse the error carefully.  Common issues include:
- CRS mismatch → call `geoprocessing.project_to_kamrup_utm()` first
- Wrong column name → check the RAG metadata for exact names
- File not found → verify the path matches the metadata file_path

Emit a new `<thought>` block explaining what went wrong and how you
will fix it, then output corrected Python code inside a fenced
```python block.  Follow all the original rules.

This is retry {retry_num} of {max_retries}.  Make it count.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AgentResult:
    """Container for a single orchestration run.

    Attributes
    ----------
    query : str
        The original natural-language task.
    metadata_context : str
        Serialised layer schemas returned by the RAG retriever.
    thought : str
        The LLM's chain-of-thought reasoning (``<thought>`` block).
    code : str
        The Python code extracted from the LLM response.
    output : str
        Captured ``stdout`` / ``stderr`` from executing the code.
    error : Optional[str]
        Traceback string if execution failed (after all retries).
    correction_thought : Optional[str]
        Thought block from the last self-correction attempt, if any.
    correction_code : Optional[str]
        Corrected code from the last self-correction attempt, if any.
    success : bool
        ``True`` if the code executed without unhandled exceptions.
    raw_llm_responses : list[str]
        Raw text of every LLM response in this run (for auditing).
    retry_count : int
        Number of self-correction retries that were attempted.
    """

    query: str = ""
    metadata_context: str = ""
    thought: str = ""
    code: str = ""
    output: str = ""
    error: Optional[str] = None
    correction_thought: Optional[str] = None
    correction_code: Optional[str] = None
    success: bool = False
    raw_llm_responses: List[str] = field(default_factory=list)
    retry_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# LLM provider helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_provider() -> str:
    """Determine which LLM provider to use based on available API keys.

    Priority order:
    1. ``LLM_PROVIDER`` environment variable (explicit override).
    2. ``GROQ_API_KEY`` → ``"groq"`` (fastest, most generous free tier)
    3. ``GOOGLE_API_KEY`` → ``"gemini"``
    4. ``OPENAI_API_KEY`` → ``"openai"``
    5. ``ANTHROPIC_API_KEY`` → ``"anthropic"``

    Returns
    -------
    str
        ``"groq"``, ``"gemini"``, ``"openai"``, or ``"anthropic"``.

    Raises
    ------
    EnvironmentError
        If no usable API key is found.
    """
    explicit = os.getenv("LLM_PROVIDER", "").lower().strip()
    if explicit in _SUPPORTED_PROVIDERS:
        return explicit

    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"

    raise EnvironmentError(
        "No LLM API key found.  Set one of: GROQ_API_KEY, GOOGLE_API_KEY, "
        "OPENAI_API_KEY, or ANTHROPIC_API_KEY in your environment "
        "or .env file.  Optionally set LLM_PROVIDER to "
        "'groq', 'gemini', 'openai', or 'anthropic'."
    )


def _get_chat_model(provider: str):
    """Instantiate and return the appropriate LangChain chat model.

    Uses cost-efficient models by default.  Structured geoprocessing
    planning and schema matching do NOT require frontier-grade models
    like GPT-4 or Claude Opus — lightweight variants are 95-97% cheaper
    and equally capable for this task.

    Parameters
    ----------
    provider : str
        ``"gemini"``, ``"openai"``, or ``"anthropic"``.

    Returns
    -------
    BaseChatModel
    """
    if provider == "groq":
        from langchain_groq import ChatGroq

        # llama-3.1-8b-instant: genuinely free (no card, no spend cap),
        # extremely fast (Groq's custom LPU chips), and ~14,400
        # requests/day on the free tier — by far the most headroom of
        # any provider here for a handful of straightforward
        # code-generation queries.  Not a deep reasoner, but this task
        # (schema-grounded geoprocessing code-gen) doesn't need one.
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=4096,
            max_retries=0,
            timeout=60,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # max_retries=0: see note in the OpenAI branch below — the
        # orchestrator's own _call_llm() now does ONE deliberate,
        # logged retry on genuine rate limits. Letting the SDK ALSO
        # retry 5x internally (the old setting) meant a single failed
        # call could fire 5+ silent requests and exhaust a free-tier
        # quota before you ever saw a response.
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0,
            max_output_tokens=8192,
            max_retries=0,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        # gpt-4o-mini is ~95% cheaper than gpt-4o and fast enough
        # for spatial code generation tasks.
        #
        # max_retries=0: the orchestrator ALREADY implements its own
        # retry loop (self-correction, up to _MAX_RETRIES).  Leaving
        # LangChain's default internal retries (2-3 extra silent HTTP
        # calls per .invoke()) on top of that means a single failed
        # query can fire 6-9 real API requests in a few seconds and
        # blow through a low RPM / quota limit before you ever see a
        # response.  We retry deliberately, once, at the orchestrator
        # level instead — see _call_llm().
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=4096,
            max_retries=0,
            timeout=60,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # claude-3-haiku is ~97% cheaper than claude-3-opus and
        # handles structured planning + code generation well.
        # See max_retries note above.
        return ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0.0,
            max_tokens=4096,
            max_retries=0,
            timeout=60,
        )

    raise ValueError(f"Unsupported LLM provider: '{provider}'.")


def _call_llm(chat_model, messages: list, *, _retry_on_rate_limit: bool = True) -> str:
    """Invoke the chat model and return the assistant's text content.

    Performs at most ONE deliberate retry, and only for a genuine
    rate-limit (too many requests per minute) — never for quota
    exhaustion, since retrying an empty wallet just wastes more
    requests against the same dead key.

    Parameters
    ----------
    chat_model : BaseChatModel
    messages : list[BaseMessage]
    _retry_on_rate_limit : bool
        Internal flag used to cap the deliberate retry to a single
        attempt (prevents recursive retry storms).

    Returns
    -------
    str
        Raw text content of the AI response.
    """
    import time

    from langchain_core.messages import AIMessage

    try:
        response: AIMessage = chat_model.invoke(messages)
        return response.content
    except Exception as e:
        error_msg = str(e)
        lowered = error_msg.lower()

        is_quota_exhausted = (
            "insufficient_quota" in lowered
            or "exceeded your current quota" in lowered
        )
        is_rate_limited = (
            not is_quota_exhausted
            and ("429" in error_msg or "rate_limit" in lowered
                 or "resource_exhausted" in lowered)
        )
        is_auth_error = (
            "401" in error_msg or "invalid_api_key" in lowered
            or "authentication" in lowered
        )

        if is_quota_exhausted:
            # No point retrying — billing/credits are the only fix.
            raise RuntimeError(
                "Your API key has NO remaining quota/credits "
                "(insufficient_quota). Retrying will not help.\n\n"
                "To resolve this:\n"
                "1. Go to https://platform.openai.com/settings/organization/billing "
                "(or your provider's billing page) and add a payment "
                "method / credits.\n"
                "2. Or switch providers in .env (GOOGLE_API_KEY / "
                "ANTHROPIC_API_KEY) if that account has quota.\n"
                "3. Or enable 'Mock Mode' in the sidebar to keep "
                "developing the UI without calling any LLM."
            ) from e

        if is_auth_error:
            raise RuntimeError(
                "Your API key was rejected (401 invalid/expired). "
                "Double-check the key in .env has no typos, extra "
                "spaces, or stray quote characters, and that it "
                "hasn't been revoked."
            ) from e

        if is_rate_limited and _retry_on_rate_limit:
            # One deliberate, short backoff — NOT a silent SDK retry
            # storm. If this single retry also fails, we give up and
            # surface the error rather than hammering the API further.
            logger.warning(
                "Rate limited — waiting 5s for a single deliberate "
                "retry before giving up."
            )
            time.sleep(5)
            return _call_llm(
                chat_model, messages, _retry_on_rate_limit=False
            )

        if is_rate_limited:
            raise RuntimeError(
                "You are being rate-limited (429 — too many requests "
                "per minute), not out of quota.\n\n"
                "To resolve this:\n"
                "1. Wait ~60 seconds before trying again.\n"
                "2. Ask fewer questions per minute, or request a "
                "higher rate limit tier from your provider.\n"
                "3. Switch providers in .env if another account has "
                "more headroom."
            ) from e

        raise


# ═══════════════════════════════════════════════════════════════════════════
# Parsing helpers
# ═══════════════════════════════════════════════════════════════════════════

_THOUGHT_RE = re.compile(
    r"<thought>(.*?)</thought>",
    re.DOTALL | re.IGNORECASE,
)

_CODE_RE = re.compile(
    r"```python\s*\n(.*?)```",
    re.DOTALL,
)


def _parse_response(text: str) -> tuple[str, str]:
    """Extract the thought block and code block from an LLM response.

    Parameters
    ----------
    text : str
        Raw LLM output.

    Returns
    -------
    tuple[str, str]
        ``(thought, code)`` — either may be empty if the LLM did not
        produce the expected tags.
    """
    thought_match = _THOUGHT_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    code_match = _CODE_RE.search(text)
    code = code_match.group(1).strip() if code_match else ""

    if not thought:
        logger.warning("LLM response did not contain a <thought> block.")
    if not code:
        logger.warning("LLM response did not contain a ```python code block.")

    return thought, code


# ═══════════════════════════════════════════════════════════════════════════
# Sandboxed execution
# ═══════════════════════════════════════════════════════════════════════════

def _build_sandbox_globals() -> Dict[str, Any]:
    """Return a restricted ``globals()`` dict for ``exec()``.

    Only the geoprocessing module, standard data-science imports, and
    safe builtins are available inside the sandbox.

    Returns
    -------
    dict
    """
    import geopandas as _gpd
    import pandas as _pd
    import json as _json
    import pathlib as _pathlib

    safe_builtins = {
        k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
        for k in (
            "print", "len", "range", "enumerate", "zip", "map", "filter",
            "sorted", "reversed", "list", "dict", "set", "tuple",
            "str", "int", "float", "bool", "type", "isinstance",
            "min", "max", "sum", "abs", "round",
            "True", "False", "None",
            "ValueError", "TypeError", "KeyError", "RuntimeError",
            "Exception",
        )
        if (isinstance(__builtins__, dict) and k in __builtins__)
        or (not isinstance(__builtins__, dict) and hasattr(__builtins__, k))
    }

    def _sandboxed_open(file, mode="r", *args, **kwargs):
        """Restrict file access to READ-ONLY, inside PROJECT_ROOT only.

        The unrestricted builtin `open()` previously let generated code
        read or overwrite ANY file the process could see (e.g. `.env`,
        other users' data, or anything writable on disk). Untrusted
        LLM output should never get raw filesystem access.
        """
        resolved = pathlib.Path(_PROJECT_ROOT / file).resolve() \
            if not pathlib.Path(file).is_absolute() \
            else pathlib.Path(file).resolve()

        if _PROJECT_ROOT not in resolved.parents and resolved != _PROJECT_ROOT:
            raise PermissionError(
                f"Sandboxed code may only access files inside "
                f"{_PROJECT_ROOT}, got: {file!r}"
            )
        if any(m in mode for m in ("w", "a", "x", "+")):
            raise PermissionError(
                "Sandboxed code may only OPEN FILES READ-ONLY. "
                "Write/append access is disabled for safety."
            )
        return open(resolved, mode, *args, **kwargs)

    safe_builtins["open"] = _sandboxed_open

    return {
        "__builtins__": safe_builtins,
        "geoprocessing": geoprocessing,
        "gpd": _gpd,
        "pd": _pd,
        "json": _json,
        "pathlib": _pathlib,
        "Path": _pathlib.Path,
        "PROJECT_ROOT": _PROJECT_ROOT,
    }


def _execute_code(code: str) -> tuple[str, Optional[str]]:
    """Execute *code* inside a sandboxed ``exec()`` call.

    Parameters
    ----------
    code : str
        Python source code to execute.

    Returns
    -------
    tuple[str, str | None]
        ``(captured_stdout, traceback_or_none)``
    """
    sandbox_globals = _build_sandbox_globals()
    sandbox_locals: Dict[str, Any] = {}

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, sandbox_globals, sandbox_locals)  # noqa: S102
        captured = stdout_buf.getvalue() + stderr_buf.getvalue()
        return captured.strip(), None

    except Exception:
        tb = traceback.format_exc()
        captured = stdout_buf.getvalue() + stderr_buf.getvalue()
        full_output = (captured.strip() + "\n" + tb).strip()
        return full_output, tb


# ═══════════════════════════════════════════════════════════════════════════
# RAG context builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_metadata_context(query: str, n_results: int = 3) -> str:
    """Query the RAG retriever and serialise matching schemas as text.

    Parameters
    ----------
    query : str
        The user's natural-language task.
    n_results : int
        Maximum number of layers to retrieve.

    Returns
    -------
    str
        Human-readable block listing each matched layer's full schema,
        ready for injection into the system prompt.
    """
    try:
        results = get_kamrup_context(query, n_results=n_results)
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s — proceeding without context.", exc)
        return "(No metadata context available — RAG retrieval failed.)"

    if not results:
        return "(No matching layers found in the metadata catalogue.)"

    return format_context_for_prompt(results)


# ═══════════════════════════════════════════════════════════════════════════
# Public API — ReAct Agent Loop
# ═══════════════════════════════════════════════════════════════════════════

def run_agent(
    user_query: str,
    *,
    provider: Optional[str] = None,
    n_context_layers: int = 3,
    auto_execute: bool = True,
    max_retries: int = _MAX_RETRIES,
) -> AgentResult:
    """Run the full Reason-Action (ReAct) orchestration loop.

    This is the main entry point for the Agentic GIS Orchestrator.  It
    chains RAG retrieval → LLM reasoning → sandboxed execution → self-
    correction into a single call.

    Parameters
    ----------
    user_query : str
        Natural-language description of the geospatial task to perform.
        Examples:
        - ``"Buffer all rivers by 2km and find intersecting villages"``
        - ``"Calculate the area of flood hazard zones"``
        - ``"Show roads within 500m of fault lines"``
    provider : str, optional
        ``"gemini"``, ``"openai"``, or ``"anthropic"``.  Auto-detected
        from environment if omitted.
    n_context_layers : int, optional
        Number of RAG results to inject into the prompt (default ``3``).
    auto_execute : bool, optional
        If ``True`` (default), the generated code is executed in a
        sandboxed ``exec()`` with self-correction on failure.
    max_retries : int, optional
        Maximum number of self-correction retries (default ``2``).

    Returns
    -------
    AgentResult
        Dataclass containing the thought process, generated code,
        execution output, and success flag.

    Raises
    ------
    TypeError
        If ``user_query`` is not a non-empty string.
    EnvironmentError
        If no LLM API key is configured.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    # -- Validate input ------------------------------------------------------
    if not isinstance(user_query, str) or not user_query.strip():
        raise TypeError(
            "'user_query' must be a non-empty string, "
            f"got {type(user_query).__name__}: {user_query!r}."
        )

    result = AgentResult(query=user_query)

    # ── STEP 1: RAG Retrieval ───────────────────────────────────────────────
    logger.info("ReAct Step 1 — Querying RAG for: \"%s\"", user_query)
    metadata_context = _build_metadata_context(
        user_query, n_results=n_context_layers
    )

    # ── Guard: Truncate context to prevent context-window bloat ──
    if len(metadata_context) > _MAX_CONTEXT_CHARS:
        logger.warning(
            "RAG context is %d chars (limit %d) — truncating.",
            len(metadata_context),
            _MAX_CONTEXT_CHARS,
        )
        metadata_context = (
            metadata_context[:_MAX_CONTEXT_CHARS]
            + "\n\n… (context truncated to save tokens)"
        )

    result.metadata_context = metadata_context
    logger.info(
        "RAG returned %d chars of context.",
        len(metadata_context),
    )

    # ── STEP 2: Construct system prompt ─────────────────────────────────────
    logger.info("ReAct Step 2 — Building system prompt with metadata context.")
    system_text = _SYSTEM_PROMPT.format(metadata_context=metadata_context)

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=user_query),
    ]

    # ── STEP 3: Initial LLM call ────────────────────────────────────────────
    resolved_provider = provider or _resolve_provider()
    chat_model = _get_chat_model(resolved_provider)

    logger.info(
        "ReAct Step 3 — Calling %s LLM (initial reasoning) …",
        resolved_provider,
    )
    raw_response = _call_llm(chat_model, messages)
    result.raw_llm_responses.append(raw_response)

    thought, code = _parse_response(raw_response)
    result.thought = thought
    result.code = code

    logger.info(
        "Parsed — thought: %d chars, code: %d chars.",
        len(thought),
        len(code),
    )

    # ── STEP 4: Execute (optional) ──────────────────────────────────────────
    if not auto_execute or not code:
        result.success = not auto_execute
        if not code:
            logger.warning("LLM produced no executable code.")
        return result

    logger.info("ReAct Step 4 — Executing generated code in sandbox …")
    output, tb = _execute_code(code)
    result.output = output

    if tb is None:
        result.success = True
        logger.info(
            "✅ Execution succeeded on first attempt.  "
            "Output: %d chars.",
            len(output),
        )
        return result

    # ── STEP 5: Self-correction loop (up to max_retries) ───────────────────
    result.error = tb
    current_code = code

    for retry_num in range(1, max_retries + 1):
        logger.warning(
            "⚠️  Execution failed (retry %d/%d).  Feeding traceback to LLM …",
            retry_num,
            max_retries,
        )

        correction_text = _CORRECTION_PROMPT.format(
            traceback=tb,
            failed_code=current_code,
            retry_num=retry_num,
            max_retries=max_retries,
        )

        # Append the failed exchange to the conversation so the LLM
        # has full context for self-correction.
        messages.append(AIMessage(content=raw_response))
        messages.append(HumanMessage(content=correction_text))

        logger.info(
            "ReAct Step 5.%d — Calling %s LLM for self-correction …",
            retry_num,
            resolved_provider,
        )

        # Guard: If the LLM call itself fails (e.g. rate limit),
        # break the retry loop instead of crashing the whole run.
        try:
            correction_response = _call_llm(chat_model, messages)
        except (RuntimeError, Exception) as llm_err:
            logger.error(
                "LLM call failed during retry %d: %s — aborting retries.",
                retry_num,
                llm_err,
            )
            result.error = (
                f"Self-correction retry {retry_num} aborted: {llm_err}\n\n"
                f"Original error:\n{tb}"
            )
            break

        result.raw_llm_responses.append(correction_response)

        corr_thought, corr_code = _parse_response(correction_response)
        result.correction_thought = corr_thought
        result.correction_code = corr_code
        result.retry_count = retry_num

        if not corr_code:
            logger.error(
                "Self-correction retry %d produced no executable code.",
                retry_num,
            )
            continue

        logger.info(
            "Executing corrected code (retry %d) …", retry_num
        )
        corr_output, corr_tb = _execute_code(corr_code)
        result.output = corr_output

        if corr_tb is None:
            # ── Success on retry! ──
            result.success = True
            result.code = corr_code
            result.thought = corr_thought or result.thought
            result.error = None
            logger.info(
                "✅ Corrected execution succeeded on retry %d.",
                retry_num,
            )
            return result

        # Update for next iteration
        result.error = corr_tb
        tb = corr_tb
        current_code = corr_code
        raw_response = correction_response

    # All retries exhausted
    result.success = False
    logger.error(
        "❌ All %d self-correction retries exhausted.  Final error:\n%s",
        max_retries,
        result.error,
    )
    return result
