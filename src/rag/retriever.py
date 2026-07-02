"""
retriever.py
============

Lightweight RAG retriever backed by ChromaDB.

This module indexes JSON metadata files that describe spatial layers
(geometry type, columns, CRS, tags …) and exposes a semantic search
function so the LLM agent can resolve natural-language references like
*"where are the fault lines?"* to concrete dataset schemas.

Workflow
--------
1. **Ingest** — ``build_metadata_index()`` reads every ``*.json`` file
   under ``data/metadata/``, flattens each file into a single text
   document, and upserts it into a persistent ChromaDB collection.

2. **Query** — ``get_layer_metadata(query_string)`` performs a
   similarity search against the collection and returns the parsed
   JSON schema(s) of the best-matching layer(s).

ChromaDB is configured to use its built-in default embedding function
(Sentence Transformers ``all-MiniLM-L6-v2``), so **no OpenAI key is
required** for the retriever alone.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Resolve paths relative to the project root (three levels up from this file).
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_METADATA_DIR = _PROJECT_ROOT / "data" / "metadata"
_CHROMA_PERSIST_DIR = _PROJECT_ROOT / ".chromadb"

# ChromaDB collection name used by all functions in this module.
_COLLECTION_NAME = "spatial_layer_metadata"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_metadata_to_text(meta: Dict[str, Any]) -> str:
    """Convert a metadata dict into a single, search-friendly text block.

    The text is a human-readable summary that embeds well for semantic
    search.  It pulls out high-signal fields (layer name, description,
    geometry type, column descriptions, tags) so the embedding model can
    match against diverse natural-language queries.

    Parameters
    ----------
    meta : dict
        Parsed contents of a metadata JSON file.

    Returns
    -------
    str
        Flattened text representation.
    """
    parts: list[str] = []

    if "layer_name" in meta:
        parts.append(f"Layer: {meta['layer_name']}")
    if "description" in meta:
        parts.append(f"Description: {meta['description']}")
    if "geometry_type" in meta:
        parts.append(f"Geometry type: {meta['geometry_type']}")
    if "crs" in meta:
        parts.append(f"CRS: {meta['crs']}")
    if "source" in meta:
        parts.append(f"Source: {meta['source']}")

    # Column descriptions — very useful for query matching.
    columns: Dict[str, Any] = meta.get("columns", {})
    if columns:
        col_lines = []
        for col_name, col_info in columns.items():
            desc = col_info.get("description", col_info.get("dtype", ""))
            col_lines.append(f"  - {col_name}: {desc}")
        parts.append("Columns:\n" + "\n".join(col_lines))

    # Tags — cheap but effective for keyword-level recall.
    tags: list = meta.get("tags", [])
    if tags:
        parts.append(f"Tags: {', '.join(str(t) for t in tags)}")

    return "\n".join(parts)


def _get_chroma_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client rooted at ``_CHROMA_PERSIST_DIR``.

    Returns
    -------
    chromadb.ClientAPI
    """
    _CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(_CHROMA_PERSIST_DIR))
    logger.debug("ChromaDB client initialised at '%s'.", _CHROMA_PERSIST_DIR)
    return client


def _get_or_create_collection(
    client: chromadb.ClientAPI,
) -> chromadb.Collection:
    """Return the spatial-metadata collection, creating it if necessary.

    Parameters
    ----------
    client : chromadb.ClientAPI

    Returns
    -------
    chromadb.Collection
    """
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_metadata_index(
    metadata_dir: Optional[Union[str, pathlib.Path]] = None,
    *,
    force_rebuild: bool = False,
) -> int:
    """Read every JSON file in *metadata_dir* and upsert into ChromaDB.

    Each JSON file is expected to describe a single spatial layer.  The
    file's stem (e.g. ``rivers_meta``) is used as the ChromaDB document
    ID, so re-running this function is idempotent.

    Parameters
    ----------
    metadata_dir : str or Path, optional
        Directory containing ``*.json`` metadata files.  Defaults to
        ``<project_root>/data/metadata``.
    force_rebuild : bool, optional
        If ``True``, delete the existing collection before re-indexing.
        Defaults to ``False`` (upsert mode).

    Returns
    -------
    int
        Number of documents indexed.

    Raises
    ------
    FileNotFoundError
        If *metadata_dir* does not exist.
    ValueError
        If the directory contains no JSON files.
    """
    metadata_dir = pathlib.Path(metadata_dir) if metadata_dir else _METADATA_DIR

    if not metadata_dir.is_dir():
        raise FileNotFoundError(
            f"Metadata directory not found: {metadata_dir}"
        )

    json_files = sorted(metadata_dir.glob("*.json"))
    if not json_files:
        raise ValueError(
            f"No JSON files found in '{metadata_dir}'. "
            "Add at least one layer metadata file before building the index."
        )

    client = _get_chroma_client()

    if force_rebuild:
        try:
            client.delete_collection(_COLLECTION_NAME)
            logger.info("Deleted existing collection '%s'.", _COLLECTION_NAME)
        except Exception:
            pass  # Collection may not exist yet — that's fine.

    collection = _get_or_create_collection(client)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for fp in json_files:
        try:
            raw = fp.read_text(encoding="utf-8")
            meta = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping '%s' — %s", fp.name, exc)
            continue

        doc_id = fp.stem  # e.g. "rivers_meta"
        flat_text = _flatten_metadata_to_text(meta)

        ids.append(doc_id)
        documents.append(flat_text)
        # Store lightweight look-up metadata inside ChromaDB.
        metadatas.append({
            "file_name": fp.name,
            "layer_name": meta.get("layer_name", doc_id),
            "geometry_type": meta.get("geometry_type", "unknown"),
        })

        logger.debug("Prepared '%s' for indexing (%d chars).", doc_id, len(flat_text))

    if not ids:
        raise ValueError(
            "All JSON files in the metadata directory were invalid or unreadable."
        )

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info(
        "Indexed %d metadata document(s) into collection '%s'.",
        len(ids),
        _COLLECTION_NAME,
    )
    return len(ids)


def get_layer_metadata(
    query_string: str,
    *,
    n_results: int = 1,
    metadata_dir: Optional[Union[str, pathlib.Path]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve the JSON schema of the spatial layer best matching a query.

    Performs a cosine-similarity search against the ChromaDB collection
    built by ``build_metadata_index()``.  If the collection is empty or
    does not exist, the index is transparently (re-)built first.

    Parameters
    ----------
    query_string : str
        Natural-language question or keyword phrase, e.g.
        ``"where are the fault lines?"`` or ``"rivers hydrology"``.
    n_results : int, optional
        Maximum number of matching layers to return (default ``1``).
    metadata_dir : str or Path, optional
        Override the default metadata directory (useful for testing).

    Returns
    -------
    list[dict]
        A list of result dicts, each containing:

        - ``layer_name`` (str) — Name of the matched layer.
        - ``file_name`` (str) — Original JSON file name.
        - ``distance`` (float) — Cosine distance (lower = more similar).
        - ``schema`` (dict) — Full parsed JSON content of the metadata
          file.

    Raises
    ------
    TypeError
        If ``query_string`` is not a non-empty string.
    RuntimeError
        If the ChromaDB query fails unexpectedly.

    Examples
    --------
    >>> results = get_layer_metadata("where are the fault lines?")
    >>> results[0]["layer_name"]
    'rivers'
    >>> results[0]["schema"]["geometry_type"]
    'MultiLineString'
    """
    # -- Validate input ------------------------------------------------------
    if not isinstance(query_string, str) or not query_string.strip():
        raise TypeError(
            "'query_string' must be a non-empty string, "
            f"got {type(query_string).__name__}: {query_string!r}."
        )

    metadata_dir = pathlib.Path(metadata_dir) if metadata_dir else _METADATA_DIR

    # -- Ensure the index exists ---------------------------------------------
    client = _get_chroma_client()
    collection = _get_or_create_collection(client)

    if collection.count() == 0:
        logger.info("Collection is empty — building index before querying.")
        build_metadata_index(metadata_dir=metadata_dir)
        # Re-fetch collection handle after upsert.
        collection = _get_or_create_collection(client)

    # -- Query ---------------------------------------------------------------
    logger.info(
        "Querying collection '%s' with: \"%s\" (n_results=%d).",
        _COLLECTION_NAME,
        query_string,
        n_results,
    )

    try:
        query_result = collection.query(
            query_texts=[query_string],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"ChromaDB query failed: {exc}"
        ) from exc

    # -- Build response list -------------------------------------------------
    results: List[Dict[str, Any]] = []

    returned_ids = query_result.get("ids", [[]])[0]
    returned_metas = query_result.get("metadatas", [[]])[0]
    returned_distances = query_result.get("distances", [[]])[0]

    for doc_id, meta, dist in zip(returned_ids, returned_metas, returned_distances):
        file_name = meta.get("file_name", f"{doc_id}.json")
        schema_path = metadata_dir / file_name

        # Load the full JSON schema from disk so the caller gets the
        # complete, unflattened metadata — not just the embedded text.
        full_schema: Dict[str, Any] = {}
        if schema_path.is_file():
            try:
                full_schema = json.loads(
                    schema_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Could not reload schema from '%s': %s", schema_path, exc
                )
        else:
            logger.warning("Schema file not found on disk: '%s'.", schema_path)

        results.append({
            "layer_name": meta.get("layer_name", doc_id),
            "file_name": file_name,
            "distance": round(dist, 6),
            "schema": full_schema,
        })

    logger.info("Returning %d result(s) for query.", len(results))
    return results
