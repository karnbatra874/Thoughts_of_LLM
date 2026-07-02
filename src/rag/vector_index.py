"""
vector_index.py
===============

ChromaDB-backed vector index for the Kamrup spatial dataset.

This module replaces a traditional ML training phase by giving the LLM
**perfect memory** of every local data layer's schema — file path, CRS,
geometry type, column names, dtypes, and sample values.

Pipeline
--------
1. ``build_kamrup_index()`` reads ``data/metadata/metadata.json``,
   optionally enriches it with the data dictionary, then flattens each
   layer into a search-friendly text document and upserts it into a
   persistent ChromaDB collection.

2. ``get_kamrup_context(user_query)`` performs a cosine-similarity
   search and returns structured context dicts that the orchestrator
   can inject directly into the LLM system prompt.

ChromaDB uses its built-in default embedding function (Sentence
Transformers ``all-MiniLM-L6-v2``), so **no OpenAI key is required**
for the retriever alone.

Usage
-----
>>> from src.rag.vector_index import build_kamrup_index, get_kamrup_context
>>> build_kamrup_index()          # one-time index build
>>> ctx = get_kamrup_context("show me flood hazard areas")
>>> ctx[0]["file_name"]
'kamrup_flood_incidents_spatial.geojson'
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_METADATA_JSON = _PROJECT_ROOT / "data" / "metadata" / "metadata.json"
_DATA_DICT_JSON = (
    _PROJECT_ROOT / "data" / "kamrup_synthetic" / "metadata"
    / "kamrup_data_dictionary.json"
)
_CHROMA_PERSIST_DIR = _PROJECT_ROOT / ".chromadb"

_COLLECTION_NAME = "kamrup_spatial_layers"


# ---------------------------------------------------------------------------
# Data-dictionary enrichment map
# ---------------------------------------------------------------------------

def _load_data_dictionary() -> Dict[str, Dict[str, Any]]:
    """Load the hand-authored data dictionary and return a lookup keyed
    by file name.

    Returns
    -------
    dict
        ``{file_name: {description, model_use, ...}, ...}``
        Empty dict if the file doesn't exist or can't be parsed.
    """
    if not _DATA_DICT_JSON.is_file():
        logger.debug("Data dictionary not found at '%s'.", _DATA_DICT_JSON)
        return {}

    try:
        raw = json.loads(_DATA_DICT_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse data dictionary: %s", exc)
        return {}

    lookup: Dict[str, Dict[str, Any]] = {}
    for section_key in ("RASTER", "VECTOR_SYNTHETIC", "VECTOR_REAL", "FLOOD_RECORDS"):
        section = raw.get("layers", {}).get(section_key, {})
        for fname, info in section.items():
            # Normalise keys like "export.json (rivers — reclip needed)"
            clean_name = fname.split(" ")[0] if " " in fname else fname
            lookup[clean_name] = info

    return lookup


# ---------------------------------------------------------------------------
# Text flattening
# ---------------------------------------------------------------------------

def _flatten_layer_to_text(
    layer: Dict[str, Any],
    enrichment: Dict[str, Any] | None = None,
) -> str:
    """Convert a single layer metadata dict into a dense, embedding-friendly
    text block.

    The text is structured so the embedding model can match against
    diverse natural-language queries — e.g. "flood hazard", "river
    buffer", "population exposure", "tectonic faults".

    Parameters
    ----------
    layer : dict
        One element from ``metadata.json["layers"]``.
    enrichment : dict, optional
        Matching entry from the data dictionary with ``description``,
        ``model_use``, ``key_columns`` etc.

    Returns
    -------
    str
    """
    parts: list[str] = []

    # ---- Core identity
    parts.append(f"Layer: {layer['file_name']}")
    parts.append(f"Format: {layer.get('format', 'Unknown')}")
    parts.append(f"File path: {layer.get('file_path', '')}")

    # ---- Geometry
    geom = ", ".join(layer.get("geometry_types", []))
    parts.append(f"Geometry type: {geom}")
    parts.append(f"Feature count: {layer.get('feature_count', '?')}")

    # ---- CRS
    parts.append(f"CRS: {layer.get('crs', 'Unknown')}")
    epsg = layer.get("crs_epsg")
    if epsg:
        parts.append(f"EPSG code: {epsg}")

    # ---- Bounding box
    bbox = layer.get("bounding_box", {})
    if bbox:
        parts.append(
            f"Bounding box: "
            f"({bbox.get('minx')}, {bbox.get('miny')}) to "
            f"({bbox.get('maxx')}, {bbox.get('maxy')})"
        )

    # ---- Columns — most important for RAG matching
    columns = layer.get("attribute_columns", [])
    if columns:
        parts.append(f"Attribute columns: {', '.join(columns)}")

    col_details = layer.get("column_details", {})
    if col_details:
        detail_lines = []
        for col_name, info in col_details.items():
            dtype = info.get("dtype", "")
            sample = info.get("sample", "")
            detail_lines.append(f"  - {col_name} ({dtype}): sample={sample}")
        parts.append("Column details:\n" + "\n".join(detail_lines))

    # ---- Data-dictionary enrichment (description, model_use, tags)
    if enrichment:
        desc = enrichment.get("description", "")
        if desc:
            parts.append(f"Description: {desc}")
        model_use = enrichment.get("model_use", "")
        if model_use:
            parts.append(f"Model use: {model_use}")
        key_cols = enrichment.get("key_columns", {})
        if key_cols:
            kc_lines = [f"  - {k}: {v}" for k, v in key_cols.items()]
            parts.append("Key columns:\n" + "\n".join(kc_lines))
        attrs = enrichment.get("attributes", [])
        if attrs:
            parts.append(f"Semantic attributes: {', '.join(attrs)}")

    # ---- Keyword boosters — inject high-recall synonyms
    fname_lower = layer["file_name"].lower()
    boosters: list[str] = []
    if "flood" in fname_lower:
        boosters.extend(["flood", "inundation", "hazard", "disaster", "monsoon"])
    if "river" in fname_lower:
        boosters.extend(["river", "waterway", "brahmaputra", "stream", "hydrology"])
    if "road" in fname_lower:
        boosters.extend(["road", "highway", "transport", "infrastructure"])
    if "village" in fname_lower:
        boosters.extend(["village", "settlement", "town", "habitation"])
    if "landuse" in fname_lower or "lulc" in fname_lower:
        boosters.extend(["landuse", "land cover", "LULC", "agriculture", "forest", "urban"])
    if "fault" in fname_lower:
        boosters.extend(["fault", "seismic", "earthquake", "tectonic", "geology"])
    if "population" in fname_lower or "pop" in fname_lower:
        boosters.extend(["population", "census", "density", "households", "exposure"])
    if "district" in fname_lower:
        boosters.extend(["district", "boundary", "administrative", "Assam", "Kamrup"])
    if boosters:
        parts.append(f"Related keywords: {', '.join(sorted(set(boosters)))}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ChromaDB helpers — with singleton caching
# ---------------------------------------------------------------------------

# Module-level cache: avoids reloading the embedding model (~2-3s) and
# reconnecting to ChromaDB on every Streamlit rerun or function call.
_CACHED_EMBEDDINGS = None
_CACHED_VECTORSTORE = None


def _get_embeddings():
    """Return a cached SentenceTransformer embedding model (singleton)."""
    global _CACHED_EMBEDDINGS
    if _CACHED_EMBEDDINGS is None:
        logger.info("Loading embedding model 'all-MiniLM-L6-v2' (one-time)…")
        _CACHED_EMBEDDINGS = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    return _CACHED_EMBEDDINGS


def _get_vector_store(*, force_new: bool = False) -> Chroma:
    """Return a cached LangChain Chroma vector store (singleton).

    Parameters
    ----------
    force_new : bool
        If ``True``, discard the cached instance and create a fresh one
        (used after ``force_rebuild`` in ``build_kamrup_index``).
    """
    global _CACHED_VECTORSTORE
    if _CACHED_VECTORSTORE is None or force_new:
        _CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _CACHED_VECTORSTORE = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=_get_embeddings(),
            persist_directory=str(_CHROMA_PERSIST_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _CACHED_VECTORSTORE


# ---------------------------------------------------------------------------
# Public API — Indexing
# ---------------------------------------------------------------------------

def build_kamrup_index(
    metadata_path: Optional[Union[str, pathlib.Path]] = None,
    *,
    force_rebuild: bool = False,
) -> int:
    """Read ``metadata.json`` and upsert every layer into ChromaDB.

    Each layer is assigned a deterministic document ID derived from its
    file name, so re-running this function is **idempotent** (upsert
    semantics).

    Parameters
    ----------
    metadata_path : str or Path, optional
        Override path to the metadata JSON.  Defaults to
        ``data/metadata/metadata.json``.
    force_rebuild : bool, optional
        Delete the existing collection before re-indexing (default
        ``False``).

    Returns
    -------
    int
        Number of layer documents indexed.

    Raises
    ------
    FileNotFoundError
        If the metadata JSON does not exist.
    ValueError
        If the JSON contains no layers.
    """
    meta_path = pathlib.Path(metadata_path) if metadata_path else _METADATA_JSON

    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {meta_path}. "
            "Run scan_spatial_data.py first to generate it."
        )

    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    layers: list[dict] = raw.get("layers", [])

    if not layers:
        raise ValueError("metadata.json contains no layers to index.")

    # Load optional data-dictionary enrichment.
    data_dict = _load_data_dictionary()

    vectorstore = _get_vector_store()

    if force_rebuild:
        try:
            vectorstore.delete_collection()
            logger.info("Deleted existing collection '%s'.", _COLLECTION_NAME)
        except Exception:
            pass
        vectorstore = _get_vector_store(force_new=True)

    documents: list[Document] = []

    for layer in layers:
        fname = layer["file_name"]
        doc_id = fname.replace(".", "_")  # e.g. "rivers_kamrup_geojson"

        enrichment = data_dict.get(fname)
        flat_text = _flatten_layer_to_text(layer, enrichment)

        metadata = {
            "file_name": fname,
            "file_path": layer.get("file_path", ""),
            "format": layer.get("format", "Unknown"),
            "crs": layer.get("crs", "Unknown"),
            "crs_epsg": layer.get("crs_epsg", 0),
            "geometry_types": ", ".join(layer.get("geometry_types", [])),
            "feature_count": layer.get("feature_count", 0),
        }

        documents.append(Document(page_content=flat_text, metadata=metadata, id=doc_id))

        logger.debug(
            "Prepared '%s' (%d chars) for indexing.", doc_id, len(flat_text)
        )

    vectorstore.add_documents(documents=documents, ids=[doc.id for doc in documents])
    logger.info(
        "Indexed %d layer(s) into ChromaDB collection '%s'.",
        len(documents),
        _COLLECTION_NAME,
    )
    return len(documents)


# ---------------------------------------------------------------------------
# Public API — Retrieval
# ---------------------------------------------------------------------------

def get_kamrup_context(
    user_query: str,
    *,
    n_results: int = 3,
    metadata_path: Optional[Union[str, pathlib.Path]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve spatial layer context most relevant to a user's question.

    Performs a cosine-similarity search against the ChromaDB collection
    built by ``build_kamrup_index()``.  If the collection is empty, the
    index is transparently built first.

    Parameters
    ----------
    user_query : str
        Natural-language question, e.g.
        ``"which layers have flood data?"`` or
        ``"show me the road network"``.
    n_results : int, optional
        Maximum number of matching layers to return (default ``3``).
    metadata_path : str or Path, optional
        Override path to ``metadata.json`` (used during auto-build).

    Returns
    -------
    list[dict]
        Each dict contains:

        - ``file_name`` (str) — e.g. ``"rivers_kamrup.geojson"``
        - ``file_path`` (str) — relative path from project root
        - ``format`` (str) — ``"GeoJSON"`` or ``"Shapefile"``
        - ``crs`` (str) — e.g. ``"EPSG:4326"``
        - ``crs_epsg`` (int) — e.g. ``4326``
        - ``geometry_types`` (str) — e.g. ``"LineString"``
        - ``feature_count`` (int) — number of features in the layer
        - ``distance`` (float) — cosine distance (lower = better match)
        - ``matched_text`` (str) — the embedded document text that was
          matched, useful for debugging prompt construction
        - ``layer_schema`` (dict) — full layer dict from metadata.json

    Raises
    ------
    TypeError
        If ``user_query`` is not a non-empty string.
    RuntimeError
        If the ChromaDB query fails.

    Examples
    --------
    >>> results = get_kamrup_context("where are the flood hazard zones?")
    >>> results[0]["file_name"]
    'kamrup_flood_incidents_spatial.geojson'
    >>> results[0]["crs"]
    'EPSG:4326'
    """
    if not isinstance(user_query, str) or not user_query.strip():
        raise TypeError(
            "'user_query' must be a non-empty string, "
            f"got {type(user_query).__name__}: {user_query!r}."
        )

    meta_path = pathlib.Path(metadata_path) if metadata_path else _METADATA_JSON

    # ---- Ensure index exists -----------------------------------------------
    vectorstore = _get_vector_store()

    if vectorstore._collection.count() == 0:
        logger.info("Collection is empty — building index automatically.")
        build_kamrup_index(metadata_path=meta_path)
        vectorstore = _get_vector_store()

    # ---- Load full metadata for schema return ------------------------------
    try:
        full_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        schema_lookup: Dict[str, dict] = {
            layer["file_name"]: layer
            for layer in full_meta.get("layers", [])
        }
    except Exception:
        schema_lookup = {}

    # ---- Query -------------------------------------------------------------
    effective_n = min(n_results, vectorstore._collection.count())

    logger.info(
        "Querying '%s' with: \"%s\" (n_results=%d).",
        _COLLECTION_NAME,
        user_query,
        effective_n,
    )

    try:
        results_with_scores = vectorstore.similarity_search_with_score(
            query=user_query,
            k=effective_n,
        )
    except Exception as exc:
        raise RuntimeError(f"ChromaDB query failed: {exc}") from exc

    # ---- Build response ----------------------------------------------------
    results: List[Dict[str, Any]] = []

    for doc, score in results_with_scores:
        meta = doc.metadata
        fname = meta.get("file_name", "unknown")
        results.append({
            "file_name": fname,
            "file_path": meta.get("file_path", ""),
            "format": meta.get("format", "Unknown"),
            "crs": meta.get("crs", "Unknown"),
            "crs_epsg": meta.get("crs_epsg", 0),
            "geometry_types": meta.get("geometry_types", ""),
            "feature_count": meta.get("feature_count", 0),
            "distance": round(score, 6),
            "matched_text": doc.page_content,
            "layer_schema": schema_lookup.get(fname, {}),
        })

    logger.info("Returning %d result(s) for query.", len(results))
    return results


# ---------------------------------------------------------------------------
# Convenience: human-readable context string for LLM prompts
# ---------------------------------------------------------------------------

def format_context_for_prompt(
    results: List[Dict[str, Any]],
    *,
    include_columns: bool = True,
    include_samples: bool = True,
) -> str:
    """Format retrieval results into a text block suitable for injection
    into an LLM system prompt.

    Parameters
    ----------
    results : list[dict]
        Output of ``get_kamrup_context()``.
    include_columns : bool
        Include the full column list (default ``True``).
    include_samples : bool
        Include sample values for each column (default ``True``).

    Returns
    -------
    str
        Markdown-formatted context block.
    """
    if not results:
        return "(No matching layers found in the spatial data catalogue.)"

    sections: list[str] = []

    for i, r in enumerate(results, 1):
        lines = [
            f"### Layer {i}: {r['file_name']}",
            f"- **File path**: `{r['file_path']}`",
            f"- **Format**: {r['format']}",
            f"- **Geometry**: {r['geometry_types']}",
            f"- **CRS**: {r['crs']} (EPSG:{r['crs_epsg']})",
            f"- **Feature count**: {r['feature_count']}",
            f"- **Match confidence**: {1 - r['distance']:.1%}",
        ]

        schema = r.get("layer_schema", {})
        if include_columns and schema:
            cols = schema.get("attribute_columns", [])
            if cols:
                lines.append(f"- **Columns**: `{'`, `'.join(cols)}`")

            if include_samples:
                col_details = schema.get("column_details", {})
                if col_details:
                    lines.append("- **Column details**:")
                    for col_name, info in col_details.items():
                        dtype = info.get("dtype", "")
                        sample = info.get("sample", "")
                        lines.append(
                            f"  - `{col_name}` ({dtype}) — sample: `{sample}`"
                        )

        bbox = schema.get("bounding_box", {})
        if bbox:
            lines.append(
                f"- **Bounding box**: "
                f"({bbox.get('minx')}, {bbox.get('miny')}) to "
                f"({bbox.get('maxx')}, {bbox.get('maxy')})"
            )

        sections.append("\n".join(lines))

    return "\n\n---\n\n".join(sections)
