"""Quick smoke test for the Kamrup RAG vector index."""

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.rag.vector_index import (
    build_kamrup_index,
    get_kamrup_context,
    format_context_for_prompt,
)

SEPARATOR = "=" * 60

# ---- Step 1: Build index ---------------------------------------------------
print(SEPARATOR)
print("  STEP 1: Building ChromaDB index from metadata.json")
print(SEPARATOR)

n = build_kamrup_index(force_rebuild=True)
print(f"  Indexed {n} layers.\n")

# ---- Step 2: Test queries --------------------------------------------------
test_queries = [
    "show me flood hazard areas",
    "where are the rivers and waterways?",
    "I need road network data",
    "population density and exposure",
    "tectonic fault lines near Kamrup",
    "land use and agriculture patterns",
    "district boundaries of Assam",
]

for query in test_queries:
    print(SEPARATOR)
    print(f"  QUERY: \"{query}\"")
    print(SEPARATOR)

    results = get_kamrup_context(query, n_results=2)

    for i, r in enumerate(results, 1):
        dist_pct = f"{(1 - r['distance']) * 100:.1f}%"
        print(f"  [{i}] {r['file_name']}")
        print(f"      Format     : {r['format']}")
        print(f"      Geometry   : {r['geometry_types']}")
        print(f"      CRS        : {r['crs']}")
        print(f"      Features   : {r['feature_count']}")
        print(f"      Confidence : {dist_pct}")
        cols = r.get("layer_schema", {}).get("attribute_columns", [])
        if cols:
            print(f"      Columns    : {', '.join(cols[:8])}{'...' if len(cols) > 8 else ''}")
        print()

# ---- Step 3: Show formatted prompt context ---------------------------------
print(SEPARATOR)
print("  STEP 3: Formatted LLM prompt context for 'floods near rivers'")
print(SEPARATOR)

results = get_kamrup_context("floods near rivers", n_results=2)
prompt_ctx = format_context_for_prompt(results, include_samples=False)
print(prompt_ctx)
print()
print(SEPARATOR)
print("  ALL TESTS PASSED")
print(SEPARATOR)
