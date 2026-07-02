"""
app.py
======

Streamlit front-end for the Agentic GIS Orchestrator.

Design adapted from: simple_agentic_gis_dashboard.html
Layout: Clean 2-column grid inspired by the user's HTML reference.

┌────────────────────────┬─────────────────────────┐
│  LEFT COLUMN           │  RIGHT COLUMN           │
│  1. Spatial Request    │  3. Map Output Preview  │
│  2. Agent Reasoning    │  4. Generated Script    │
│     (Chain-of-Thought) │  5. GeoJSON Export      │
└────────────────────────┴─────────────────────────┘

Run with: python -m streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import pathlib
import textwrap
from datetime import datetime
from typing import Optional

import folium
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so relative imports resolve.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent.orchestrator import run_agent, AgentResult  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Page configuration
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Agentic GIS Orchestrator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Caching: Prevent ChromaDB and Embeddings from reloading on every interaction
# ---------------------------------------------------------------------------
from src.rag import vector_index

# Save original functions before patching to avoid infinite recursion
_orig_get_embeddings = vector_index._get_embeddings
_orig_get_vector_store = vector_index._get_vector_store

@st.cache_resource
def get_cached_embeddings():
    """Cache the heavy SentenceTransformer model across Streamlit reruns."""
    return _orig_get_embeddings()

@st.cache_resource
def get_cached_vector_store():
    """Cache the ChromaDB connection across Streamlit reruns."""
    return _orig_get_vector_store()

# Monkey-patch the RAG system to use our Streamlit-cached versions
vector_index._get_embeddings = get_cached_embeddings
vector_index._get_vector_store = get_cached_vector_store

# Note: We intentionally do NOT pre-warm the cache here at the top level.
# Pre-warming blocks the Streamlit UI from rendering on the very first load
# (which can take a while if the embedding model weights are downloading).
# Instead, it will load lazily inside the st.spinner() when the user runs 
# their first query.

# ═══════════════════════════════════════════════════════════════════════════
# CSS — Slate dark theme matching the HTML reference
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    /* ---------- Google Fonts ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ---------- Root tokens ---------- */
    :root {
        --bg-body:       #0f172a;
        --bg-surface:    #1e293b;
        --bg-card:       #1e293b;
        --bg-input:      #0f172a;
        --border-color:  #334155;
        --border-focus:  #3b82f6;
        --accent-blue:   #3b82f6;
        --accent-cyan:   #22d3ee;
        --accent-green:  #34d399;
        --accent-emerald:#10b981;
        --accent-red:    #ef4444;
        --accent-amber:  #f59e0b;
        --text-white:    #f8fafc;
        --text-primary:  #e2e8f0;
        --text-muted:    #94a3b8;
        --text-dim:      #64748b;
        --text-dark:     #475569;
        --font-sans:     'Inter', -apple-system, system-ui, sans-serif;
        --font-mono:     'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    }

    /* ---------- Global reset ---------- */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: var(--bg-body) !important;
        color: var(--text-primary);
        font-family: var(--font-sans);
    }

    header[data-testid="stHeader"] {
        background: var(--bg-surface) !important;
        border-bottom: 1px solid var(--border-color);
    }

    /* ---------- Card ---------- */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }

    .card-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-white);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* ---------- Log entry ---------- */
    .log-entry {
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(51, 65, 85, 0.5);
        font-size: 0.75rem;
        line-height: 1.6;
        font-family: var(--font-mono);
    }
    .log-entry:last-child { border-bottom: none; }
    .log-status {
        color: var(--accent-blue);
        font-weight: 600;
    }
    .log-text {
        color: var(--text-primary);
        margin-left: 0.25rem;
    }

    /* ---------- Code block ---------- */
    .code-wrapper {
        background: var(--bg-input);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
    }
    .code-header {
        background: var(--bg-card);
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid var(--border-color);
        font-size: 0.625rem;
        font-family: var(--font-mono);
        color: var(--text-dim);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .code-body {
        padding: 1rem;
        font-family: var(--font-mono);
        font-size: 0.75rem;
        line-height: 1.7;
        color: #93c5fd;
        overflow-x: auto;
        white-space: pre;
        max-height: 380px;
        overflow-y: auto;
    }

    /* ---------- Thought block ---------- */
    .thought-block {
        background: rgba(59, 130, 246, 0.06);
        border-left: 3px solid var(--accent-blue);
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.78rem;
        line-height: 1.65;
        color: var(--text-primary);
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* ---------- Error block ---------- */
    .error-block {
        background: rgba(239, 68, 68, 0.06);
        border-left: 3px solid var(--accent-red);
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        line-height: 1.6;
        color: #fca5a5;
        white-space: pre-wrap;
        max-height: 200px;
        overflow-y: auto;
    }

    /* ---------- Output block ---------- */
    .output-block {
        background: rgba(52, 211, 153, 0.05);
        border-left: 3px solid var(--accent-green);
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-family: var(--font-mono);
        font-size: 0.75rem;
        line-height: 1.6;
        color: var(--accent-green);
        white-space: pre-wrap;
        max-height: 200px;
        overflow-y: auto;
    }

    /* ---------- Status badge ---------- */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        letter-spacing: 0.03em;
    }
    .badge-online {
        background: rgba(16, 185, 129, 0.15);
        color: var(--accent-emerald);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-success {
        background: rgba(52, 211, 153, 0.12);
        color: var(--accent-green);
        border: 1px solid rgba(52, 211, 153, 0.25);
    }
    .badge-error {
        background: rgba(239, 68, 68, 0.12);
        color: var(--accent-red);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .badge-idle {
        background: rgba(148, 163, 184, 0.10);
        color: var(--text-muted);
        border: 1px solid rgba(148, 163, 184, 0.18);
    }
    .badge-thinking {
        background: rgba(59, 130, 246, 0.12);
        color: var(--accent-blue);
        border: 1px solid rgba(59, 130, 246, 0.25);
        animation: pulse 1.5s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* ---------- Map container ---------- */
    .map-container {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }

    /* ---------- Map legend ---------- */
    .map-legend {
        background: rgba(30, 41, 59, 0.92);
        backdrop-filter: blur(8px);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 0.6rem 0.8rem;
        font-size: 0.68rem;
        color: var(--text-muted);
        margin-top: 0.5rem;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-bottom: 0.25rem;
    }
    .legend-item:last-child { margin-bottom: 0; }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ---------- Metadata block ---------- */
    .meta-block {
        background: rgba(34, 211, 238, 0.04);
        border-left: 3px solid var(--accent-cyan);
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-family: var(--font-mono);
        font-size: 0.7rem;
        line-height: 1.6;
        color: var(--text-muted);
        overflow-x: auto;
        white-space: pre;
        max-height: 200px;
        overflow-y: auto;
    }

    /* ---------- Streamlit overrides ---------- */
    .stChatInput > div {
        border-radius: 8px !important;
        border: 1px solid var(--border-color) !important;
        background: var(--bg-input) !important;
    }
    .stChatInput > div:focus-within {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15) !important;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        background: var(--bg-card) !important;
    }

    div[data-testid="stDownloadButton"] > button {
        background: var(--accent-blue) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: var(--font-sans) !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        padding: 0.55rem 1.1rem !important;
        transition: background 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: #2563eb !important;
    }

    /* Streamlit button overrides for example queries */
    div[data-testid="stButton"] > button {
        background: transparent !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-muted) !important;
        font-size: 0.78rem !important;
        text-align: left !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.8rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: var(--accent-blue) !important;
        color: var(--text-white) !important;
        background: rgba(59, 130, 246, 0.06) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-surface); }
    ::-webkit-scrollbar-thumb { background: var(--text-dark); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULTS: dict = {
    "agent_result": None,
    "query_history": [],
    "run_count": 0,
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def _html(content: str) -> None:
    """Render raw HTML."""
    st.markdown(content, unsafe_allow_html=True)


def _escape(text: str) -> str:
    """HTML-escape text for safe rendering."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _try_extract_geojson(result: AgentResult) -> Optional[dict]:
    """Parse agent output as GeoJSON FeatureCollection / Feature."""
    if not result.output:
        return None
    try:
        parsed = json.loads(result.output)
        if isinstance(parsed, dict) and parsed.get("type") in (
            "FeatureCollection", "Feature", "GeometryCollection",
        ):
            return parsed
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def _try_extract_geodataframe(result: AgentResult):
    """Best-effort extraction of a GeoDataFrame from agent output."""
    geojson = _try_extract_geojson(result)
    if geojson is None:
        return None
    try:
        return gpd.GeoDataFrame.from_features(
            geojson.get("features", [geojson]),
            crs="EPSG:4326",
        )
    except Exception:
        return None


def _build_provenance_script(result: AgentResult) -> str:
    """Build a full reproducible Python script from agent-generated code."""
    code_body = result.correction_code or result.code or "# (no code generated)"
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"
        Auto-generated GIS Pipeline Script
        ====================================
        Query : {result.query}
        Date  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        Produced by the Agentic GIS Orchestrator.
        Fully reproducible — install deps and run directly.
        \"\"\"

        import json
        import geopandas as gpd
        from pathlib import Path
        from src.tools import geoprocessing

        PROJECT_ROOT = Path(__file__).resolve().parent

        # ── Pipeline ──────────────────────────────────────────
        {textwrap.indent(code_body, "        ").strip()}
    """)


def _render_status_badge(result: AgentResult | None) -> str:
    if result is None:
        return '<span class="badge badge-idle">● Idle</span>'
    if result.success:
        return '<span class="badge badge-success">✓ Success</span>'
    return '<span class="badge badge-error">✗ Error</span>'


# ═══════════════════════════════════════════════════════════════════════════
# Map renderer
# ═══════════════════════════════════════════════════════════════════════════

def _render_map(result: AgentResult) -> None:
    """Render an interactive Folium map with layered styling."""
    # Kamrup center as default (26.14°N, 91.74°E)
    m = folium.Map(
        location=[26.14, 91.74],
        zoom_start=10,
        tiles="CartoDB dark_matter",
    )

    gdf = _try_extract_geodataframe(result)
    has_lines = False
    has_points = False
    has_polygons = False

    if gdf is not None and not gdf.empty:
        if gdf.crs and not gdf.crs.equals("EPSG:4326"):
            gdf = gdf.to_crs(epsg=4326)

        points = gdf[gdf.geometry.geom_type.isin(["Point", "MultiPoint"])]
        lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        polygons = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

        # Lines — cyan
        if not lines.empty:
            has_lines = True
            fg = folium.FeatureGroup(name="🌊 Lines")
            for _, row in lines.iterrows():
                props = {c: row[c] for c in lines.columns if c != "geometry"}
                popup_rows = "".join(
                    f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
                    for k, v in list(props.items())[:8]
                    if v is not None
                )
                gj = folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda _: {
                        "color": "#22d3ee", "weight": 3, "opacity": 0.9,
                    },
                )
                gj.add_child(folium.Popup(
                    f'<table style="font-size:11px">{popup_rows}</table>',
                    max_width=280,
                ))
                gj.add_child(folium.Tooltip(
                    str(props.get("name", props.get("waterway", "Line")))
                ))
                fg.add_child(gj)
            fg.add_to(m)

        # Points — red circle markers
        if not points.empty:
            has_points = True
            fg = folium.FeatureGroup(name="📍 Points")
            for _, row in points.iterrows():
                props = {c: row[c] for c in points.columns if c != "geometry"}
                centroid = row.geometry.centroid
                popup_rows = "".join(
                    f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
                    for k, v in list(props.items())[:8]
                    if v is not None
                )
                folium.CircleMarker(
                    location=[centroid.y, centroid.x],
                    radius=7,
                    color="#ef4444",
                    fill=True,
                    fill_color="#ef4444",
                    fill_opacity=0.8,
                    popup=folium.Popup(
                        f'<table style="font-size:11px">{popup_rows}</table>',
                        max_width=280,
                    ),
                    tooltip=str(props.get("name", props.get("amenity", "Point"))),
                ).add_to(fg)
            fg.add_to(m)

        # Polygons — semi-transparent blue
        if not polygons.empty:
            has_polygons = True
            fg = folium.FeatureGroup(name="🔷 Polygons")
            for _, row in polygons.iterrows():
                props = {c: row[c] for c in polygons.columns if c != "geometry"}
                popup_rows = "".join(
                    f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
                    for k, v in list(props.items())[:8]
                    if v is not None
                )
                gj = folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda _: {
                        "color": "#3b82f6", "weight": 1.5,
                        "fillColor": "#3b82f6", "fillOpacity": 0.18,
                    },
                )
                gj.add_child(folium.Popup(
                    f'<table style="font-size:11px">{popup_rows}</table>',
                    max_width=280,
                ))
                gj.add_child(folium.Tooltip("Polygon / Buffer"))
                fg.add_child(gj)
            fg.add_to(m)

        # Fit bounds
        bounds = gdf.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    folium.LayerControl(collapsed=False).add_to(m)

    _html('<div class="map-container">')
    st_folium(m, use_container_width=True, height=380, returned_objects=[])
    _html("</div>")

    # Dynamic legend
    if gdf is not None and not gdf.empty:
        legend_items = []
        if has_lines:
            legend_items.append(
                '<div class="legend-item">'
                '<div class="legend-dot" style="background:#22d3ee"></div>'
                'Lines / Rivers</div>'
            )
        if has_points:
            legend_items.append(
                '<div class="legend-item">'
                '<div class="legend-dot" style="background:#ef4444"></div>'
                'Points / Facilities</div>'
            )
        if has_polygons:
            legend_items.append(
                '<div class="legend-item">'
                '<div class="legend-dot" style="background:#3b82f6"></div>'
                'Polygons / Buffers</div>'
            )
        if legend_items:
            _html(f'<div class="map-legend">{"".join(legend_items)}</div>')


# ═══════════════════════════════════════════════════════════════════════════
# LAYOUT — Sidebar & Header
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Settings")
    st.session_state.mock_mode = st.toggle(
        "🧪 Enable Mock Mode (No API)", 
        value=st.session_state.get("mock_mode", False),
        help="Use pre-saved agent responses to design the UI without burning API quota."
    )
    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.75rem; color:var(--text-dim);'>"
        "When Mock Mode is enabled, the app uses dummy data and bypasses the LLM."
        "</span>", 
        unsafe_allow_html=True
    )

_html("""
<div style="
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border-color);
    padding: 1rem 1.5rem;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
">
    <div>
        <h1 style="
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-white);
            margin: 0;
            letter-spacing: -0.01em;
        ">Agentic GIS Orchestrator</h1>
        <p style="
            font-size: 0.75rem;
            color: var(--text-dim);
            margin: 0.2rem 0 0 0;
        ">Chain-of-Thought LLM System for Spatial Analysis</p>
    </div>
    <span class="badge badge-online">System Online</span>
</div>
""")


# ═══════════════════════════════════════════════════════════════════════════
# LAYOUT — 2-column grid (matching the HTML reference)
# ═══════════════════════════════════════════════════════════════════════════

col_left, col_right = st.columns([1, 1], gap="large")


# ── LEFT COLUMN ─────────────────────────────────────────────────────────

with col_left:

    # ── 1. Spatial Request ──
    _html('<div class="card">')
    _html('<div class="card-title">1. Spatial Request</div>')

    user_query = st.chat_input(
        placeholder="e.g., Calculate buffer zones for pipelines overlaying structural fault-lines in UTM Sector 46N...",
        key="spatial_query_input",
    )

    # Example buttons
    _html('<div style="margin-top: 0.75rem;">')
    examples = [
        "Find flood incidents within 2km of rivers in Kamrup",
        "Buffer all fault lines by 1.5km and intersect with roads",
        "Calculate the area of all landuse polygons in Kamrup",
        "Show population grid cells with flood exposure > 50%",
    ]
    for ex in examples:
        if st.button(f"▸ {ex}", key=f"ex_{hash(ex)}", use_container_width=True):
            user_query = ex
    _html('</div>')
    _html('</div>')  # end card

    # ── Process query ──
    if user_query:
        st.session_state.run_count += 1
        st.session_state.query_history.append(user_query)

        if st.session_state.mock_mode:
            import time
            with st.spinner("🧪 Mock Mode: Simulating Agent Processing..."):
                time.sleep(1)
                
                # Create a mock polygon as output for the map
                import shapely.geometry
                mock_poly = shapely.geometry.box(91.70, 26.10, 91.80, 26.20)
                mock_gdf = gpd.GeoDataFrame([{'name': 'Mock Buffer Zone', 'geometry': mock_poly}], crs="EPSG:4326")
                
                agent_result = AgentResult(
                    query=user_query,
                    metadata_context="Layer: dummy_layer.shp\nGeometry: Polygon\nCRS: EPSG:4326",
                    thought="[MOCK MODE]\n1. Retrieving layers...\n2. Reprojecting to UTM 46N...\n3. Executing buffer...",
                    code="import geopandas as gpd\nprint('Executing mock geoprocessing script...')\n",
                    output=mock_gdf.to_json(),
                    success=True,
                )
        else:
            with st.spinner("🧠 Agent is thinking …"):
                try:
                    agent_result = run_agent(user_query, auto_execute=True)
                except EnvironmentError as e:
                    agent_result = AgentResult(
                        query=user_query,
                        thought="(Agent could not run — see error below.)",
                        error=str(e),
                        success=False,
                    )
                except Exception as e:
                    agent_result = AgentResult(
                        query=user_query,
                        thought="(Unexpected error during agent execution.)",
                        error=str(e),
                        success=False,
                    )
        st.session_state.agent_result = agent_result

    result: AgentResult | None = st.session_state.agent_result

    # ── 2. Agent Reasoning (Chain-of-Thought) ──
    _html('<div class="card" style="min-height: 340px;">')
    _html(
        f'<div class="card-title">'
        f'<span>2. Agent Reasoning (Chain-of-Thought)</span>'
        f'{_render_status_badge(result)}'
        f'</div>'
    )

    if result is None:
        _html(
            '<div style="'
            'background: var(--bg-input);'
            'border: 1px solid var(--border-color);'
            'border-radius: 8px;'
            'padding: 2rem 1rem;'
            'text-align: center;'
            'font-family: var(--font-mono);'
            'font-size: 0.75rem;'
            'color: var(--text-dim);'
            'font-style: italic;'
            '">Agent logs will appear here...</div>'
        )
    else:
        _html(
            '<div style="'
            'background: var(--bg-input);'
            'border: 1px solid var(--border-color);'
            'border-radius: 8px;'
            'padding: 1rem;'
            'max-height: 500px;'
            'overflow-y: auto;'
            '">'
        )

        # Build step-by-step log entries (mimicking the HTML design)
        logs = []

        # Log: Query
        logs.append(("User Query:", _escape(result.query)))

        # Log: RAG context
        if result.metadata_context and "No metadata" not in result.metadata_context:
            # Extract layer names from context
            context_preview = result.metadata_context[:200].replace('\n', ' ')
            logs.append(("RAG Context Found:", _escape(context_preview) + "…"))
        else:
            logs.append(("RAG Context:", "No matching layers found."))

        # Log: Thought
        if result.thought:
            thought_preview = result.thought[:300]
            if len(result.thought) > 300:
                thought_preview += "…"
            logs.append(("Reasoning:", _escape(thought_preview)))

        # Log: Code generated
        if result.code:
            logs.append(("Code Generated:", f"{len(result.code)} characters of Python"))

        # Log: Execution
        if result.success:
            logs.append(("Execution:", "✓ Code executed successfully"))
        elif result.error:
            logs.append(("Execution Error:", "Self-correction attempted"))

        # Log: Self-correction
        if result.correction_code:
            if result.success:
                logs.append((
                    "Self-Correction:",
                    f"✓ Fixed on retry {result.retry_count}"
                ))
            else:
                logs.append(("Self-Correction:", "✗ All retries exhausted"))

        for status, text in logs:
            _html(
                f'<div class="log-entry">'
                f'<span class="log-status">{status}</span>'
                f'<span class="log-text">{text}</span>'
                f'</div>'
            )

        _html('</div>')  # end log container

        # Expandable: Full thought block
        if result.thought:
            with st.expander("💭 Full Chain-of-Thought", expanded=False):
                _html(f'<div class="thought-block">{_escape(result.thought)}</div>')

        # Expandable: RAG metadata
        if result.metadata_context:
            with st.expander("📂 Retrieved Metadata Context", expanded=False):
                _html(
                    f'<div class="meta-block">'
                    f'{_escape(result.metadata_context)}</div>'
                )

        # Expandable: Self-correction details
        if result.correction_thought or result.correction_code:
            with st.expander("🔄 Self-Correction Details", expanded=False):
                if result.correction_thought:
                    _html(
                        f'<div class="thought-block">'
                        f'{_escape(result.correction_thought)}</div>'
                    )
                if result.correction_code:
                    _html(
                        f'<div class="code-wrapper"><div class="code-header">'
                        f'corrected_code.py</div>'
                        f'<div class="code-body">'
                        f'{_escape(result.correction_code)}</div></div>'
                    )

        # Error output
        if result.error and not result.success:
            _html(f'<div class="error-block">{_escape(result.error)}</div>')

    _html('</div>')  # end card


# ── RIGHT COLUMN ────────────────────────────────────────────────────────

with col_right:

    # ── 3. Map Output Preview ──
    _html('<div class="card">')
    _html('<div class="card-title">3. Map Output Preview</div>')

    if result is not None and result.success:
        _render_map(result)
    else:
        # Default placeholder map (Kamrup center)
        m_default = folium.Map(
            location=[26.14, 91.74],
            zoom_start=10,
            tiles="CartoDB dark_matter",
        )
        folium.LayerControl(collapsed=False).add_to(m_default)
        _html('<div class="map-container">')
        st_folium(
            m_default,
            use_container_width=True,
            height=380,
            returned_objects=[],
        )
        _html("</div>")
        _html(
            '<div class="map-legend">'
            '<div style="font-style: italic;">Awaiting execution...</div>'
            '</div>'
        )

    _html('</div>')  # end card

    # ── 4. Generated Python Script ──
    _html('<div class="card" style="min-height: 300px;">')
    _html('<div class="card-title">4. Generated Python Script</div>')

    if result is not None and result.code:
        final_code = result.correction_code or result.code
        _html(
            f'<div class="code-wrapper">'
            f'<div class="code-header">'
            f'<span>orchestration_output.py</span>'
            f'<span style="color: var(--accent-green);">'
            f'{len(final_code)} chars</span>'
            f'</div>'
            f'<div class="code-body">{_escape(final_code)}</div>'
            f'</div>'
        )

        # Execution output (if not GeoJSON — avoid dumping raw JSON)
        if result.output and not _try_extract_geojson(result):
            _html(
                '<div style="margin-top: 0.75rem;">'
                '<div style="font-size: 0.68rem; font-weight: 600; '
                'text-transform: uppercase; letter-spacing: 0.08em; '
                'color: var(--text-dim); margin-bottom: 0.3rem;">'
                'Execution Output</div>'
            )
            _html(f'<div class="output-block">{_escape(result.output)}</div>')
            _html('</div>')

        # Code provenance (expandable)
        with st.expander("📋 Full Reproducible Script", expanded=False):
            provenance = _build_provenance_script(result)
            _html(
                f'<div class="code-wrapper">'
                f'<div class="code-header">pipeline_script.py</div>'
                f'<div class="code-body">{_escape(provenance)}</div>'
                f'</div>'
            )
    else:
        _html(
            '<div class="code-wrapper">'
            '<div class="code-header">orchestration_output.py</div>'
            '<div class="code-body" style="color: var(--text-dim);">'
            '# Python geoprocessing code will appear here...</div>'
            '</div>'
        )

    _html('</div>')  # end card

    # ── 5. GeoJSON Export ──
    _html('<div class="card">')
    _html('<div class="card-title">5. Export</div>')

    if result is not None:
        geojson_data = _try_extract_geojson(result)
        if geojson_data is not None:
            geojson_str = json.dumps(geojson_data, indent=2, ensure_ascii=False)
            feature_count = len(geojson_data.get("features", []))
            _html(
                f'<div style="font-size: 0.75rem; color: var(--text-muted); '
                f'margin-bottom: 0.5rem;">'
                f'GeoJSON FeatureCollection — {feature_count} features</div>'
            )
            st.download_button(
                label="⬇️ Download GeoJSON Result",
                data=geojson_str,
                file_name="agent_result.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )
        else:
            _html(
                '<div style="font-size: 0.75rem; color: var(--text-dim);">'
                'No GeoJSON output available for this result.</div>'
            )
    else:
        _html(
            '<div style="font-size: 0.75rem; color: var(--text-dim);">'
            'Run a query to generate downloadable GeoJSON.</div>'
        )

    _html('</div>')  # end card


# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════

_html("""
<div style="
    text-align: center;
    padding: 1rem 0 1.5rem 0;
    margin-top: 1rem;
    border-top: 1px solid var(--border-color);
">
    <span style="font-size: 0.68rem; color: var(--text-dim);">
        Agentic GIS Orchestrator · Chain-of-Thought Spatial AI ·
        Built with LangChain, ChromaDB, GeoPandas &amp; Folium
    </span>
</div>
""")
