"""
scan_spatial_data.py
====================

Temporary utility that scans the /data/kamrup_synthetic/ directory tree
for Shapefiles (.shp) and GeoJSON (.geojson / .json with FeatureCollection)
files, prints a detailed report, and writes a consolidated metadata.json
into /data/metadata/ for the RAG system.

Run:  python scan_spatial_data.py
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
from collections import OrderedDict

import geopandas as gpd

# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "kamrup_synthetic"
METADATA_OUT = PROJECT_ROOT / "data" / "metadata"
METADATA_FILE = METADATA_OUT / "metadata.json"

# Subdirectories to scan (recurse all of them)
SCAN_DIRS = [
    DATA_DIR / "vector",
    DATA_DIR / "flood_records",
]

# Supported file extensions for spatial vector data
SPATIAL_EXTS = {".shp", ".geojson"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_geojson_json(path: pathlib.Path) -> bool:
    """Return True if a .json file looks like a GeoJSON FeatureCollection."""
    if path.suffix.lower() != ".json":
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            # Read just the first 500 chars to sniff the type field.
            head = f.read(500)
        return '"FeatureCollection"' in head or '"Feature"' in head
    except Exception:
        return False


def _geometry_type_label(geom_type: str | None) -> str:
    """Normalise geometry type names for the report."""
    if geom_type is None:
        return "Unknown"
    mapping = {
        "Point": "Point",
        "MultiPoint": "MultiPoint",
        "LineString": "LineString",
        "MultiLineString": "MultiLineString",
        "Polygon": "Polygon",
        "MultiPolygon": "MultiPolygon",
        "GeometryCollection": "GeometryCollection",
    }
    return mapping.get(geom_type, geom_type)


def scan_file(filepath: pathlib.Path) -> dict | None:
    """Read a spatial vector file and return a metadata dict."""
    try:
        gdf = gpd.read_file(filepath)
    except Exception as exc:
        print(f"  ⚠  Could not read '{filepath.name}': {exc}")
        return None

    # Determine geometry types present in the data.
    if gdf.empty or gdf.geometry is None or gdf.geometry.is_empty.all():
        geom_types = ["Empty"]
    else:
        geom_types = sorted(
            gdf.geometry.dropna().geom_type.unique().tolist()
        )

    # Column names (excluding geometry column).
    geom_col = gdf.geometry.name
    attribute_columns = [c for c in gdf.columns if c != geom_col]

    # Column dtypes.
    col_info = OrderedDict()
    for col in attribute_columns:
        col_info[col] = {
            "dtype": str(gdf[col].dtype),
            "sample": str(gdf[col].dropna().iloc[0]) if not gdf[col].dropna().empty else None,
        }

    # CRS
    crs_str = str(gdf.crs) if gdf.crs else "None / Undefined"
    crs_epsg = gdf.crs.to_epsg() if gdf.crs else None

    # Bounding box
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

    return {
        "file_name": filepath.name,
        "file_path": str(filepath.relative_to(PROJECT_ROOT)),
        "format": "Shapefile" if filepath.suffix == ".shp" else "GeoJSON",
        "crs": crs_str,
        "crs_epsg": crs_epsg,
        "feature_count": len(gdf),
        "geometry_types": geom_types,
        "bounding_box": {
            "minx": round(float(bounds[0]), 6),
            "miny": round(float(bounds[1]), 6),
            "maxx": round(float(bounds[2]), 6),
            "maxy": round(float(bounds[3]), 6),
        },
        "attribute_columns": list(col_info.keys()),
        "column_details": col_info,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("  KAMRUP SPATIAL DATA SCANNER")
    print("  Scanning:", DATA_DIR)
    print("=" * 72)

    # Collect spatial files.
    spatial_files: list[pathlib.Path] = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.is_dir():
            print(f"⚠  Directory not found: {scan_dir}")
            continue
        for f in sorted(scan_dir.iterdir()):
            if f.suffix.lower() in SPATIAL_EXTS:
                spatial_files.append(f)
            elif f.suffix.lower() == ".json" and _is_geojson_json(f):
                spatial_files.append(f)

    if not spatial_files:
        print("❌  No spatial files found.")
        sys.exit(1)

    print(f"\nFound {len(spatial_files)} spatial file(s) to scan.\n")


    all_metadata: list[dict] = []

    for i, fp in enumerate(spatial_files, 1):
        print(f"--- [{i}/{len(spatial_files)}] {fp.name} ---")
        meta = scan_file(fp)
        if meta is None:
            continue

        all_metadata.append(meta)

        # Console report.
        print(f"  Format        : {meta['format']}")
        print(f"  CRS           : {meta['crs']}")
        if meta["crs_epsg"]:
            print(f"  EPSG Code     : {meta['crs_epsg']}")
        print(f"  Features      : {meta['feature_count']}")
        print(f"  Geometry Type : {', '.join(meta['geometry_types'])}")
        print(f"  Bounding Box  : {meta['bounding_box']}")
        print(f"  Attributes    : {meta['attribute_columns']}")
        print()

    # ---- Write metadata.json -----------------------------------------------
    METADATA_OUT.mkdir(parents=True, exist_ok=True)

    output = {
        "project": "Agentic GIS Orchestrator — Kamrup Flood Risk Dataset",
        "district": "Kamrup, Assam, India",
        "scan_source": str(DATA_DIR),
        "total_layers": len(all_metadata),
        "layers": all_metadata,
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print("=" * 72)
    print(f"✅  Metadata written to: {METADATA_FILE}")
    print(f"   {len(all_metadata)} layer(s) catalogued.")
    print("=" * 72)


if __name__ == "__main__":
    main()
