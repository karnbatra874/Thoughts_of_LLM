"""Smoke test for the four new Kamrup geoprocessing tools."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.tools.geoprocessing import (
    project_to_kamrup_utm,
    create_buffer,
    intersect_features,
    calculate_area,
    CRSMismatchError,
)
import geopandas as gpd
from shapely.geometry import Point, box

SEP = "=" * 60

# --- Test 1: project_to_kamrup_utm ---
print(SEP)
print("  Test 1: project_to_kamrup_utm")
print(SEP)
gdf = gpd.GeoDataFrame(
    {"name": ["Guwahati"]},
    geometry=[Point(91.74, 26.14)],
    crs="EPSG:4326",
)
proj = project_to_kamrup_utm(gdf)
assert proj.crs.to_epsg() == 32646, f"Expected 32646, got {proj.crs.to_epsg()}"
# Idempotency check
proj2 = project_to_kamrup_utm(proj)
assert proj2.crs.to_epsg() == 32646
print(f"  Input: EPSG:4326 -> Output: EPSG:{proj.crs.to_epsg()}")
print("  PASS\n")

# --- Test 2: create_buffer ---
print(SEP)
print("  Test 2: create_buffer")
print(SEP)
buf = create_buffer(gdf, 500)
assert buf.crs.to_epsg() == 32646
assert buf.geometry.iloc[0].geom_type == "Polygon"
print(f"  CRS: EPSG:{buf.crs.to_epsg()}, Geom: {buf.geometry.iloc[0].geom_type}")
print("  PASS\n")

# --- Test 3: intersect_features ---
print(SEP)
print("  Test 3: intersect_features (matching CRS)")
print(SEP)
a = gpd.GeoDataFrame(
    {"name": ["zone"]},
    geometry=[box(500000, 2800000, 510000, 2810000)],
    crs="EPSG:32646",
)
b = gpd.GeoDataFrame(
    {"type": ["park"]},
    geometry=[box(505000, 2805000, 515000, 2815000)],
    crs="EPSG:32646",
)
result = intersect_features(a, b)
assert len(result) == 1
assert result.geometry.iloc[0].area == 5000 * 5000  # 25 km^2
print(f"  Intersected: {len(result)} feature, area={result.geometry.iloc[0].area} m^2")
print("  PASS\n")

# --- Test 3b: CRS mismatch ---
print(SEP)
print("  Test 3b: intersect_features (CRS mismatch)")
print(SEP)
bad = gpd.GeoDataFrame(
    {"name": ["bad"]},
    geometry=[Point(91, 26)],
    crs="EPSG:4326",
)
try:
    intersect_features(a, bad)
    print("  FAIL — no error raised!")
    sys.exit(1)
except CRSMismatchError as e:
    print(f"  CRSMismatchError raised correctly")
    print(f"  expected_epsg: {e.expected_epsg}")
    print(f"  actual_crs_b: {e.actual_crs_b}")
    print("  PASS\n")

# --- Test 4: calculate_area ---
print(SEP)
print("  Test 4: calculate_area")
print(SEP)
poly = gpd.GeoDataFrame(
    {"name": ["square_1km"]},
    geometry=[box(0, 0, 1000, 1000)],
    crs="EPSG:32646",
)
ar = calculate_area(poly)
assert "area_sq_km" in ar.columns
assert round(ar["area_sq_km"].iloc[0], 2) == 1.0
print(f"  area_sq_km: {ar['area_sq_km'].iloc[0]}")

# Also test with geographic CRS (auto-projects)
geo_poly = gpd.GeoDataFrame(
    {"name": ["kamrup_area"]},
    geometry=[box(91.5, 26.0, 91.6, 26.1)],
    crs="EPSG:4326",
)
ar2 = calculate_area(geo_poly)
assert "area_sq_km" in ar2.columns
assert ar2["area_sq_km"].iloc[0] > 0
print(f"  Geographic auto-project area: {ar2['area_sq_km'].iloc[0]:.2f} km^2")
print("  PASS\n")

print(SEP)
print("  ALL 4 TESTS PASSED")
print(SEP)
