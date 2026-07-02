"""
geoprocessing.py
================

Core geoprocessing utilities for the Agentic GIS Orchestrator.

This module exposes deterministic, CRS-aware spatial operations that the
LLM agent can invoke as tool calls.  Every public function enforces strict
input validation so that failures surface as clear, actionable error
messages rather than silent geometric corruption.

Functions
---------
fetch_osm_data       – Download live vector data from OpenStreetMap.
project_to_local_utm – Auto-detect and reproject to the local UTM zone.
buffer_vector        – Buffer geometries by a metric distance.
intersect_layers     – Compute the spatial intersection of two layers.
reproject_layer      – Reproject a layer to a target EPSG code.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Union

import geopandas as gpd
import osmnx as ox
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_geodataframe(gdf: object, param_name: str = "gdf") -> None:
    """Raise ``TypeError`` if *gdf* is not a non-empty GeoDataFrame with a
    valid geometry column.

    Parameters
    ----------
    gdf : object
        The object to validate.
    param_name : str, optional
        Name used in error messages so the caller knows *which* argument
        failed (default ``"gdf"``).

    Raises
    ------
    TypeError
        If *gdf* is not a ``GeoDataFrame``.
    ValueError
        If *gdf* is empty or has no active geometry column.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(
            f"'{param_name}' must be a GeoDataFrame, "
            f"got {type(gdf).__name__}."
        )
    if gdf.empty:
        raise ValueError(
            f"'{param_name}' is an empty GeoDataFrame — nothing to process."
        )
    if gdf.geometry.name not in gdf.columns:
        raise ValueError(
            f"'{param_name}' has no active geometry column."
        )


def _validate_crs_exists(gdf: gpd.GeoDataFrame, param_name: str = "gdf") -> None:
    """Raise ``ValueError`` if the GeoDataFrame has no CRS set.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame whose CRS will be checked.
    param_name : str, optional
        Name used in error messages (default ``"gdf"``).

    Raises
    ------
    ValueError
        If ``gdf.crs`` is ``None``.
    """
    if gdf.crs is None:
        raise ValueError(
            f"'{param_name}' has no CRS defined. "
            "Set a CRS with GeoDataFrame.set_crs() before calling this function."
        )


def _is_projected(crs: CRS) -> bool:
    """Return ``True`` if the CRS is projected (metric), ``False`` otherwise."""
    return crs.is_projected


# ---------------------------------------------------------------------------
# OSM tag mapping
# ---------------------------------------------------------------------------

_OSM_TAG_MAP: Dict[str, Dict[str, object]] = {
    # Waterways
    "rivers":      {"waterway": "river"},
    "river":       {"waterway": "river"},
    "streams":     {"waterway": "stream"},
    "stream":      {"waterway": "stream"},
    "canals":      {"waterway": "canal"},
    "canal":       {"waterway": "canal"},
    "waterways":   {"waterway": True},
    "water":       {"natural": "water"},
    # Transport
    "roads":       {"highway": True},
    "road":        {"highway": True},
    "highways":    {"highway": ["motorway", "trunk", "primary", "secondary"]},
    "railway":     {"railway": True},
    "railways":    {"railway": True},
    # Buildings & land use
    "buildings":   {"building": True},
    "building":    {"building": True},
    "parks":       {"leisure": "park"},
    "park":        {"leisure": "park"},
    "forests":     {"landuse": "forest"},
    "forest":      {"landuse": "forest"},
    "landuse":     {"landuse": True},
    # Natural features
    "coastline":   {"natural": "coastline"},
    "fault lines": {"geological": "fault"},
    "fault_lines": {"geological": "fault"},
    "cliffs":      {"natural": "cliff"},
    # Amenities
    "hospitals":   {"amenity": "hospital"},
    "schools":     {"amenity": "school"},
    "restaurants":  {"amenity": "restaurant"},
}


# ---------------------------------------------------------------------------
# Public API — Data Fetching
# ---------------------------------------------------------------------------

def fetch_osm_data(
    location_name: str,
    feature_type: str,
    *,
    custom_tags: Optional[Dict[str, object]] = None,
    timeout: int = 30,
) -> gpd.GeoDataFrame:
    """Download vector geometries from OpenStreetMap for a named place.

    Uses the `osmnx` library to geocode *location_name* and fetch
    features matching *feature_type* within that area.  The returned
    GeoDataFrame is always in **EPSG:4326** (WGS 84).

    Parameters
    ----------
    location_name : str
        A natural-language place name that the Nominatim geocoder can
        resolve, e.g. ``"Roorkee, India"`` or ``"Manhattan, NY"``.
    feature_type : str
        Human-readable feature category such as ``"rivers"``,
        ``"roads"``, ``"buildings"``.  Mapped internally to OSM tags
        via a built-in lookup table.  Case-insensitive.
    custom_tags : dict, optional
        If supplied, overrides the built-in tag mapping.  Must follow
        the ``osmnx.features_from_place`` tag format, e.g.
        ``{"waterway": "river"}``.
    timeout : int, optional
        Network timeout in seconds for the Overpass API request
        (default ``30``).

    Returns
    -------
    gpd.GeoDataFrame
        Features in EPSG:4326.  Returns an **empty** GeoDataFrame
        (with a ``geometry`` column) if the API returns no results or
        if a network error occurs — the function never raises on
        transient failures.

    Raises
    ------
    TypeError
        If ``location_name`` or ``feature_type`` are not non-empty
        strings.
    ValueError
        If ``feature_type`` cannot be resolved to an OSM tag and no
        ``custom_tags`` are provided.

    Examples
    --------
    >>> gdf = fetch_osm_data("Roorkee, India", "rivers")
    >>> gdf.crs.to_epsg()
    4326
    >>> "geometry" in gdf.columns
    True
    """
    # -- Validate inputs -----------------------------------------------------
    if not isinstance(location_name, str) or not location_name.strip():
        raise TypeError(
            "'location_name' must be a non-empty string, "
            f"got {type(location_name).__name__}: {location_name!r}."
        )
    if not isinstance(feature_type, str) or not feature_type.strip():
        raise TypeError(
            "'feature_type' must be a non-empty string, "
            f"got {type(feature_type).__name__}: {feature_type!r}."
        )

    # -- Resolve tags --------------------------------------------------------
    if custom_tags:
        tags = custom_tags
    else:
        key = feature_type.strip().lower()
        tags = _OSM_TAG_MAP.get(key)
        if tags is None:
            raise ValueError(
                f"Unknown feature_type '{feature_type}'. "
                f"Supported types: {', '.join(sorted(_OSM_TAG_MAP))}. "
                f"Alternatively, pass custom_tags={{...}} directly."
            )

    logger.info(
        "Fetching OSM data — location='%s', feature_type='%s', tags=%s.",
        location_name,
        feature_type,
        tags,
    )

    # -- Configure osmnx timeout ---------------------------------------------
    ox.settings.timeout = timeout

    # -- Fetch from Overpass -------------------------------------------------
    _empty = gpd.GeoDataFrame(
        columns=["geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )

    try:
        gdf = ox.features_from_place(location_name, tags=tags)
    except ox._errors.InsufficientResponseError:
        logger.warning(
            "OSM returned no features for '%s' with tags %s.",
            location_name,
            tags,
        )
        return _empty
    except Exception as exc:
        # Catch network timeouts, geocoding failures, etc.
        logger.error(
            "OSM fetch failed for '%s': %s — returning empty GeoDataFrame.",
            location_name,
            exc,
        )
        return _empty

    if gdf.empty:
        logger.warning("OSM query succeeded but returned 0 features.")
        return _empty

    # osmnx may return a MultiIndex; flatten it.
    if isinstance(gdf.index, gpd.pd.MultiIndex):
        gdf = gdf.reset_index(drop=True)

    # Ensure CRS is set (osmnx should set it, but be defensive).
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    logger.info(
        "Fetched %d features from OSM (CRS: %s).",
        len(gdf),
        gdf.crs,
    )
    return gdf


def project_to_local_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject a WGS 84 GeoDataFrame to its automatically detected local UTM zone.

    The UTM zone is determined from the **centroid** of the layer's
    total bounds.  This is essential before running metric operations
    like buffering or distance calculations on data fetched from OSM
    (which arrives in geographic EPSG:4326).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input layer.  Must have a CRS set (typically EPSG:4326).

    Returns
    -------
    gpd.GeoDataFrame
        A **new** GeoDataFrame reprojected to the appropriate UTM zone.
        The original is never modified.

    Raises
    ------
    TypeError
        If *gdf* is not a GeoDataFrame.
    ValueError
        If *gdf* is empty or has no CRS.

    Notes
    -----
    The UTM EPSG code is computed as:

    * **Northern hemisphere**: ``32600 + zone_number``
    * **Southern hemisphere**: ``32700 + zone_number``

    where ``zone_number = floor((lon + 180) / 6) + 1``.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> gdf = gpd.GeoDataFrame(
    ...     {"id": [1]},
    ...     geometry=[Point(77.89, 29.87)],  # Roorkee, India
    ...     crs="EPSG:4326",
    ... )
    >>> projected = project_to_local_utm(gdf)
    >>> projected.crs.to_epsg()  # UTM zone 44N
    32644
    """
    _validate_geodataframe(gdf, "gdf")
    _validate_crs_exists(gdf, "gdf")

    # Work in WGS 84 to compute the centroid longitude/latitude.
    gdf_wgs = gdf if gdf.crs.to_epsg() == 4326 else gdf.to_crs(epsg=4326)

    bounds = gdf_wgs.total_bounds  # [minx, miny, maxx, maxy]
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    # Compute UTM zone number.
    zone_number = int(math.floor((center_lon + 180) / 6)) + 1

    # Northern or southern hemisphere.
    if center_lat >= 0:
        epsg = 32600 + zone_number
    else:
        epsg = 32700 + zone_number

    logger.info(
        "Auto-detected UTM zone %d%s (EPSG:%d) from centroid (%.4f, %.4f).",
        zone_number,
        "N" if center_lat >= 0 else "S",
        epsg,
        center_lon,
        center_lat,
    )

    result = gdf.to_crs(epsg=epsg)
    logger.info("Reprojected %d features to EPSG:%d.", len(result), epsg)
    return result


# ---------------------------------------------------------------------------
# Public API — Spatial Analysis
# ---------------------------------------------------------------------------

def buffer_vector(
    input_gdf: gpd.GeoDataFrame,
    distance: Union[int, float],
) -> gpd.GeoDataFrame:
    """Buffer every geometry in a GeoDataFrame by a fixed metric distance.

    The function requires the GeoDataFrame to carry a **projected** (metric)
    CRS so that the *distance* parameter is interpreted in metres (or the
    linear unit of the CRS).  Passing a geographic CRS (e.g. EPSG:4326)
    will raise a ``ValueError`` because buffering in decimal-degrees
    produces geometrically meaningless results.

    Parameters
    ----------
    input_gdf : gpd.GeoDataFrame
        Source layer whose geometries will be buffered.  Must have a valid
        projected CRS and at least one row.
    distance : int or float
        Buffer distance in the linear unit of ``input_gdf.crs`` (typically
        metres).  Positive values expand geometries; negative values shrink
        them.  A value of ``0`` is accepted but returns a copy with
        identical geometries.

    Returns
    -------
    gpd.GeoDataFrame
        A **new** GeoDataFrame with the same attribute columns and CRS,
        whose geometry column contains the buffered shapes.  The original
        ``input_gdf`` is never modified.

    Raises
    ------
    TypeError
        If ``input_gdf`` is not a GeoDataFrame or ``distance`` is not
        numeric.
    ValueError
        If ``input_gdf`` is empty, has no CRS, or uses a geographic
        (non-projected) CRS.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> gdf = gpd.GeoDataFrame(
    ...     {"id": [1]},
    ...     geometry=[Point(500000, 4500000)],
    ...     crs="EPSG:32633",
    ... )
    >>> buffered = buffer_vector(gdf, distance=100)
    >>> buffered.geometry.iloc[0].geom_type
    'Polygon'
    """
    # -- Input validation ----------------------------------------------------
    _validate_geodataframe(input_gdf, "input_gdf")
    _validate_crs_exists(input_gdf, "input_gdf")

    if not isinstance(distance, (int, float)):
        raise TypeError(
            f"'distance' must be numeric (int or float), "
            f"got {type(distance).__name__}."
        )

    if not _is_projected(input_gdf.crs):
        raise ValueError(
            f"Buffer requires a projected (metric) CRS. "
            f"The input layer uses '{input_gdf.crs}', which is geographic. "
            f"Reproject with reproject_layer() to a projected CRS first "
            f"(e.g. an appropriate UTM zone)."
        )

    # -- Processing ----------------------------------------------------------
    logger.info(
        "Buffering %d geometries by %.4f units (CRS: %s).",
        len(input_gdf),
        distance,
        input_gdf.crs,
    )

    buffered_geometry = input_gdf.geometry.buffer(distance)

    result = input_gdf.copy()
    result[result.geometry.name] = buffered_geometry

    logger.info("Buffer complete — %d features returned.", len(result))
    return result


def intersect_layers(
    gdf_a: gpd.GeoDataFrame,
    gdf_b: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Compute the spatial intersection of two GeoDataFrames.

    Returns only the portions of ``gdf_a`` that overlap with ``gdf_b``,
    retaining attributes from **both** layers.  The function enforces
    strict CRS equality — if the two layers are in different coordinate
    systems the caller must reproject one of them first.

    Parameters
    ----------
    gdf_a : gpd.GeoDataFrame
        The first (left) input layer.
    gdf_b : gpd.GeoDataFrame
        The second (right) input layer.  Must share exactly the same CRS
        as ``gdf_a``.

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame containing only the intersecting geometries
        with merged attributes from both inputs.  May be empty if the two
        layers do not overlap.

    Raises
    ------
    TypeError
        If either argument is not a GeoDataFrame.
    ValueError
        If either layer is empty, has no CRS, or the two CRS definitions
        do not match.

    Notes
    -----
    *  Column-name collisions are resolved by GeoPandas with automatic
       ``_left`` / ``_right`` suffixes.
    *  The output CRS is inherited from ``gdf_a``.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import box
    >>> a = gpd.GeoDataFrame(
    ...     {"name": ["zone"]},
    ...     geometry=[box(0, 0, 2, 2)],
    ...     crs="EPSG:32633",
    ... )
    >>> b = gpd.GeoDataFrame(
    ...     {"type": ["park"]},
    ...     geometry=[box(1, 1, 3, 3)],
    ...     crs="EPSG:32633",
    ... )
    >>> result = intersect_layers(a, b)
    >>> result.geometry.iloc[0].area
    1.0
    """
    # -- Input validation ----------------------------------------------------
    _validate_geodataframe(gdf_a, "gdf_a")
    _validate_geodataframe(gdf_b, "gdf_b")
    _validate_crs_exists(gdf_a, "gdf_a")
    _validate_crs_exists(gdf_b, "gdf_b")

    if not gdf_a.crs.equals(gdf_b.crs):
        raise ValueError(
            f"CRS mismatch — gdf_a uses '{gdf_a.crs}' while gdf_b uses "
            f"'{gdf_b.crs}'. Reproject one of the layers with "
            f"reproject_layer() so both share the same CRS before "
            f"intersecting."
        )

    # -- Processing ----------------------------------------------------------
    logger.info(
        "Intersecting %d features (gdf_a) with %d features (gdf_b) — CRS: %s.",
        len(gdf_a),
        len(gdf_b),
        gdf_a.crs,
    )

    result = gpd.overlay(gdf_a, gdf_b, how="intersection", keep_geom_type=False)

    if result.empty:
        logger.warning(
            "Intersection produced an empty GeoDataFrame — "
            "the two layers may not overlap spatially."
        )
    else:
        logger.info("Intersection complete — %d features returned.", len(result))

    return result


def reproject_layer(
    gdf: gpd.GeoDataFrame,
    target_epsg: int,
) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to the specified EPSG code.

    This is a thin, validation-heavy wrapper around
    ``GeoDataFrame.to_crs()`` that ensures the target EPSG code is
    syntactically valid and that the source layer already carries a CRS
    (reprojection from an undefined CRS is undefined behaviour).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The layer to reproject.  Must already have a CRS assigned.
    target_epsg : int
        Numeric EPSG code for the desired output CRS (e.g. ``4326`` for
        WGS 84, ``32633`` for UTM zone 33N).

    Returns
    -------
    gpd.GeoDataFrame
        A **new** GeoDataFrame reprojected to the target CRS.  The
        original ``gdf`` is never modified.

    Raises
    ------
    TypeError
        If ``gdf`` is not a GeoDataFrame or ``target_epsg`` is not an
        integer.
    ValueError
        If ``gdf`` is empty, has no CRS, or ``target_epsg`` cannot be
        resolved to a valid CRS.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> gdf = gpd.GeoDataFrame(
    ...     {"id": [1]},
    ...     geometry=[Point(12.4964, 41.9028)],
    ...     crs="EPSG:4326",
    ... )
    >>> reprojected = reproject_layer(gdf, target_epsg=32633)
    >>> reprojected.crs.to_epsg()
    32633
    """
    # -- Input validation ----------------------------------------------------
    _validate_geodataframe(gdf, "gdf")
    _validate_crs_exists(gdf, "gdf")

    if not isinstance(target_epsg, int):
        raise TypeError(
            f"'target_epsg' must be an integer EPSG code, "
            f"got {type(target_epsg).__name__} ({target_epsg!r})."
        )

    # Validate that the EPSG code resolves to a real CRS.
    try:
        target_crs = CRS.from_epsg(target_epsg)
    except Exception as exc:
        raise ValueError(
            f"EPSG:{target_epsg} could not be resolved to a valid CRS. "
            f"Underlying error: {exc}"
        ) from exc

    # Skip work if already in the target CRS.
    if gdf.crs.equals(target_crs):
        logger.info(
            "Layer is already in EPSG:%d — returning a copy.", target_epsg
        )
        return gdf.copy()

    # -- Processing ----------------------------------------------------------
    logger.info(
        "Reprojecting %d features from '%s' → EPSG:%d.",
        len(gdf),
        gdf.crs,
        target_epsg,
    )

    result = gdf.to_crs(epsg=target_epsg)

    logger.info("Reprojection complete — CRS is now '%s'.", result.crs)
    return result


# ---------------------------------------------------------------------------
# Custom Errors
# ---------------------------------------------------------------------------

class CRSMismatchError(ValueError):
    """Raised when two GeoDataFrames are not both in the required CRS.

    This is a specialised ``ValueError`` that includes the actual CRS
    values found on each input so the caller (or the LLM self-correction
    loop) can construct an actionable fix.

    Attributes
    ----------
    expected_epsg : int
        The EPSG code that was required (e.g. ``32646``).
    actual_crs_a : str
        String representation of the first GeoDataFrame's CRS.
    actual_crs_b : str
        String representation of the second GeoDataFrame's CRS.
    """

    def __init__(
        self,
        message: str,
        *,
        expected_epsg: int = 32646,
        actual_crs_a: str = "",
        actual_crs_b: str = "",
    ):
        super().__init__(message)
        self.expected_epsg = expected_epsg
        self.actual_crs_a = actual_crs_a
        self.actual_crs_b = actual_crs_b


# ---------------------------------------------------------------------------
# Kamrup-specific constants
# ---------------------------------------------------------------------------

# UTM Zone 46N — the canonical projected CRS for Kamrup / Assam.
_KAMRUP_UTM_EPSG = 32646


# ---------------------------------------------------------------------------
# Public API — Kamrup Spatial Tools
# ---------------------------------------------------------------------------

def project_to_kamrup_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject any GeoDataFrame to **EPSG:32646** (UTM Zone 46N, Assam).

    UTM Zone 46N covers longitudes 90°E–96°E, which encompasses the
    entire Kamrup Metropolitan and Kamrup Rural districts.  All metric
    spatial operations (buffering, distance, area) in this project must
    use this CRS to produce results in **metres / square metres**.

    If the input is *already* in EPSG:32646 the function returns a
    shallow copy without reprojecting, avoiding unnecessary computation.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        A GeoDataFrame in any CRS (typically EPSG:4326 from raw data
        files).  Must have a CRS assigned and at least one feature.

    Returns
    -------
    gpd.GeoDataFrame
        A **new** GeoDataFrame reprojected to EPSG:32646.  The original
        ``gdf`` is never modified.

    Raises
    ------
    TypeError
        If ``gdf`` is not a ``GeoDataFrame``.
    ValueError
        If ``gdf`` is empty, lacks a geometry column, or has no CRS set.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> gdf = gpd.GeoDataFrame(
    ...     {"id": [1]},
    ...     geometry=[Point(91.74, 26.14)],   # Guwahati
    ...     crs="EPSG:4326",
    ... )
    >>> projected = project_to_kamrup_utm(gdf)
    >>> projected.crs.to_epsg()
    32646
    """
    _validate_geodataframe(gdf, "gdf")
    _validate_crs_exists(gdf, "gdf")

    current_epsg = gdf.crs.to_epsg()
    if current_epsg == _KAMRUP_UTM_EPSG:
        logger.info(
            "GeoDataFrame is already in EPSG:%d — returning copy.",
            _KAMRUP_UTM_EPSG,
        )
        return gdf.copy()

    logger.info(
        "Reprojecting %d features from EPSG:%s → EPSG:%d (UTM Zone 46N).",
        len(gdf),
        current_epsg or "unknown",
        _KAMRUP_UTM_EPSG,
    )

    result = gdf.to_crs(epsg=_KAMRUP_UTM_EPSG)

    logger.info(
        "Projection complete — %d features now in EPSG:%d.",
        len(result),
        _KAMRUP_UTM_EPSG,
    )
    return result


def create_buffer(
    gdf: gpd.GeoDataFrame,
    distance_meters: Union[int, float],
) -> gpd.GeoDataFrame:
    """Buffer geometries by a distance in **metres**, auto-projecting first.

    This is the **recommended** high-level buffer tool for the Kamrup
    dataset.  It chains two operations:

    1. ``project_to_kamrup_utm(gdf)`` — ensures the data is in a metric
       CRS (EPSG:32646) so the buffer distance is in metres.
    2. ``geometry.buffer(distance_meters)`` — applies the buffer.

    The returned GeoDataFrame retains the projected CRS (EPSG:32646).
    If you need to render the result on a Leaflet/Folium map, reproject
    it back to EPSG:4326 afterwards with ``reproject_layer(result, 4326)``.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Source layer whose geometries will be buffered.  Can be in *any*
        CRS — the function handles reprojection automatically.
    distance_meters : int or float
        Buffer distance in **metres**.  Positive values expand
        geometries; negative values shrink them (erosion).  Zero is
        accepted but effectively returns a copy.

    Returns
    -------
    gpd.GeoDataFrame
        A **new** GeoDataFrame in EPSG:32646 whose geometry column
        contains the buffered Polygons / MultiPolygons.  All original
        attribute columns are preserved.

    Raises
    ------
    TypeError
        If ``gdf`` is not a GeoDataFrame or ``distance_meters`` is not
        numeric.
    ValueError
        If ``gdf`` is empty, has no geometry, or has no CRS.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> gdf = gpd.GeoDataFrame(
    ...     {"name": ["Guwahati"]},
    ...     geometry=[Point(91.74, 26.14)],
    ...     crs="EPSG:4326",
    ... )
    >>> buffered = create_buffer(gdf, distance_meters=500)
    >>> buffered.crs.to_epsg()
    32646
    >>> buffered.geometry.iloc[0].geom_type
    'Polygon'
    """
    # -- Input validation ----------------------------------------------------
    _validate_geodataframe(gdf, "gdf")
    _validate_crs_exists(gdf, "gdf")

    if not isinstance(distance_meters, (int, float)):
        raise TypeError(
            f"'distance_meters' must be numeric (int or float), "
            f"got {type(distance_meters).__name__}."
        )

    # -- Step 1: Project to metric CRS ---------------------------------------
    projected = project_to_kamrup_utm(gdf)

    # -- Step 2: Buffer in metres --------------------------------------------
    logger.info(
        "Buffering %d geometries by %.2f metres (CRS: EPSG:%d).",
        len(projected),
        distance_meters,
        _KAMRUP_UTM_EPSG,
    )

    buffered_geometry = projected.geometry.buffer(distance_meters)

    result = projected.copy()
    result[result.geometry.name] = buffered_geometry

    logger.info("Buffer complete — %d features returned.", len(result))
    return result


def intersect_features(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Compute the spatial intersection of two GeoDataFrames in EPSG:32646.

    Both inputs **must** be in EPSG:32646 (UTM Zone 46N).  If either
    layer is in a different CRS, the function raises a
    ``CRSMismatchError`` with a clear message telling the caller (or the
    LLM self-correction loop) exactly what went wrong and how to fix it.

    This strict enforcement prevents the silent geometric corruption
    that occurs when intersecting layers in mismatched coordinate
    systems.

    Parameters
    ----------
    gdf1 : gpd.GeoDataFrame
        The first (left) input layer.  Must be in EPSG:32646.
    gdf2 : gpd.GeoDataFrame
        The second (right) input layer.  Must be in EPSG:32646.

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame in EPSG:32646 containing only the
        intersecting geometries with merged attributes from both
        inputs.  Column-name collisions are resolved with ``_1`` /
        ``_2`` suffixes.  May be empty if the layers don't overlap.

    Raises
    ------
    TypeError
        If either argument is not a GeoDataFrame.
    ValueError
        If either layer is empty, has no geometry, or has no CRS.
    CRSMismatchError
        If either layer's CRS is **not** EPSG:32646.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import box
    >>> a = gpd.GeoDataFrame(
    ...     {"name": ["zone"]},
    ...     geometry=[box(500000, 2800000, 510000, 2810000)],
    ...     crs="EPSG:32646",
    ... )
    >>> b = gpd.GeoDataFrame(
    ...     {"type": ["park"]},
    ...     geometry=[box(505000, 2805000, 515000, 2815000)],
    ...     crs="EPSG:32646",
    ... )
    >>> result = intersect_features(a, b)
    >>> len(result) > 0
    True
    """
    # -- Input validation ----------------------------------------------------
    _validate_geodataframe(gdf1, "gdf1")
    _validate_geodataframe(gdf2, "gdf2")
    _validate_crs_exists(gdf1, "gdf1")
    _validate_crs_exists(gdf2, "gdf2")

    # -- Strict EPSG:32646 enforcement ---------------------------------------
    epsg1 = gdf1.crs.to_epsg()
    epsg2 = gdf2.crs.to_epsg()

    errors: list[str] = []
    if epsg1 != _KAMRUP_UTM_EPSG:
        errors.append(
            f"gdf1 is in EPSG:{epsg1} — expected EPSG:{_KAMRUP_UTM_EPSG}."
        )
    if epsg2 != _KAMRUP_UTM_EPSG:
        errors.append(
            f"gdf2 is in EPSG:{epsg2} — expected EPSG:{_KAMRUP_UTM_EPSG}."
        )

    if errors:
        raise CRSMismatchError(
            "Both GeoDataFrames must be in EPSG:32646 (UTM Zone 46N) "
            "before intersecting.  "
            + " ".join(errors)
            + " Call project_to_kamrup_utm() on each layer first.",
            expected_epsg=_KAMRUP_UTM_EPSG,
            actual_crs_a=str(gdf1.crs),
            actual_crs_b=str(gdf2.crs),
        )

    # -- Processing ----------------------------------------------------------
    logger.info(
        "Intersecting %d features (gdf1) × %d features (gdf2) — CRS: EPSG:%d.",
        len(gdf1),
        len(gdf2),
        _KAMRUP_UTM_EPSG,
    )

    result = gpd.overlay(
        gdf1,
        gdf2,
        how="intersection",
        keep_geom_type=False,
    )

    if result.empty:
        logger.warning(
            "Intersection produced 0 features — the two layers may "
            "not overlap spatially."
        )
    else:
        logger.info(
            "Intersection complete — %d features returned.", len(result)
        )

    return result


def calculate_area(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute the area of polygon geometries in **square kilometres**.

    The function auto-projects to EPSG:32646 if the input is not
    already in a projected CRS, so the area is always computed in
    metric units (m²) and then converted to km².

    A new column ``area_sq_km`` is added to the returned GeoDataFrame.
    If the column already exists it is overwritten with freshly computed
    values.

    Non-polygon geometries (Points, Lines) will have an area of ``0.0``.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input layer with polygon (or any) geometries.

    Returns
    -------
    gpd.GeoDataFrame
        A **new** GeoDataFrame with an additional ``area_sq_km`` column.
        The CRS matches whichever projected CRS was used for the
        calculation (EPSG:32646 if reprojection was needed, or the
        original CRS if it was already projected).

    Raises
    ------
    TypeError
        If ``gdf`` is not a GeoDataFrame.
    ValueError
        If ``gdf`` is empty, has no geometry, or has no CRS.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import box
    >>> gdf = gpd.GeoDataFrame(
    ...     {"name": ["test"]},
    ...     geometry=[box(0, 0, 1000, 1000)],     # 1 km × 1 km
    ...     crs="EPSG:32646",
    ... )
    >>> result = calculate_area(gdf)
    >>> round(result["area_sq_km"].iloc[0], 2)
    1.0
    """
    _validate_geodataframe(gdf, "gdf")
    _validate_crs_exists(gdf, "gdf")

    # Ensure we're in a projected CRS for meaningful area computation.
    if _is_projected(gdf.crs):
        working = gdf.copy()
        logger.info(
            "CRS '%s' is projected — computing area directly.", gdf.crs
        )
    else:
        logger.info(
            "CRS '%s' is geographic — projecting to EPSG:%d for area "
            "calculation.",
            gdf.crs,
            _KAMRUP_UTM_EPSG,
        )
        working = project_to_kamrup_utm(gdf)

    # Compute area in m², convert to km².
    area_m2 = working.geometry.area
    working["area_sq_km"] = area_m2 / 1_000_000

    logger.info(
        "Area calculated for %d features — total: %.4f km².",
        len(working),
        working["area_sq_km"].sum(),
    )

    return working
