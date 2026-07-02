# Kamrup Flood Dataset — Layer Descriptions (RAG Reference)

## CRS
All layers: **EPSG:4326 (WGS84 Geographic)**. Coordinates in decimal degrees. 
Bounding box: 25.46–26.49°N, 90.48–91.50°E (Kamrup district, Assam, India).

---

## Raster Layers (ESRI ASCII Grid / GeoTIFF-equivalent)

| Layer | Unit | Range | Model Use |
|-------|------|-------|-----------|
| kamrup_dem_30m.asc | metres | 5–220m | Flood inundation depth base |
| kamrup_slope.asc | degrees | 0–30° | Flow velocity, pooling risk |
| kamrup_elevation_class.asc | class 1-5 | 1=Low, 5=Hills | Categorical ML feature |
| kamrup_rainfall_annual_mm.asc | mm/year | 1200–2500 | Rainfall-flood trigger |
| kamrup_temperature_mean_c.asc | °C | 18–28°C | Evapotranspiration |
| kamrup_ndvi_sentinel2.asc | NDVI | -0.1–0.85 | Land cover proxy |
| kamrup_flood_depth_2022.asc | metres | 0–3m | Training label (inundation) |

Load in QGIS: Layer → Add Layer → Add Raster Layer → select .asc file.

---

## Vector Layers

| Layer | Type | Features | Key Fields |
|-------|------|----------|-----------|
| Assam_Districts_Shapefiles.shp | Polygon | 33 districts | Dist_Name, State_Name |
| kamrup_fault_lines.geojson | LineString | 5 faults | name, fault_type, activity_status |
| kamrup_landuse_lulc.geojson | Polygon | 57 | lulc_class, area_ha, flood_vulnerable |
| kamrup_population_grid.geojson | Polygon | 100 cells | population_2020, flood_exposed_pct |

---

## Flood & Disaster Records (CSV/GeoJSON)

| Layer | Rows | Temporal Coverage | Key Feature |
|-------|------|------------------|-------------|
| kamrup_flood_incidents_2010_2023.csv | 43 | 2010–2023 | severity, gauge, damage |
| kamrup_annual_flood_summary_2010_2023.csv | 14 | 2010–2023 | yearly aggregates, ENSO |
| kamrup_monthly_rainfall_temperature_2010_2023.csv | 168 | 2010–2023 | monthly IMD climate |
| kamrup_brahmaputra_gauge_daily_2022.csv | 184 | May–Oct 2022 | daily gauge + discharge |
| kamrup_block_vulnerability.csv | 12 | 2020 baseline | block risk scores |
| kamrup_flood_incidents_spatial.geojson | 43 | 2010–2023 | geocoded event points |

---

## Danger Levels (Brahmaputra at Guwahati)
- **Normal**: < 4.50m  
- **Warning**: 4.50–5.10m  
- **Danger**: > 5.10m  

---

## Units Reference
| Measurement | Unit |
|---|---|
| Elevation / water level / flood depth | metres (m) |
| Rainfall | millimetres (mm) |
| Temperature | degrees Celsius (°C) |
| Slope | degrees (°) |
| Area (crop damage) | hectares (ha) |
| Area (inundation) | square kilometres (km²) |
| Financial (SDRF, crop loss) | lakh INR (₹ 100,000) |
| River discharge | cumecs (m³/s) |
| Population exposure | count (persons) |
