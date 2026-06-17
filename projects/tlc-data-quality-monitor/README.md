# NYC TLC Taxi Data Quality Monitor

![Animated NYC taxi pickup density map](assets/pickup_density_period.gif)

A local data-engineering project that ingests official NYC TLC yellow taxi trip records, validates and cleans  trip data, builds persistent analytical tables in DuckDB, generates monthly and multi-month reports, and produces an animated geospatial visualization of taxi pickup density by hour.

```text
official TLC Parquet files
→ raw trip table
→ staging view
→ persistent clean fact table
→ zone dimension table
→ monthly reports
→ period reports
→ animated pickup-density map
```

## Project structure

```text
app/
  download_tlc_month.py
  profile_tlc_month.py
  ingest_tlc_month.py
  load_taxi_zones.py
  analyze_tlc_month.py
  analyze_tlc_zones.py
  analyze_tlc_period.py
  build_pickup_density_hybrid_taxi_brand.py
  build_pickup_density_period_hybrid_taxi_brand.py
  run_pipeline.py

assets/
  pickup_density_period.gif

data/
  raw/
  db/

outputs/
  reports/
```


## Setup

```bash
cd ~/physical-ai-systems
source .venv/bin/activate

cd projects/tlc-data-quality-monitor
pip install -r requirements.txt
```

Useful packages:

```text
duckdb
pandas
pyarrow
requests
geopandas
pydeck
```

## Taxi zone geometry

The map needs taxi zone geometry. Put a GeoParquet or GeoJSON file here:

```text
data/raw/tlc/taxi_zones_4326.parquet
```

or pass a custom path:

```bash
--zones-geometry data/raw/tlc/taxi_zones.geojson
```

## Run one month

```bash
python app/run_pipeline.py \
  --taxi-type yellow \
  --year 2026 \
  --month 1 \
  --force-ingest
```

Monthly outputs include:

```text
outputs/reports/yellow_2026_01_monthly_summary.json
outputs/reports/yellow_2026_01_daily_revenue.csv
outputs/reports/yellow_2026_01_hourly_demand.csv
outputs/reports/yellow_2026_01_top_pickup_zones_named.csv
outputs/reports/yellow_2026_01_top_routes_named.csv
outputs/reports/maps/yellow_2026_01_pickup_density_hybrid.html
```

## Run multiple months

```bash
python app/run_pipeline.py \
  --taxi-type yellow \
  --year 2026 \
  --month-range 1-3 \
  --force-ingest
```

This produces monthly outputs for each loaded month plus a period report and period visualization.

Period outputs:

```text
outputs/reports/yellow_2026_01_03_period_summary.json
outputs/reports/yellow_2026_01_03_monthly_trend.csv
outputs/reports/yellow_2026_01_03_hourly_profile_avg_day.csv
outputs/reports/yellow_2026_01_03_top_pickup_zones_avg_day.csv
outputs/reports/yellow_2026_01_03_zone_hour_profile_avg_day.csv
outputs/reports/maps/yellow_2026_01_03_pickup_density_period_hybrid.html
```

## Data model

### `raw_yellow_trips`

Persistent raw table. Stores all loaded raw records with source metadata:

```text
taxi_type
data_year
data_month
source_file
ingested_at_utc
```

### `stg_yellow_trips`

A staging view over the raw table. Standardizes names and derives:

```text
pickup_datetime
dropoff_datetime
pickup_date
pickup_hour
duration_min
speed_mph
```

### `clean_yellow_trips`

Persistent clean fact table. Stores all cleaned monthly partitions. Each run replaces only the requested month partition instead of wiping the whole table.

Cleaning removes rows with impossible or suspicious values:

```text
pickup outside target month
dropoff before pickup
non-positive duration
duration over 24 hours
non-positive distance
distance over 100 miles
speed over 100 mph
negative fare amount
negative total amount
```

## Monthly vs period analysis

Monthly reports remain separate because they are useful for data quality, demand, revenue, and anomaly checks by month.

The period visualization aggregates across selected months using:

```text
average pickups per active pickup day at each hour
```

This avoids making longer months look busier just because they have more days.


