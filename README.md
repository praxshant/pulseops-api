---
title: PulseOps API
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# PulseOps — Production ML Discharge Forecasting API

Next-calendar-day, organisation-level hospital discharge forecasting for NHS England
operational planning. This Space serves the **champion ExtraTrees model** (pulled from the
DagsHub MLflow registry at build time and baked into the image) behind a FastAPI service.

The model source of truth is the DagsHub/MLflow registry; this container serves an immutable
copy so inference has **no runtime registry dependency**.

## Endpoints

| Method | Path           | Description                                        |
|--------|----------------|----------------------------------------------------|
| GET    | `/health`      | Liveness probe                                     |
| GET    | `/docs`        | Interactive OpenAPI UI                             |
| POST   | `/predict`     | Next-day discharge forecast (17 features required) |
| GET    | `/drift`       | PSI-based feature drift vs. training reference     |
| GET    | `/performance` | Rolling MAE / RMSE / WAPE from inference log       |
| GET    | `/metrics`     | Prometheus metrics                                 |

## Example prediction

```bash
curl -X POST https://<your-space>.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "org_code": "RXX",
    "forecast_date": "2026-09-01",
    "features": {
      "day_of_week": 1, "is_weekend": 0,
      "lag_1_discharge": 120, "lag_7_discharge": 115, "lag_14_discharge": 110, "lag_28_discharge": 108,
      "rolling_7d_discharge": 117, "rolling_14d_discharge": 113, "rolling_28d_discharge": 111,
      "previous_month_total_attendances": 8500, "previous_month_emergency_admissions": 2100,
      "previous_month_over_4_hours": 300, "ae_context_available": 1,
      "occupied_beds": 450, "available_beds": 500, "occupancy_rate": 0.9, "bed_context_available": 1
    }
  }'
```

## Architecture

```
DagsHub / MLflow registry  →  pull_champion.py  →  baked into Docker image  →  this Space
```

Model: ExtraTrees | Chronological train/val/test split | Test MAE ≈ 10.3, WAPE ≈ 13.4%
