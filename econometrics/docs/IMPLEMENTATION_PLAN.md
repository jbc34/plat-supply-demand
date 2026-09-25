# Econometric Rent Growth Forecasting Module — Implementation Plan

**Location**: `supply-demand/econometrics/`
**Status**: Technical Specification (Pre-Implementation)
**Last Updated**: 2026-04-11

---

## 1. Executive Summary

This document specifies a standalone econometric rent growth forecasting module that lives alongside the state-space market cycle paper (`supply_demand/`) in the same repository. It addresses three categories of gaps in the existing forecasting infrastructure:

**(A) Econometric Models** — VAR, cointegration/ECM, spatial spillover, seasonality decomposition
**(B) Data Enrichment** — permit-to-delivery lag model, IRS migration flows, interest rate channel, concession tracking
**(C) Process Infrastructure** — backtest registry, ensemble weighting, proper confidence intervals, stress testing

The module is metro-agnostic (YAML config-driven), production-ready for deal underwriting, and has zero modifications to any paper code.

---

## 2. Architecture

### 2.1 Design Principles

1. **Paper isolation** — zero imports from `supply_demand/` into `econometrics/` except the `FREDClient` utility (stable, non-paper code). Zero modifications to any file under `supply_demand/`.
2. **Config-driven** — all metro-specific parameters (FRED series IDs, BLS MSA codes, submarket coordinates, permit lag months) live in YAML files, not Python code.
3. **Protocol-based models** — every model implements the same `ForecastModel` protocol (`fit/predict/residuals`). The ensemble combiner only knows about the protocol, never concrete classes.
4. **Append-only audit trail** — every forecast is logged to a JSON registry with a UUID. Actuals are filled in retroactively. Nothing is ever deleted.
5. **Soft dependencies** — spatial models (`libpysal`/`spreg`) and gradient boosting (`lightgbm`) are optional extras. The module degrades gracefully without them.

### 2.2 Integration Points

| System | Direction | Mechanism |
|--------|-----------|-----------|
| **market-study-agent** `etl/supply_model.py` | Read-only | Reads CoStar submarket panel parquets from `data/public/processed/supply/` |
| **geostack** | Read-only | Calls `migration.py`, `trends.py`, `affordability.py` for demand-side features |
| **FRED API** | Read-only | Wraps existing `supply_demand.data.sources.FREDClient` for rate series |
| **Census BPS API** | Read-only | Permit data via HTTP (no key required) |
| **IRS SOI** | Read-only | County-to-county migration CSVs from irs.gov |

### 2.3 Directory Structure

```
supply-demand/
├── econometrics/                          # NEW — top-level package
│   ├── __init__.py                        # __version__, public API
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py                    # EconSettings (paths, env vars)
│   │   └── metros/
│   │       ├── _template.yaml             # Schema reference with comments
│   │       ├── dallas_tx.yaml
│   │       ├── austin_tx.yaml
│   │       └── ...
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── costar_panel.py            # Read CoStar parquets
│   │   │   ├── fred_macro.py              # FRED rate series (wraps FREDClient)
│   │   │   ├── census_permits.py          # BPS building permits
│   │   │   ├── irs_migration.py           # IRS SOI net migration + income delta
│   │   │   └── concessions.py             # Effective vs. asking rent spread
│   │   └── transforms/
│   │       ├── __init__.py
│   │       ├── panel_builder.py           # Merge all sources to quarterly panel
│   │       ├── permit_lag.py              # Calibrate permit-to-delivery lag
│   │       └── spatial_weights.py         # Build W matrix for spatial models
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                        # ForecastModel protocol + ForecastResult
│   │   ├── var_engine.py                  # VAR(p) with lag-order selection
│   │   ├── ecm_engine.py                  # Johansen cointegration + VECM
│   │   ├── spatial_engine.py              # Spatial lag/error (libpysal/spreg)
│   │   ├── stl_decomp.py                 # STL seasonality decomposition
│   │   ├── ridge_lasso.py                # Ridge/LASSO/Elastic Net
│   │   └── gradient_boost.py             # LightGBM / sklearn GBM wrapper
│   ├── ensemble/
│   │   ├── __init__.py
│   │   ├── weighter.py                    # Inverse-RMSE weights from backtest registry
│   │   ├── combiner.py                    # Weighted point forecast + CI
│   │   └── confidence.py                  # Empirical quantile CI (80%/95%)
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── registry.py                    # Append-only JSON forecast log
│   │   ├── runner.py                      # Expanding-window backtest loop
│   │   └── metrics.py                     # RMSE, MAE, directional accuracy, coverage
│   ├── scenarios/
│   │   ├── __init__.py
│   │   ├── shock_library.py               # Named shocks: GFC_2008, COVID_2020, etc.
│   │   └── stress_tester.py               # Apply shock → run ensemble → forecast
│   ├── reports/
│   │   ├── __init__.py
│   │   ├── markdown_writer.py             # Forecast report MD
│   │   └── chart_builder.py               # Fan charts, scenario overlays, IRFs
│   └── cli/
│       ├── __init__.py                    # main() dispatcher with subcommands
│       ├── forecast.py                    # econ forecast --metro dallas_tx
│       ├── backtest.py                    # econ backtest --metro dallas_tx
│       └── stress.py                      # econ stress --metro dallas_tx --scenario GFC_2008
│
├── data/econometrics/                     # NEW — output data
│   ├── forecasts/{metro_id}/              # Parquet outputs per metro + vintage
│   ├── registry/{metro_id}_backtest.json  # Append-only JSON logs
│   ├── scenarios/                         # Stress test parquets
│   └── reports/                           # MD + PNG outputs
│
├── tests/econometrics/                    # NEW — test subfolder
│   ├── __init__.py
│   ├── conftest.py                        # Synthetic panel fixture (60Q, deterministic)
│   ├── test_var_engine.py
│   ├── test_ecm_engine.py
│   ├── test_spatial_engine.py
│   ├── test_ensemble_weighter.py
│   ├── test_backtest_registry.py
│   ├── test_permit_lag.py
│   └── test_stress_tester.py
│
├── supply_demand/                         # UNTOUCHED — paper code
├── research/                              # UNTOUCHED — paper experiments
└── pyproject.toml                         # Two targeted additions only
```

---

## 3. Part (A): Econometric Models

### 3.1 VAR Engine (`models/var_engine.py`)

**Purpose**: Model rent growth, vacancy, deliveries, and absorption as an endogenous system where each variable depends on its own lags and the lags of all other variables.

**Why it matters**: The existing ridge/OLS models treat vacancy and deliveries as exogenous. In reality, rent growth affects absorption → affects vacancy → affects future deliveries → affects future rent growth. VAR captures this feedback loop.

**Specification**:
- **Endogenous variables** (from metro YAML `endogenous_vars`):
  - `rent_growth_yoy` — effective rent YoY % change (target)
  - `vacancy_rate` — submarket vacancy rate
  - `deliveries_pct_inventory` — trailing 12-month deliveries as % of inventory
  - `absorption_pct_inventory` — trailing 12-month absorption as % of inventory
- **Exogenous variables** (from metro YAML `exogenous_vars`, optional):
  - `mortgage_rate_30y` — Freddie Mac 30Y fixed rate
  - `unemployment_rate` — BLS LAUS MSA unemployment
  - `net_migration_returns` — IRS SOI net tax returns
  - `permit_pipeline` — BPS permits lagged by metro-specific months
- **Lag order selection**: `statsmodels.tsa.vector_ar.var_model.VAR.select_order()` evaluating AIC, BIC, HQIC, and FPE. Default: choose minimum AIC with `maxlags=8`.
- **Estimation**: OLS per equation (via `statsmodels`).
- **Forecasting**: `VARResults.forecast_interval(y_last, steps=horizon, alpha=0.20)` for 80% CI; `alpha=0.05` for 95% CI. Extract the `rent_growth_yoy` equation from the multivariate forecast.
- **Impulse Response Functions (IRF)**: `VARResults.irf(periods=20)` — used for chart output showing how a 1 std dev shock to deliveries propagates through the system.

**Implementation details**:
```python
class VAREngine:
    """VAR(p) model for endogenous rent-vacancy-supply system."""

    def __init__(self, max_lags: int = 8, ic: str = "aic"):
        self.max_lags = max_lags
        self.ic = ic
        self._result: VAR Results | None = None

    def fit(self, panel: pd.DataFrame) -> None:
        """Fit VAR model with automatic lag selection."""
        endog = panel[self.endog_cols].dropna()
        model = VAR(endog)
        lag_order = model.select_order(maxlags=self.max_lags)
        selected_p = lag_order.selected_orders[self.ic]
        self._result = model.fit(selected_p)

    def predict(self, horizon: int = 8) -> ForecastResult:
        """Forecast with 80% and 95% confidence intervals."""
        # forecast_interval returns (point, lower, upper) for each alpha
        point, lo_80, hi_80 = self._result.forecast_interval(
            self._result.y[-self._result.k_ar:], steps=horizon, alpha=0.20
        )
        _, lo_95, hi_95 = self._result.forecast_interval(
            self._result.y[-self._result.k_ar:], steps=horizon, alpha=0.05
        )
        # Extract rent_growth_yoy column index
        idx = self.endog_cols.index("rent_growth_yoy")
        return ForecastResult(
            point=point[:, idx],
            ci_80_lo=lo_80[:, idx], ci_80_hi=hi_80[:, idx],
            ci_95_lo=lo_95[:, idx], ci_95_hi=hi_95[:, idx],
            model_name="VAREngine",
            ...
        )

    def irf(self, periods: int = 20) -> IRFResult:
        """Compute impulse response functions for charting."""
        return self._result.irf(periods=periods)
```

**Testing strategy**: Fit on synthetic 60-quarter panel with known VAR(2) DGP. Assert:
- Selected lag order matches DGP
- Point forecast for 4 quarters is within 2 std devs of true values
- 95% CI brackets contain the point estimate
- IRF from deliveries shock shows expected negative rent growth response

### 3.2 ECM Engine (`models/ecm_engine.py`)

**Purpose**: Test whether rents and incomes share a long-run equilibrium (cointegration), and if so, estimate how fast rents revert to that equilibrium when they diverge.

**Why it matters**: Your CAGR-based income projections (geostack `projections.py`) implicitly assume rent grows in lockstep with income. The ECM tests this assumption empirically and measures the speed of reversion — critical for post-renovation rent sustainability.

**Specification**:
- **Step 1: Johansen cointegration test** on `[log_effective_rent, log_median_hh_income]`
  - `statsmodels.tsa.vector_ar.vecm.coint_johansen(data, det_order=0, k_ar_diff=2)`
  - Report trace statistic and max eigenvalue vs. critical values at 90%, 95%, 99%
  - If rank >= 1: cointegration exists, proceed to VECM
  - If rank == 0: no cointegration, log warning, fall back to VAR in first differences
- **Step 2: VECM estimation** (if cointegrated)
  - `statsmodels.tsa.vector_ar.vecm.VECM(data, k_ar_diff=p, coint_rank=r)`
  - The **error correction coefficient (alpha)** tells you: how much of the rent-income gap is closed each quarter
  - A coefficient of -0.15 means 15% of the disequilibrium corrects per quarter → ~4 quarter half-life
- **Step 3: Forecast** via `VECMResults.predict(steps=horizon)`
  - Extract rent equation forecast
  - Confidence intervals from residual bootstrap (VECM doesn't have built-in `forecast_interval`)

**Key output for IC decks**: The cointegrating vector and error correction speed. Example narrative: *"Rents and incomes in the Far North Dallas catchment are cointegrated (Johansen trace stat 28.4 > 20.04 critical at 95%). The error correction speed is -0.12 per quarter, implying a 5.4-quarter half-life. This means a $200/month post-renovation rent premium above equilibrium would erode by ~$100 within 14 months absent income growth."*

**Fallback behavior**: When cointegration is not found (common for short time series < 40 quarters), the engine automatically produces a VAR in first differences (Δrent, Δincome) with a logged warning. The ensemble treats this as a valid model with potentially higher RMSE.

### 3.3 Spatial Spillover Engine (`models/spatial_engine.py`)

**Purpose**: Capture how rent growth in adjacent submarkets (Addison, Richardson, Plano) affects Far North Dallas rent growth beyond what own-submarket fundamentals explain.

**Why it matters**: Submarket boundaries are arbitrary. A new Class A tower in Addison pulls tenants from Far North Dallas; a supply glut in Plano creates pricing pressure. The existing cascade model handles tier filtration (Class A → B → C) but not geographic spillover.

**Specification**:
- **Spatial weights matrix W** (`data/transforms/spatial_weights.py`):
  - Built from submarket centroid lat/lon pairs in the metro YAML
  - Default: inverse-distance weighting: `w_ij = 1/d_ij`, row-normalized
  - Alternative: queen contiguity (if submarket polygons are available)
  - The YAML `spatial_contiguity` field controls which method
- **Spatial Lag Model (SLM)**:
  - `rent_growth_i = rho * (W * rent_growth) + X_i * beta + epsilon_i`
  - Where `rho` is the spatial autoregressive coefficient (the key parameter)
  - `rho > 0` means positive spillover (rent growth in neighbors lifts yours)
  - Estimated via ML or 2SLS using `spreg.ML_Lag` or `spreg.GM_Lag`
- **Spatial Error Model (SEM)** (diagnostic alternative):
  - `rent_growth_i = X_i * beta + u_i` where `u_i = lambda * (W * u) + epsilon_i`
  - Tests whether spatial correlation is in the errors (omitted spatial variables) vs. the DV
  - Moran's I test on OLS residuals determines which specification fits better

**Data requirement**: Need at least 4 submarkets with quarterly rent growth series to construct a meaningful W matrix. With the DFW multi-submarket CoStar data, this is satisfied.

**Dependency**: `libpysal>=4.9` and `spreg>=1.4` as optional extras in `pyproject.toml`. The module does a lazy import and skips gracefully if not installed.

**Testing strategy**: Synthetic spatial panel with known `rho=0.3`. Assert estimated rho is within 0.1 of true value. Test that Moran's I correctly identifies spatial lag vs. spatial error DGP.

### 3.4 STL Decomposition (`models/stl_decomp.py`)

**Purpose**: Separate the seasonal, trend, and residual components of quarterly rent data before feeding it into the VAR/ECM models.

**Why it matters**: CoStar quarterly data has seasonal patterns (Q2/Q3 lease-up season shows higher absorption). Without decomposition, the VAR attributes seasonal variance to structural dynamics, inflating the lag coefficients and producing noisy forecasts.

**Specification**:
- **Method**: `statsmodels.tsa.seasonal.STL` with `period=4` (quarterly)
- **Not a ForecastModel** — this is a pre-processing transform applied before the VAR/ECM fit step
- **Input**: Raw quarterly rent growth series
- **Output**: Three components:
  - `trend` — the underlying growth path (this is what VAR should model)
  - `seasonal` — recurring Q1-Q4 pattern (added back to the final forecast)
  - `residual` — noise (contributes to CI width)
- **Integration**: `panel_builder.py` optionally runs STL on each endogenous variable before passing to models. The metro YAML has a `decompose_seasonal: true/false` flag.

**Implementation**:
```python
def decompose(series: pd.Series, period: int = 4) -> STLComponents:
    """STL decomposition of a quarterly series."""
    stl = STL(series.dropna(), period=period, robust=True)
    result = stl.fit()
    return STLComponents(
        trend=result.trend,
        seasonal=result.seasonal,
        residual=result.resid,
        original=series,
    )
```

---

## 4. Part (B): Data Enrichment

### 4.1 Permit-to-Delivery Lag Model (`data/transforms/permit_lag.py`)

**Purpose**: Estimate how many months it takes for building permits to convert to actual unit deliveries in a specific metro. This turns BPS permit data into a forward supply pressure variable.

**Why it matters**: Permits are the earliest observable supply signal — they lead deliveries by 12-24 months. But the lag varies by metro (Austin: ~14 months, DFW: ~18 months, Houston: ~20 months due to labor availability). A calibrated lag converts the permit series into a usable forecasting input.

**Method**:
1. Fetch quarterly Census BPS permits for the metro CBSA (`census_permits.py`)
2. Fetch quarterly CoStar deliveries from the panel (`costar_panel.py`)
3. For each candidate lag (12, 14, 16, 18, 20, 22, 24 months):
   - Shift permits forward by `lag` months
   - Compute correlation and MSE between lagged permits and actual deliveries
4. Select the lag that maximizes correlation (or minimizes MSE)
5. Store the calibrated lag in the metro YAML sidecar (`_calibrated.yaml`)

**Output**: `permit_pipeline` column in the quarterly panel — the BPS permit count shifted by the calibrated lag, representing expected deliveries N months from now.

**CLI**: `econ calibrate-permit-lag --metro dallas_tx`

**Validation**: Plot lagged permits vs. actual deliveries. The correlation should be ≥ 0.5 for the model to use permits; below that, log a warning and exclude permits from exogenous variables.

### 4.2 IRS SOI Migration Flows (`data/loaders/irs_migration.py`)

**Purpose**: Add net household migration and income quality of arrivals as demand-side features.

**Why it matters**: Net migration is the single strongest structural demand driver for Sunbelt multifamily. DFW gained ~150K net new tax returns (proxy for households) in the 2020-2023 period. This directly feeds housing demand that vacancy and absorption statistics only partially capture.

**Data source**: IRS Statistics of Income, county-to-county migration files
- Inflow: `https://www.irs.gov/pub/irs-soi/countyinflow{YY1}{YY2}.csv`
- Outflow: `https://www.irs.gov/pub/irs-soi/countyoutflow{YY1}{YY2}.csv`
- Available annually, ~18 month lag (2023→2024 data available mid-2025)

**Features produced** (annual, interpolated to quarterly):
| Feature | Definition | Economic meaning |
|---------|-----------|-----------------|
| `net_migration_returns` | Inflow returns - outflow returns | Net household formation from migration |
| `income_delta_pct` | (in_avg_agi - out_avg_agi) / out_avg_agi | Quality of migration — are richer or poorer households arriving? |
| `diff_state_movers_pct` | ACS B07010 out-of-state movers / total pop | External demand pressure intensity |

**Integration with geostack**: If `geostack.migration.soi_migration()` is importable (geostack installed as editable dep), use it directly. Otherwise, implement a standalone SOI CSV parser — the data format is stable and simple.

**Quarterly interpolation**: Annual SOI data is interpolated to quarters using linear interpolation. The migration effect is structural (not seasonal), so linear is appropriate. The metro YAML specifies `migration_interpolation: linear` (could also support `step` for no interpolation).

### 4.3 Interest Rate Channel (`data/loaders/fred_macro.py`)

**Purpose**: Capture how changes in mortgage rates affect renter demand. Higher mortgage rates → fewer home purchases → more renters → demand tailwind for apartments.

**FRED series to pull**:
| Series ID | Name | Frequency | Use |
|-----------|------|-----------|-----|
| `MORTGAGE30US` | 30-Year Fixed Mortgage Rate | Weekly → quarterly avg | Direct housing affordability signal |
| `DGS10` | 10-Year Treasury Constant Maturity | Daily → quarterly avg | Risk-free rate benchmark |
| `FEDFUNDS` | Federal Funds Effective Rate | Daily → quarterly avg | Monetary policy stance |
| `BAA10Y` | Moody's BAA Corporate Bond Spread | Daily → quarterly avg | Credit risk premium |
| `T10Y2Y` | 10Y-2Y Treasury Spread | Daily → quarterly avg | Yield curve (recession indicator) |

**Derived features**:
| Feature | Formula | Economic meaning |
|---------|---------|-----------------|
| `mortgage_rate_30y` | Quarterly average of MORTGAGE30US | Direct cost of homeownership |
| `mortgage_rate_chg_yoy` | YoY change in mortgage rate | Direction of affordability pressure |
| `term_spread` | T10Y2Y quarterly average | Yield curve slope — negative = recession risk |
| `credit_spread` | BAA10Y quarterly average | Corporate stress indicator |

**Implementation**: Wrap `supply_demand.data.sources.FREDClient` which already handles FRED API authentication via `FRED_API_KEY` env var. Add a `resample_to_quarterly(series, method="mean")` utility for frequency conversion.

### 4.4 Concession Spread Tracking (`data/loaders/concessions.py`)

**Purpose**: Track the gap between asking and effective rent as a leading indicator of market softening.

**Why it matters**: When concessions widen (effective rent drops below asking), it signals demand weakness 1-2 quarters before vacancy rises or rent growth turns negative. This is the earliest soft signal available from CoStar data.

**Computation**:
```python
concession_spread = (asking_rent_unit - effective_rent_unit) / asking_rent_unit
concession_spread_chg = concession_spread - concession_spread.shift(4)  # YoY change
```

**Source**: Already present in the CoStar panel (`asking_rent_unit` and `effective_rent_unit` columns). No additional data fetch needed — this is a derived transform.

**Feature**: `concession_spread` and `concession_spread_chg_yoy` added to the quarterly panel as exogenous variables.

---

## 5. Part (C): Process Infrastructure

### 5.1 Backtest Registry (`backtest/registry.py`)

**Purpose**: Maintain an append-only log of every forecast produced, then retroactively fill in actuals when new CoStar data arrives. This enables accuracy measurement and ensemble weight calibration.

**Schema** (one JSON entry per forecast vintage per model):
```json
{
    "run_id": "uuid4",
    "metro_id": "dallas_tx",
    "submarket_id": "far_north_dallas",
    "vintage_date": "2026-Q1",
    "model_name": "VAREngine",
    "horizon_quarters": 8,
    "forecast_periods": ["2026-Q2", "2026-Q3", "2026-Q4", "2027-Q1", ...],
    "point_forecast": [2.1, 1.8, 1.5, 1.2, ...],
    "ci_80_lo": [0.8, 0.5, ...],
    "ci_80_hi": [3.4, 3.1, ...],
    "ci_95_lo": [-0.2, -0.5, ...],
    "ci_95_hi": [4.4, 4.1, ...],
    "actuals": [null, null, null, null, ...],
    "rmse": null,
    "mae": null,
    "directional_accuracy": null,
    "ci_80_coverage": null,
    "ci_95_coverage": null,
    "run_timestamp": "2026-04-11T10:00:00Z",
    "panel_hash": "sha256:abc123..."
}
```

**Operations**:
- `append(result: ForecastResult)` — write new entry
- `fill_actuals(metro_id, panel)` — for all entries where actuals are null and the period has passed, fill from the panel and compute metrics
- `load_completed(metro_id, min_quarters=4)` — return entries where actuals are fully populated (for ensemble weighting)
- `accuracy_report(metro_id)` — summary table of RMSE/MAE/coverage by model

**Storage**: One JSON file per metro: `data/econometrics/registry/{metro_id}_backtest.json`. Append-only with atomic writes (write to `.tmp`, then `os.rename`).

**CLI**: `econ backtest --fill-actuals --metro dallas_tx`

### 5.2 Ensemble Weighting (`ensemble/weighter.py`)

**Purpose**: Combine multiple models (VAR, ECM, Ridge, GBM, Spatial) into a single forecast using weights calibrated to recent backtest performance.

**Method**: Inverse-RMSE weighting from the backtest registry.

```
w_i = (1 / RMSE_i) / sum(1 / RMSE_j for all models j)
```

**Cold-start handling**: Models with fewer than 4 completed backtests get equal weight: `1 / N` where N is the number of enabled models. This prevents a single lucky early forecast from dominating.

**Model enable/disable**: The metro YAML has an `enabled_models` list:
```yaml
enabled_models:
  - VAREngine
  - ECMEngine
  - RidgeLassoEngine
  # SpatialEngine omitted → excluded from ensemble
```

**Weight decay (optional)**: More recent backtests can be weighted more heavily using exponential decay: `w_i,t = (1/RMSE_i,t) * lambda^(T-t)` where `lambda = 0.9` and `T` is the current quarter. The metro YAML `ensemble_decay: 0.9` controls this. Set to `1.0` for equal weighting of all vintages.

### 5.3 Confidence Intervals (`ensemble/confidence.py`)

**Purpose**: Produce proper 80% and 95% forecast bands from the empirical distribution of model residuals, not from parametric assumptions.

**Why not ±1 RMSE?**:
- ±1 RMSE assumes normally distributed errors (often violated in rent data — fat tails during shocks)
- ±1 RMSE is a ~68% band, not 80% or 95%
- Different models have different residual distributions; the ensemble CI should reflect the combined uncertainty

**Method**: Empirical quantile bootstrap.
1. Collect residuals from all ensemble members, weighted by ensemble weight
2. For each forecast horizon h (1Q, 2Q, ..., 8Q):
   - Pool the h-step-ahead residuals across all models and backtest vintages
   - Weight each residual by the model's current ensemble weight
   - Compute the 10th, 20th, 80th, 90th percentiles of the weighted distribution
   - CI_80 = point ± [10th percentile, 90th percentile]
   - CI_95 = point ± [2.5th percentile, 97.5th percentile]
3. CIs naturally widen with horizon (h-step residuals have more variance than 1-step)

**Output**: `ci_80_lo`, `ci_80_hi`, `ci_95_lo`, `ci_95_hi` columns in `ForecastResult`.

### 5.4 Stress Testing (`scenarios/stress_tester.py`)

**Purpose**: Answer "what happens to our rent growth forecast if X macro shock occurs?" using historically-calibrated scenarios.

**Shock Library** (`scenarios/shock_library.py`):

| Shock | Pattern | Duration | Key Variables Affected |
|-------|---------|----------|----------------------|
| `GFC_2008` | 2008-2009 financial crisis | 6Q (ramp-in 2Q) | Employment -6pp, mortgage rate +120bps, migration -25%, absorption -40% |
| `COVID_2020` | 2020 pandemic shock | 4Q (sharp V) | Absorption -60% for 2Q then +30% rebound, vacancy +300bps peak, rent growth -5pp trough |
| `RATE_SHOCK_400BPS` | Rapid rate increase | 4Q (step function) | Mortgage rate +400bps, no employment change, migration -10% |
| `SUPPLY_GLUT` | 2015-2017 DFW overbuilding pattern | 8Q (sustained) | Deliveries +50% above trend, absorption unchanged, vacancy +200bps |

**Method**:
1. Take the current quarterly panel
2. For each future quarter in the shock duration:
   - Override the affected exogenous variables with the shock values (additive or multiplicative as specified)
   - Ramp in linearly over `ramp_in_quarters`
3. Run the ensemble forecast on the shocked panel
4. Compare to the baseline (unshocked) forecast
5. Output: `StressTestResult` with baseline forecast, shocked forecast, and the delta

**CLI**: `econ stress --metro dallas_tx --scenario GFC_2008 --horizon 8`

---

## 6. Data Flow

```
                    External Sources
                    ────────────────
CoStar parquets ──► costar_panel.py ─┐
  (from msa)                          │
FRED API ─────────► fred_macro.py ───┤
Census BPS ───────► census_permits.py ┤──► panel_builder.py
IRS SOI ──────────► irs_migration.py ─┤    (quarterly aligned DataFrame)
CoStar panel ─────► concessions.py ──┘         │
                                          Optional: STL decomp
                                               │
                              ┌─────────────────┼────────────────────┐
                              ▼                 ▼                    ▼
                         VAREngine         ECMEngine          SpatialEngine
                         .fit(panel)       .fit(panel)        .fit(panel, W)
                         .predict(H)       .predict(H)        .predict(H)
                         .residuals()      .residuals()       .residuals()
                              │                 │                    │
                         RidgeLasso        GBMEngine                │
                         .fit(panel)       .fit(panel)              │
                         .predict(H)       .predict(H)              │
                              │                 │                    │
                              └─────────────────┴────────────────────┘
                                               │
                                    ensemble/weighter.py
                                      reads backtest registry
                                      inverse-RMSE weights
                                               │
                                    ensemble/combiner.py
                                      weighted point forecast
                                               │
                                    ensemble/confidence.py
                                      empirical 80%/95% bands
                                               │
                              ┌────────────────┼────────────────┐
                              ▼                ▼                ▼
                    backtest/registry.py  reports/           scenarios/
                    append forecast log   markdown_writer    stress_tester
                    (JSON, append-only)   chart_builder      shock_library
                                          → .md + .png       → stressed forecast
```

---

## 7. Metro Configuration Schema

**File**: `econometrics/config/metros/dallas_tx.yaml`

```yaml
metro_id: dallas_tx
display_name: "Dallas-Fort Worth Multifamily"
msa_code: "19100"           # CBSA code for BLS LAUS
state_fips: "48"
counties: ["085", "113", "121", "439"]   # Collin, Dallas, Denton, Tarrant

# CoStar data
costar_panel_path: "data/paid/costar-submarket/"  # relative to market-study-agent
costar_market_key: "dallas_multifamily"

# Submarkets (for spatial model W matrix)
submarkets:
  - id: far_north_dallas
    name: "Far North Dallas"
    lat: 32.988
    lon: -96.770
    star_rating: "1 & 2 Star"
  - id: addison_galleria
    name: "Addison/Galleria"
    lat: 32.954
    lon: -96.837
    star_rating: "All"
  - id: richardson_utd
    name: "Richardson/UTD"
    lat: 32.950
    lon: -96.730
    star_rating: "All"
  - id: plano
    name: "Plano"
    lat: 33.020
    lon: -96.699
    star_rating: "All"

# Model specification
endogenous_vars:
  - rent_growth_yoy
  - vacancy_rate
  - deliveries_pct_inventory
  - absorption_pct_inventory

exogenous_vars:
  - mortgage_rate_30y
  - unemployment_rate
  - net_migration_returns
  - income_delta_pct
  - permit_pipeline
  - concession_spread

# FRED series IDs
fred_series:
  mortgage_rate: "MORTGAGE30US"
  treasury_10y: "DGS10"
  fed_funds: "FEDFUNDS"
  baa_spread: "BAA10Y"
  term_spread: "T10Y2Y"

# BLS
bls_msa_code: "019100"     # 6-digit LAUS code

# Permit model
permit_lag_months: 18       # Calibrated by permit_lag.py
permit_cbsa: "19100"        # CBSA for Census BPS query

# Migration
migration_counties: ["085", "113"]  # Collin + Dallas for SOI
migration_interpolation: linear

# Spatial model
spatial_contiguity: inverse_distance

# Seasonality
decompose_seasonal: true

# Ensemble
enabled_models:
  - VAREngine
  - ECMEngine
  - RidgeLassoEngine
  - GBMEngine
  # - SpatialEngine  # Enable when libpysal installed
ensemble_decay: 0.9
min_backtest_quarters: 4

# Forecast defaults
default_horizon_quarters: 8
```

---

## 8. Output Specifications

### 8.1 Parquet (`data/econometrics/forecasts/{metro_id}/`)

**File**: `{metro_id}_forecast_{vintage_date}.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `period` | str | "2026-Q2", "2026-Q3", ... |
| `rent_growth_yoy_point` | float | Ensemble point forecast (%) |
| `rent_growth_yoy_ci80_lo` | float | 80% CI lower bound |
| `rent_growth_yoy_ci80_hi` | float | 80% CI upper bound |
| `rent_growth_yoy_ci95_lo` | float | 95% CI lower bound |
| `rent_growth_yoy_ci95_hi` | float | 95% CI upper bound |
| `model_weights` | str (JSON) | {"VAREngine": 0.35, "ECMEngine": 0.25, ...} |
| `vintage_date` | str | "2026-Q1" |

### 8.2 JSON Backtest Registry (`data/econometrics/registry/`)

See Section 5.1 schema.

### 8.3 Markdown Report (`data/econometrics/reports/`)

**File**: `forecast_{metro_id}_{vintage_date}.md`

Sections:
1. **Forecast Summary** — point forecast table with CI bands
2. **Model Weights** — current ensemble weights with backtest RMSE per model
3. **Key Drivers** — which variables moved most since last vintage
4. **Cointegration Status** — Johansen test results, error correction speed
5. **Spatial Dynamics** — rho coefficient, neighbor effects
6. **Scenario Analysis** — baseline vs. stress test comparison
7. **Backtest Accuracy** — rolling RMSE chart, coverage rates
8. **Data Sources** — citations with as-of dates

### 8.4 Charts (300 dpi PNG)

| Chart | Description |
|-------|-------------|
| `fan_chart.png` | Point forecast with shaded 80%/95% CI bands, historical actuals as solid line |
| `scenario_overlay.png` | Baseline + 2-3 stress scenarios on same axes |
| `irf_deliveries.png` | VAR impulse response: deliveries shock → rent growth response over 20Q |
| `irf_vacancy.png` | VAR impulse response: vacancy shock → rent growth response |
| `ensemble_weights.png` | Stacked bar of model weights over time (as backtest accumulates) |
| `backtest_accuracy.png` | Rolling RMSE by model, with ensemble RMSE overlaid |
| `permit_lag_calibration.png` | Lagged permits vs. actual deliveries (from permit_lag.py) |

---

## 9. Build Sequence

### Phase 1: Scaffold and Config (Week 1)
- [ ] Create directory structure with all `__init__.py` files
- [ ] Write `config/settings.py` with `EconSettings`
- [ ] Write `config/metros/_template.yaml` with full schema documentation
- [ ] Write `config/metros/dallas_tx.yaml` with DFW-specific values
- [ ] Update `pyproject.toml` (add `"econometrics*"` to packages, add optional extras)
- [ ] Update `.gitignore` (add `data/econometrics/registry/`)
- [ ] Verify `pip install -e ".[econometrics]"` works

### Phase 2: Data Loaders (Week 2)
- [ ] `costar_panel.py` — read CoStar parquets, output clean quarterly DataFrame
- [ ] `fred_macro.py` — FRED series with quarterly resampling
- [ ] `census_permits.py` — BPS permit series
- [ ] `irs_migration.py` — SOI parser with quarterly interpolation
- [ ] `concessions.py` — derived concession spread from CoStar panel
- [ ] `panel_builder.py` — merge all sources, validate no gaps > 2Q
- [ ] `permit_lag.py` — lag calibration with cross-correlation
- [ ] `spatial_weights.py` — W matrix from submarket coordinates
- [ ] `tests/econometrics/conftest.py` — synthetic panel fixture

### Phase 3: Core Models (Weeks 3-4)
- [ ] `models/base.py` — `ForecastModel` protocol + `ForecastResult` dataclass
- [ ] `models/stl_decomp.py` — STL wrapper (prerequisite for clean model input)
- [ ] `models/var_engine.py` — VAR with lag selection + IRF
- [ ] `models/ecm_engine.py` — Johansen + VECM with fallback
- [ ] `models/ridge_lasso.py` — expanding-window CV
- [ ] `models/gradient_boost.py` — LightGBM/sklearn wrapper (soft dep)
- [ ] `models/spatial_engine.py` — spatial lag model (soft dep)
- [ ] Tests for each model

### Phase 4: Backtest Infrastructure (Week 5)
- [ ] `backtest/registry.py` — append-only JSON with atomic writes
- [ ] `backtest/metrics.py` — RMSE, MAE, directional accuracy, CI coverage
- [ ] `backtest/runner.py` — expanding-window loop
- [ ] Tests for registry round-trip

### Phase 5: Ensemble and CIs (Week 5)
- [ ] `ensemble/weighter.py` — inverse-RMSE with cold-start fallback
- [ ] `ensemble/confidence.py` — empirical quantile CI
- [ ] `ensemble/combiner.py` — weighted forecast + CI assembly
- [ ] Tests for weighting and CI coverage

### Phase 6: Scenarios and Reports (Week 6)
- [ ] `scenarios/shock_library.py` — GFC_2008, COVID_2020, RATE_SHOCK_400BPS, SUPPLY_GLUT
- [ ] `scenarios/stress_tester.py` — shock application + ensemble re-forecast
- [ ] `reports/chart_builder.py` — fan chart, scenario overlay, IRF, backtest accuracy
- [ ] `reports/markdown_writer.py` — structured report with all sections

### Phase 7: CLI (Week 6)
- [ ] `cli/forecast.py` — full pipeline: load → build panel → fit → ensemble → report
- [ ] `cli/backtest.py` — fill actuals + accuracy report
- [ ] `cli/stress.py` — shock application
- [ ] Register `econ` entrypoint in `pyproject.toml`

### Phase 8: Demo and Documentation (Week 7)
- [ ] `notebooks/06_econometric_forecasting.ipynb` — end-to-end walkthrough
- [ ] Update this document with any changes made during implementation

---

## 10. Dependencies

### Required (add to `pyproject.toml` `dependencies`)
Already present: `statsmodels>=0.14`, `scipy>=1.10`, `scikit-learn>=1.3`, `numpy`, `pandas`, `matplotlib`

### Optional extras (new `[project.optional-dependencies]` group)
```toml
econometrics = [
    "pyarrow>=14.0",       # Parquet I/O
    "libpysal>=4.9",       # Spatial weights
    "spreg>=1.4",          # Spatial regression
    "lightgbm>=4.0",       # Gradient boosting
]
```

### Python version
`>=3.10` (matches existing project constraint). No 3.11+ syntax.

---

## 11. Testing Strategy

- All tests use **synthetic data** — no CoStar parquets, no API calls, no PostGIS
- **Deterministic seeds**: `np.random.default_rng(42)` everywhere
- **DGP-based validation**: generate data from a known VAR(2) or cointegrated system, fit the model, assert parameter recovery within tolerance
- **Edge case coverage**: empty panels, single-quarter data, missing exogenous variables, all-NaN columns
- **CI coverage tests**: over 1000 synthetic samples, assert 80% CI covers ~80% of actuals (tolerance ±5pp)
- **Regression direction tests**: high supply pressure → negative rent growth forecast; positive migration → positive rent growth forecast

---

## 12. Risk Register

| Risk | Mitigation |
|------|-----------|
| Short CoStar history (<40Q) limits VAR/ECM power | Fall back to Ridge with fewer features; log warning about low-confidence estimates |
| IRS SOI data lag (18 months) | Use latest available + linear extrapolation; flag stale data in reports |
| Spatial model requires ≥4 submarkets | `enabled_models` list in YAML; exclude SpatialEngine for metros with <4 submarkets |
| statsmodels Johansen test fragile for small samples | Wrap in try/except; fall back to Engle-Granger two-step if Johansen fails |
| Backtest registry file corruption | Atomic writes (tmp + rename); validate JSON schema on read |
| Ensemble weight instability with few backtests | Equal-weight cold start until min_backtest_quarters met |
