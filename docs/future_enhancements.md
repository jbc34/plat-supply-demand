# Future Enhancements

Notes on potential features, data sources, and improvements to explore.

---

## Capital (K) Proxy Improvements

The current Capital (K) state variable uses MORTGAGE30US and DGS10 as proxies. These additional data sources could provide better signal for capital flows into real estate:

### CMBS Spreads
- **Description**: Commercial Mortgage-Backed Securities spreads over treasuries
- **Rationale**: Directly reflects cost and availability of commercial real estate debt
- **Potential sources**:
  - Bloomberg CMBS indices
  - Trepp CMBS data
  - FRED may have some aggregate measures
- **Loading**: Negative (wider spreads = less capital availability)

### Freddie Mac Spreads
- **Description**: Freddie Mac multifamily loan spreads
- **Rationale**: Direct measure of multifamily financing costs vs risk-free rate
- **Potential sources**:
  - Freddie Mac Multifamily Securitization data
  - FRED series (search for Freddie Mac spreads)
- **Loading**: Negative (wider spreads = tighter capital)

### Chicago Fed National Financial Conditions Index (NFCI)
- **FRED Series**: `NFCI`
- **Description**: Composite index of financial conditions (credit, leverage, risk)
- **Rationale**: Broad measure of financial system stress/ease
- **Loading**: Negative (higher NFCI = tighter conditions = less capital)
- **Sub-indices available**:
  - `NFCICREDIT` - Credit subindex
  - `NFCILEVERAGE` - Leverage subindex
  - `NFCIRISK` - Risk subindex

### Other Potential Capital Proxies
- **Bank Lending Standards** (`DRTSCILM`, `DRTSCLCC` in FRED) - Survey of bank lending tightness
- **REIT Capital Flows** - Public REIT equity/debt issuance
- **Transaction Volumes** - Real Capital Analytics or CoStar transaction data
- **Private Equity Dry Powder** - Capital waiting to be deployed in RE
- **Foreign Capital Flows** - Cross-border RE investment data

---

## Implementation Notes

To add a new proxy to the Texas Multifamily market:

1. Add data fetching in `supply_demand/data/sources/fred.py` (or new source file)
2. Update `get_proxy_config()` in `supply_demand/markets/multifamily/texas.py`:
   ```python
   "nfci": {
       "state": "K",
       "loading": -1.0,  # Negative: higher NFCI = tighter conditions
       "column": "NFCI",
   },
   ```
3. Ensure data alignment (monthly frequency, same date range)
4. Re-run calibration to estimate optimal loadings

---

## Other Future Enhancements

### Data Sources
- [ ] CoStar API integration for rent/vacancy actuals
- [ ] Census ACS data for demographic demand drivers
- [ ] Yardi or RealPage market data

### Model Improvements
- [ ] Time-varying parameters (regime switching)
- [ ] Regional spillover effects between Texas MSAs
- [ ] Forecast confidence intervals

### Validation
- [ ] Backtest against actual rent growth
- [ ] Compare to Mueller cycle methodology
- [ ] Out-of-sample forecast accuracy

---

*Last updated: 2026-01-14*
