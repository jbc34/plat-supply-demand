# Supply-Demand Market Dynamics Estimator

A state-space estimation system for tracking market cycle dynamics through a novel "spiral coordinate" framework. The system estimates latent market state variables (Demand, Supply, Capital Flow) from observable proxies and transforms them into polar coordinates representing position and velocity in the market cycle.

## Core Concept

Markets oscillate around a long-term equilibrium (fundamental value). From the perspective of that equilibrium, Supply, Demand, and Capital Flow trace helical paths through time. This system estimates current position (r, θ) and velocity (ṙ, θ̇) in that spiral coordinate system.

## Features

- **Kalman Filter Estimation**: State-space model for estimating latent demand, supply, and capital flow
- **Spiral Coordinates**: Novel polar transformation for intuitive cycle positioning
- **Multiple Data Sources**: FRED, BLS, Census, and CoStar integration
- **Texas Multifamily Focus**: Pre-configured for Dallas, Houston, Austin, San Antonio markets
- **Interactive Dashboard**: Streamlit-based visualization of cycle position

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/supply-demand.git
cd supply-demand

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e ".[dev]"

# Copy environment template and add your API keys
cp .env.example .env
# Edit .env with your FRED_API_KEY and BLS_API_KEY
```

## Quick Start

### 1. Get API Keys (Free)

- **FRED API Key**: https://fred.stlouisfed.org/docs/api/api_key.html
- **BLS API Key**: https://data.bls.gov/registrationEngine/

### 2. Run Estimation

```python
from supply_demand.markets.multifamily.texas import TexasMultifamilyMarket
from supply_demand.models.coordinates import compute_spiral_coordinates

# Initialize Dallas market
market = TexasMultifamilyMarket(msa='dallas')

# Load data
data = market.load_data(start_date='2010-01-01')

# Get current cycle position
coords = market.get_current_position()

print(f"Cycle Phase: {coords.quadrant}")
print(f"Amplitude: {coords.r:.3f}")
print(f"Phase Angle: {coords.theta * 180 / 3.14159:.1f}°")
```

### 3. Launch Dashboard

```bash
python scripts/run_dashboard.py
# or
streamlit run supply_demand/visualization/dashboard.py
```

## Project Structure

```
supply-demand/
├── supply_demand/
│   ├── config/           # Configuration and settings
│   │   ├── settings.py   # Environment configuration
│   │   ├── markets.yaml  # Market definitions
│   │   └── proxies.yaml  # Observable proxy mappings
│   │
│   ├── data/
│   │   ├── sources/      # Data source clients
│   │   │   ├── fred.py   # FRED API client
│   │   │   ├── bls.py    # BLS API client
│   │   │   └── costar.py # CoStar data loader
│   │   └── pipelines/    # Data processing
│   │
│   ├── models/
│   │   ├── equilibrium/  # Equilibrium estimation
│   │   ├── state_space/  # Kalman filter
│   │   ├── dynamics/     # Coupled dynamics
│   │   └── coordinates/  # Polar transformation
│   │
│   ├── analysis/         # Validation and diagnostics
│   ├── markets/          # Market implementations
│   └── visualization/    # Plotting and dashboard
│
├── notebooks/            # Jupyter notebooks
├── tests/               # Test suite
└── scripts/             # CLI scripts
```

## Mathematical Framework

### State Vector
```
x_t = [D_t, S_t, K_t, dD_t, dS_t, dK_t]'
```
where D=Demand, S=Supply, K=Capital flow deviations from equilibrium, and dX=velocities.

### State Transition
```
x_{t+1} = F @ x_t + η_t
```
where F encodes coupled dynamics with damping.

### Observation
```
y_t = H @ x_t + ε_t
```
where H maps latent states to observable proxies.

### Polar Transform
Extract amplitude r and phase θ from state estimates:
- **r** = distance from equilibrium (cycle amplitude)
- **θ** = position in cycle (0-360°)

### Cycle Quadrants
- **0-90°**: Early Expansion
- **90-180°**: Peak / Late Expansion
- **180-270°**: Contraction
- **270-360°**: Recovery

## Configuration

### API Keys (.env)
```bash
FRED_API_KEY=your_key_here
BLS_API_KEY=your_key_here
COSTAR_API_KEY=your_key_here  # Optional
```

### Market Configuration (markets.yaml)
```yaml
markets:
  dallas_multifamily:
    name: "Dallas-Fort Worth Multifamily"
    msa_code: "19100"
    model_parameters:
      alpha_1: 0.3   # Income sensitivity
      beta_1: 0.25   # Price sensitivity
      tau_2: 18      # Construction lag (months)
```

## Notebooks

1. **01_data_exploration.ipynb** - Load and visualize data sources
2. **02_equilibrium_estimation.ipynb** - Estimate long-term equilibrium
3. **03_state_space_calibration.ipynb** - Calibrate Kalman filter
4. **04_polar_transformation.ipynb** - Transform to spiral coordinates
5. **05_validation.ipynb** - Backtest and validate model

## Running Tests

```bash
pytest tests/ -v
```

## Key Dependencies

- numpy, pandas, scipy - Numerical computing
- statsmodels - Statistical models
- matplotlib, plotly - Visualization
- streamlit - Dashboard
- requests - API clients

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black supply_demand/
ruff check supply_demand/

# Type checking
mypy supply_demand/
```

## Next Steps

1. **Get API keys** and configure `.env`
2. **Run notebook 01** to explore data
3. **Calibrate model** on historical data (2010-2020)
4. **Validate** against 2021-2024 data
5. **Deploy dashboard** for ongoing monitoring

## License

MIT

## Contributing

Contributions welcome! Please read the contribution guidelines first.
