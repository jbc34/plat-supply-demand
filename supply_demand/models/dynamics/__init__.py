"""Dynamic system models for market interactions."""

from supply_demand.models.dynamics.delay_differential import (
    DelayDifferentialSystem,
    CoupledDynamics,
)
from supply_demand.models.dynamics.discretization import (
    discretize_continuous_system,
    euler_discretization,
    matrix_exponential_discretization,
)
from supply_demand.models.dynamics.stability import (
    analyze_stability,
    compute_eigenvalues,
    StabilityResult,
)

__all__ = [
    "DelayDifferentialSystem",
    "CoupledDynamics",
    "discretize_continuous_system",
    "euler_discretization",
    "matrix_exponential_discretization",
    "analyze_stability",
    "compute_eigenvalues",
    "StabilityResult",
]
