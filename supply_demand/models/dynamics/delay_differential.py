"""Coupled delay differential equations for market dynamics.

Models the interactions between Demand, Supply, and Capital Flow
with time delays that capture real-world response lags.

The continuous-time system:
    dD/dt = f_D(D, S, K, P; alpha) + eta_D
    dS/dt = f_S(D, S, K, P; beta) + eta_S
    dK/dt = f_K(D, S, K, P; gamma) + eta_K
    dP/dt = lambda * (D - S)

Where:
    - D, S, K = Demand, Supply, Capital deviations from equilibrium
    - P = Price deviation from equilibrium
    - alpha, beta, gamma = Parameter sets
    - lambda = Price adjustment speed
    - eta = Noise terms
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class SystemState:
    """State of the dynamic system at a point in time."""

    D: float   # Demand deviation
    S: float   # Supply deviation
    K: float   # Capital flow deviation
    dD: float  # Demand velocity
    dS: float  # Supply velocity
    dK: float  # Capital velocity
    P: float = 0.0  # Price deviation (optional)


class CoupledDynamics:
    """Continuous-time coupled dynamics of market variables.

    Models how Demand, Supply, and Capital Flow interact over time
    with configurable coupling strengths and delays.
    """

    def __init__(
        self,
        alpha_1: float = 0.3,   # Income -> Demand
        alpha_2: float = 0.2,   # Price -> Demand
        alpha_3: float = 0.15,  # Capital -> Demand
        beta_1: float = 0.25,   # Price -> Supply
        beta_2: float = 0.2,    # Capital -> Supply
        beta_3: float = 0.1,    # Supply adjustment
        gamma_1: float = 0.3,   # Return gap -> Capital
        gamma_2: float = 0.15,  # Price accel -> Capital
        gamma_3: float = 0.2,   # Imbalance -> Capital
        lambda_: float = 0.1,   # Price adjustment speed
        damping: float = 0.1,   # Overall damping factor
    ):
        """Initialize coupled dynamics.

        Args:
            alpha_*: Demand equation parameters
            beta_*: Supply equation parameters
            gamma_*: Capital equation parameters
            lambda_: Price adjustment speed
            damping: Velocity damping coefficient
        """
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.gamma_3 = gamma_3
        self.lambda_ = lambda_
        self.damping = damping

    def demand_dynamics(
        self,
        D: float,
        S: float,
        K: float,
        dD: float,
        income_shock: float = 0.0,
    ) -> float:
        """Demand velocity equation.

        dD/dt depends on:
        - Price effects (via S-D imbalance)
        - Capital flow effects
        - Own damping
        - Exogenous income shocks

        Args:
            D: Current demand deviation
            S: Current supply deviation
            K: Current capital flow
            dD: Current demand velocity
            income_shock: Exogenous income change

        Returns:
            Rate of change of demand velocity (d²D/dt²)
        """
        # Price effect: excess demand raises prices, slowing demand growth
        price_effect = -self.alpha_2 * self.lambda_ * (D - S)

        # Capital flow effect: more capital enables demand
        capital_effect = self.alpha_3 * K

        # Income effect
        income_effect = self.alpha_1 * income_shock

        # Damping
        damping = -self.damping * dD

        return price_effect + capital_effect + income_effect + damping

    def supply_dynamics(
        self,
        D: float,
        S: float,
        K: float,
        dS: float,
        cost_shock: float = 0.0,
    ) -> float:
        """Supply velocity equation.

        dS/dt depends on:
        - Price effects (via D-S imbalance)
        - Capital flow effects
        - Own mean reversion
        - Exogenous cost shocks

        Args:
            D: Current demand deviation
            S: Current supply deviation
            K: Current capital flow
            dS: Current supply velocity
            cost_shock: Exogenous cost change

        Returns:
            Rate of change of supply velocity (d²S/dt²)
        """
        # Price effect: higher prices (excess demand) attract supply
        price_effect = self.beta_1 * self.lambda_ * (D - S)

        # Capital flow effect: more capital enables development
        capital_effect = self.beta_2 * K

        # Mean reversion / adjustment
        adjustment = -self.beta_3 * S

        # Cost effect (negative - higher costs slow supply)
        cost_effect = -cost_shock

        # Damping
        damping = -self.damping * dS

        return price_effect + capital_effect + adjustment + cost_effect + damping

    def capital_dynamics(
        self,
        D: float,
        S: float,
        K: float,
        dK: float,
        dD: float,
        rate_shock: float = 0.0,
    ) -> float:
        """Capital flow velocity equation.

        dK/dt depends on:
        - Return gap (via D-S imbalance)
        - Price acceleration
        - Own mean reversion
        - Interest rate shocks

        Args:
            D: Current demand deviation
            S: Current supply deviation
            K: Current capital flow
            dK: Current capital velocity
            dD: Demand velocity (proxy for price acceleration)
            rate_shock: Interest rate change

        Returns:
            Rate of change of capital velocity (d²K/dt²)
        """
        # Return gap: excess demand implies higher returns, attracting capital
        return_effect = self.gamma_1 * (D - S)

        # Price acceleration: rising prices attract capital
        acceleration_effect = self.gamma_2 * dD

        # Imbalance sensitivity
        imbalance_effect = self.gamma_3 * (D - S)

        # Rate effect (negative - higher rates reduce capital flow)
        rate_effect = -rate_shock

        # Damping (capital is stickier)
        damping = -self.damping * 1.5 * dK

        return return_effect + acceleration_effect + imbalance_effect + rate_effect + damping

    def system_rhs(
        self,
        t: float,
        y: np.ndarray,
        shocks: Optional[Callable[[float], Tuple[float, float, float]]] = None,
    ) -> np.ndarray:
        """Right-hand side of the full ODE system.

        State vector y = [D, S, K, dD, dS, dK]

        Args:
            t: Time
            y: State vector
            shocks: Function returning (income, cost, rate) shocks at time t

        Returns:
            dy/dt vector
        """
        D, S, K, dD, dS, dK = y

        # Get exogenous shocks
        if shocks is not None:
            income_shock, cost_shock, rate_shock = shocks(t)
        else:
            income_shock = cost_shock = rate_shock = 0.0

        # Compute derivatives
        dy = np.zeros(6)

        # Position derivatives
        dy[0] = dD  # dD/dt = velocity
        dy[1] = dS
        dy[2] = dK

        # Velocity derivatives (accelerations)
        dy[3] = self.demand_dynamics(D, S, K, dD, income_shock)
        dy[4] = self.supply_dynamics(D, S, K, dS, cost_shock)
        dy[5] = self.capital_dynamics(D, S, K, dK, dD, rate_shock)

        return dy

    def simulate(
        self,
        y0: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
        shocks: Optional[Callable] = None,
        method: str = "RK45",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the system forward in time.

        Args:
            y0: Initial state [D, S, K, dD, dS, dK]
            t_span: (t_start, t_end)
            t_eval: Times at which to return solution
            shocks: Exogenous shock function
            method: Integration method

        Returns:
            Tuple of (time points, state trajectories)
        """
        def rhs(t, y):
            return self.system_rhs(t, y, shocks)

        result = solve_ivp(
            rhs,
            t_span,
            y0,
            method=method,
            t_eval=t_eval,
            dense_output=True,
        )

        return result.t, result.y.T


class DelayDifferentialSystem(CoupledDynamics):
    """Extension with explicit time delays.

    Delays are important because:
    - Construction takes time (tau_construction)
    - Capital deployment has friction (tau_capital)
    - Information diffuses slowly (tau_info)
    """

    def __init__(
        self,
        tau_construction: int = 18,  # Months from start to delivery
        tau_capital: int = 6,        # Months for capital deployment
        tau_info: int = 2,           # Months for information diffusion
        **kwargs
    ):
        """Initialize delay differential system.

        Args:
            tau_construction: Construction delay in months
            tau_capital: Capital deployment delay
            tau_info: Information delay
            **kwargs: Parameters passed to CoupledDynamics
        """
        super().__init__(**kwargs)
        self.tau_construction = tau_construction
        self.tau_capital = tau_capital
        self.tau_info = tau_info

    def simulate_with_delays(
        self,
        y0: np.ndarray,
        T: int,
        dt: float = 1.0,
        shocks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Simulate with explicit handling of delays.

        Uses a simple scheme where delayed values are looked up
        from the history.

        Args:
            y0: Initial state
            T: Number of time steps
            dt: Time step size
            shocks: T x 3 array of (income, cost, rate) shocks

        Returns:
            T x 6 array of state trajectories
        """
        n_states = 6
        max_delay = max(
            self.tau_construction,
            self.tau_capital,
            self.tau_info
        )

        # Initialize with extended history
        history = np.zeros((T + max_delay, n_states))
        history[max_delay] = y0

        for t in range(max_delay, T + max_delay - 1):
            y = history[t]
            D, S, K, dD, dS, dK = y

            # Get delayed values
            D_delayed = history[t - self.tau_info, 0]
            S_delayed = history[t - self.tau_construction, 1]
            K_delayed = history[t - self.tau_capital, 2]

            # Get shocks
            if shocks is not None and t - max_delay < len(shocks):
                income, cost, rate = shocks[t - max_delay]
            else:
                income = cost = rate = 0.0

            # Simple Euler integration with delays
            dy = np.zeros(n_states)

            dy[0] = dD
            dy[1] = dS
            dy[2] = dK

            # Use delayed values in dynamics
            price_effect_D = -self.alpha_2 * self.lambda_ * (D_delayed - S_delayed)
            dy[3] = price_effect_D + self.alpha_3 * K_delayed - self.damping * dD

            price_effect_S = self.beta_1 * self.lambda_ * (D_delayed - S_delayed)
            dy[4] = price_effect_S + self.beta_2 * K_delayed - self.beta_3 * S - self.damping * dS

            return_effect = self.gamma_1 * (D_delayed - S_delayed)
            dy[5] = return_effect + self.gamma_2 * dD - self.damping * 1.5 * dK

            history[t + 1] = y + dy * dt

        # Return just the simulation period
        return history[max_delay:T + max_delay]
