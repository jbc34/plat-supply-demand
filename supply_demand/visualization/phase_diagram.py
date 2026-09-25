"""Phase diagram visualization."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def plot_phase_diagram(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str = "State Variable 1",
    y_label: str = "State Variable 2",
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Phase Diagram",
    show_equilibrium: bool = True,
    fig_size: Tuple[int, int] = (10, 8),
):
    """Plot 2D phase diagram.

    Shows the trajectory of two state variables against each other.

    Args:
        x: First state variable
        y: Second state variable
        x_label: Label for x-axis
        y_label: Label for y-axis
        dates: Optional datetime index for coloring
        title: Plot title
        show_equilibrium: Show equilibrium point at origin
        fig_size: Figure size

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=fig_size)

    # Color by time
    colors = np.linspace(0, 1, len(x))
    scatter = ax.scatter(x, y, c=colors, cmap='viridis', alpha=0.6, s=20)

    # Connect points with lines
    ax.plot(x, y, 'b-', alpha=0.3, linewidth=0.5)

    # Highlight current position
    ax.scatter([x[-1]], [y[-1]], c='red', s=100, zorder=5, label='Current')

    # Show equilibrium
    if show_equilibrium:
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.scatter([0], [0], c='green', s=100, marker='x', label='Equilibrium')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(scatter)
    if dates is not None:
        cbar.set_label('Time')

    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_state_trajectories(
    states: pd.DataFrame,
    state_names: Optional[list] = None,
    title: str = "State Trajectories",
    fig_size: Tuple[int, int] = (14, 10),
):
    """Plot time series of all state variables.

    Args:
        states: DataFrame with state estimates
        state_names: Names to include (default: all)
        title: Plot title
        fig_size: Figure size

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    if state_names is None:
        state_names = [c for c in states.columns if not c.endswith('_se')]

    n_states = len(state_names)
    fig, axes = plt.subplots(n_states, 1, figsize=fig_size, sharex=True)

    if n_states == 1:
        axes = [axes]

    for ax, name in zip(axes, state_names):
        ax.plot(states.index, states[name], 'b-', linewidth=1)

        # Add confidence band if available
        se_col = f"{name}_se"
        if se_col in states.columns:
            ax.fill_between(
                states.index,
                states[name] - 2 * states[se_col],
                states[name] + 2 * states[se_col],
                alpha=0.2,
            )

        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig
