"""Spiral visualization for market state."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def plot_spiral(
    r: float,
    theta: float,
    title: str = "Market Cycle Position",
    show_quadrants: bool = True,
    fig_size: Tuple[int, int] = (8, 8),
):
    """Plot current position on spiral/polar diagram.

    Args:
        r: Radial distance (amplitude)
        theta: Angular position (phase)
        title: Plot title
        show_quadrants: Whether to show cycle quadrants
        fig_size: Figure size

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(projection='polar'))

    # Plot current position
    ax.plot(theta, r, 'ro', markersize=15, label='Current Position')

    # Add quadrant labels
    if show_quadrants:
        quadrants = [
            (np.pi / 4, "Early\nExpansion"),
            (3 * np.pi / 4, "Late\nExpansion"),
            (5 * np.pi / 4, "Contraction"),
            (7 * np.pi / 4, "Recovery"),
        ]

        max_r = max(r * 1.5, 1.0)
        for angle, label in quadrants:
            ax.annotate(
                label,
                xy=(angle, max_r * 0.7),
                ha='center',
                va='center',
                fontsize=10,
            )

    # Draw quadrant lines
    for angle in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
        ax.plot([angle, angle], [0, max(r * 1.5, 1.0)], 'k--', alpha=0.3)

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_ylim(0, max(r * 1.5, 1.0))

    return fig


def plot_spiral_trajectory(
    r: np.ndarray,
    theta: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Market Cycle Trajectory",
    highlight_current: bool = True,
    fig_size: Tuple[int, int] = (10, 8),
):
    """Plot spiral trajectory over time.

    Args:
        r: Amplitude time series
        theta: Phase time series
        dates: Optional datetime index for coloring
        title: Plot title
        highlight_current: Highlight current position
        fig_size: Figure size

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(projection='polar'))

    # Create line segments for color mapping
    points = np.array([theta, r]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color by time
    colors = np.linspace(0, 1, len(r))

    lc = LineCollection(segments, cmap='viridis', alpha=0.7)
    lc.set_array(colors)
    lc.set_linewidth(2)
    ax.add_collection(lc)

    # Highlight current position
    if highlight_current:
        ax.plot(theta[-1], r[-1], 'ro', markersize=12, label='Current', zorder=5)

    # Add quadrant labels
    max_r = np.max(r) * 1.2
    quadrants = [
        (np.pi / 4, "Expansion"),
        (3 * np.pi / 4, "Peak"),
        (5 * np.pi / 4, "Contraction"),
        (7 * np.pi / 4, "Recovery"),
    ]

    for angle, label in quadrants:
        ax.annotate(
            label,
            xy=(angle, max_r * 0.85),
            ha='center',
            va='center',
            fontsize=9,
            alpha=0.7,
        )

    ax.set_ylim(0, max_r)
    ax.set_title(title, fontsize=14, pad=20)

    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax, pad=0.1)
    if dates is not None:
        cbar.set_label('Time')
        n_ticks = 5
        tick_positions = np.linspace(0, 1, n_ticks)
        tick_labels = [dates[int(i * (len(dates) - 1))].strftime('%Y') for i in tick_positions]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)

    ax.legend(loc='upper right')

    return fig


def plot_3d_spiral(
    r: np.ndarray,
    theta: np.ndarray,
    time: np.ndarray,
    title: str = "3D Market Spiral",
    fig_size: Tuple[int, int] = (12, 8),
):
    """Plot 3D spiral with time as vertical axis.

    Args:
        r: Amplitude time series
        theta: Phase time series
        time: Time values for z-axis
        title: Plot title
        fig_size: Figure size

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    # Convert to cartesian for 3D plot
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = time

    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(time)))

    # Plot trajectory
    for i in range(len(time) - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], z[i:i + 2], color=colors[i], linewidth=2)

    # Highlight current position
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color='red', s=100, label='Current')

    # Add equilibrium line
    ax.plot([0, 0], [0, 0], [z[0], z[-1]], 'k--', alpha=0.5, label='Equilibrium')

    ax.set_xlabel('X (D-S)')
    ax.set_ylabel('Y (K)')
    ax.set_zlabel('Time')
    ax.set_title(title)
    ax.legend()

    return fig
