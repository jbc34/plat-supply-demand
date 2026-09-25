"""Streamlit dashboard for market cycle monitoring."""


def create_dashboard():
    """Create Streamlit dashboard application.

    Run with: streamlit run supply_demand/visualization/dashboard.py
    """
    try:
        import streamlit as st
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
        import numpy as np
    except ImportError:
        print("Streamlit and Plotly required. Install with: pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="Market Cycle Monitor",
        page_icon="📊",
        layout="wide",
    )

    st.title("Market Cycle Monitor")
    st.markdown("Real-time tracking of market cycle position using spiral coordinates")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    market = st.sidebar.selectbox(
        "Select Market",
        ["Dallas Multifamily", "Houston Multifamily", "Austin Multifamily", "San Antonio Multifamily"]
    )

    # Demo data (replace with actual data in production)
    np.random.seed(42)
    n_points = 120  # 10 years monthly

    # Generate synthetic cycle data
    t = np.linspace(0, 4 * np.pi, n_points)
    cycle = np.sin(t) + 0.3 * np.sin(2 * t)
    amplitude = 1 + 0.3 * np.sin(0.5 * t)

    dates = pd.date_range(start="2015-01-01", periods=n_points, freq="MS")

    # Compute spiral coordinates
    r = np.abs(cycle) * amplitude
    theta = t % (2 * np.pi)

    # Current position
    current_r = r[-1]
    current_theta = theta[-1]
    theta_deg = np.degrees(current_theta)

    # Determine quadrant
    if 0 <= theta_deg < 90:
        quadrant = "Early Expansion"
        quadrant_color = "green"
    elif 90 <= theta_deg < 180:
        quadrant = "Late Expansion / Peak"
        quadrant_color = "yellow"
    elif 180 <= theta_deg < 270:
        quadrant = "Contraction"
        quadrant_color = "red"
    else:
        quadrant = "Recovery"
        quadrant_color = "blue"

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Current Phase",
            value=quadrant,
        )

    with col2:
        st.metric(
            label="Amplitude (r)",
            value=f"{current_r:.2f}",
            delta=f"{r[-1] - r[-2]:.3f}",
        )

    with col3:
        st.metric(
            label="Phase Angle",
            value=f"{theta_deg:.1f}°",
        )

    with col4:
        st.metric(
            label="Cycle Progress",
            value=f"{(theta_deg / 360 * 100):.0f}%",
        )

    # Charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Spiral Position")

        # Create polar plot
        fig_polar = go.Figure()

        # Add trajectory
        fig_polar.add_trace(go.Scatterpolar(
            r=r,
            theta=np.degrees(theta),
            mode='lines',
            line=dict(color='lightblue', width=2),
            name='Historical'
        ))

        # Add current position
        fig_polar.add_trace(go.Scatterpolar(
            r=[current_r],
            theta=[theta_deg],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Current'
        ))

        # Add quadrant annotations
        for angle, label in [(45, "Expansion"), (135, "Peak"), (225, "Contraction"), (315, "Recovery")]:
            fig_polar.add_annotation(
                x=0.5 + 0.35 * np.cos(np.radians(angle)),
                y=0.5 + 0.35 * np.sin(np.radians(angle)),
                text=label,
                showarrow=False,
                xref="paper",
                yref="paper",
            )

        fig_polar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(r) * 1.2]),
                angularaxis=dict(direction="clockwise"),
            ),
            showlegend=True,
            height=400,
        )

        st.plotly_chart(fig_polar, use_container_width=True)

    with col_right:
        st.subheader("Time Series")

        # Create time series plot
        fig_ts = go.Figure()

        fig_ts.add_trace(go.Scatter(
            x=dates,
            y=r,
            mode='lines',
            name='Amplitude (r)',
            line=dict(color='blue'),
        ))

        fig_ts.add_trace(go.Scatter(
            x=dates,
            y=theta / np.pi,
            mode='lines',
            name='Phase (π)',
            line=dict(color='orange'),
            yaxis='y2',
        ))

        fig_ts.update_layout(
            yaxis=dict(title='Amplitude'),
            yaxis2=dict(title='Phase (π)', overlaying='y', side='right'),
            height=400,
            legend=dict(x=0, y=1.1, orientation='h'),
        )

        st.plotly_chart(fig_ts, use_container_width=True)

    # Detailed metrics
    st.subheader("Detailed Metrics")

    metrics_df = pd.DataFrame({
        "Metric": [
            "Current Phase",
            "Amplitude (r)",
            "Phase Angle (θ)",
            "Radial Velocity (ṙ)",
            "Angular Velocity (θ̇)",
            "Months Since Last Trough",
            "Months to Next Quadrant",
        ],
        "Value": [
            quadrant,
            f"{current_r:.3f}",
            f"{theta_deg:.1f}°",
            f"{(r[-1] - r[-2]):.4f}",
            f"{np.degrees(theta[-1] - theta[-2]):.2f}°/month",
            f"{np.argmax(r[::-1] < 0.3)}" if any(r < 0.3) else "N/A",
            f"{(90 - theta_deg % 90) / 3:.1f}",  # Rough estimate
        ],
        "Interpretation": [
            f"Market is in {quadrant.lower()} phase",
            "Distance from equilibrium",
            "Position in cycle",
            "Amplitude increasing" if r[-1] > r[-2] else "Amplitude decreasing",
            "Cycle speed",
            "Time since last minimum",
            "Estimated at current velocity",
        ],
    })

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This dashboard uses demonstration data. "
        "Connect to real data sources for production use."
    )


if __name__ == "__main__":
    create_dashboard()
