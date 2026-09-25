#!/usr/bin/env python
"""Launch the Streamlit dashboard.

Usage:
    python scripts/run_dashboard.py
    # or
    streamlit run supply_demand/visualization/dashboard.py
"""

import subprocess
import sys


def main():
    print("Launching Market Cycle Dashboard...")
    print("Open http://localhost:8501 in your browser")
    print("Press Ctrl+C to stop\n")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "supply_demand/visualization/dashboard.py",
        "--server.headless=true",
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
