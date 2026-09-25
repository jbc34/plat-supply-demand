"""NMHC (National Multifamily Housing Council) Survey Data Loader.

Loads and processes NMHC Quarterly Survey of Apartment Conditions data.

The survey measures:
- Market Tightness: Are apartment market conditions tighter, looser, or unchanged?
- Sales Volume: Is sales volume of apartment properties higher, lower, or unchanged?
- Equity Financing: Is equity financing more available, less available, or unchanged?
- Debt Financing: Are borrowing conditions improving, worsening, or unchanged?

Index interpretation:
- Above 50: Expansion/improvement in measured conditions
- Below 50: Contraction/deterioration in measured conditions
- Equal to 50: No change from previous quarter

Diffusion formula: 50 + 0.5*(positive% - negative%)
"""

import re
from pathlib import Path
from typing import Optional

import pandas as pd


class NMHCClient:
    """Client for loading NMHC survey data.

    Example:
        >>> client = NMHCClient()
        >>> data = client.load_data()
        >>> print(data.columns)
        >>> k_proxies = client.get_k_proxies()
    """

    # Column name mapping
    COLUMN_MAPPING = {
        'Market Tightness Index1': 'market_tightness',
        'Sales Volume Index2': 'sales_volume',
        'Equity Financing Index3': 'equity_financing',
        'Debt Financing Index4': 'debt_financing',
    }

    # K proxy columns (capital availability indicators)
    K_PROXY_COLUMNS = ['equity_financing', 'debt_financing', 'sales_volume']

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize NMHC client.

        Args:
            data_path: Path to NMHC Excel file. If None, uses default location.
        """
        if data_path is None:
            # Default path relative to project root
            self.data_path = Path(__file__).parent.parent.parent.parent / "data" / "NMHC conditions.xlsx"
        else:
            self.data_path = Path(data_path)

        self._data = None

    def _clean_value(self, value) -> float:
        """Clean index value by removing footnote markers."""
        if pd.isna(value):
            return float('nan')

        if isinstance(value, (int, float)):
            return float(value)

        # Convert to string and remove non-numeric characters except decimal point
        value_str = str(value)
        cleaned = re.sub(r'[^\d.]', '', value_str)

        if cleaned == '':
            return float('nan')

        return float(cleaned)

    def load_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Load and clean NMHC survey data.

        Args:
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            DataFrame with cleaned NMHC indices
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"NMHC data file not found: {self.data_path}")

        # Read Excel file
        df = pd.read_excel(self.data_path)

        # Find the date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break

        if date_col is None:
            raise ValueError("Could not find date column in NMHC data")

        # Set date as index
        df['date'] = pd.to_datetime(df[date_col])
        df = df.set_index('date').sort_index()

        # Drop unnamed columns
        df = df[[c for c in df.columns if 'unnamed' not in c.lower() and c != date_col]]

        # Rename columns
        rename_dict = {}
        for old_name, new_name in self.COLUMN_MAPPING.items():
            for col in df.columns:
                if old_name.lower() in col.lower() or new_name in col.lower():
                    rename_dict[col] = new_name
                    break

        df = df.rename(columns=rename_dict)

        # Clean values (remove footnote markers)
        for col in df.columns:
            df[col] = df[col].apply(self._clean_value)

        # Filter by date range
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        self._data = df
        return df

    def get_k_proxies(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Get K (capital) proxy columns from NMHC data.

        Returns columns relevant for capital availability:
        - equity_financing: Equity capital availability
        - debt_financing: Debt capital availability
        - sales_volume: Transaction activity (reflects capital deployment)

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with K proxy columns
        """
        if self._data is None:
            self.load_data(start_date, end_date)

        # Get available K proxy columns
        available_cols = [c for c in self.K_PROXY_COLUMNS if c in self._data.columns]

        if not available_cols:
            raise ValueError("No K proxy columns found in NMHC data")

        k_data = self._data[available_cols].copy()

        # Filter by date if data was already loaded
        if start_date:
            k_data = k_data[k_data.index >= pd.to_datetime(start_date)]
        if end_date:
            k_data = k_data[k_data.index <= pd.to_datetime(end_date)]

        return k_data

    def get_capital_index(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.Series:
        """Get composite capital availability index.

        Combines equity and debt financing indices into single K proxy.

        Returns:
            Series with composite capital index
        """
        k_proxies = self.get_k_proxies(start_date, end_date)

        # Average of equity and debt financing
        if 'equity_financing' in k_proxies.columns and 'debt_financing' in k_proxies.columns:
            capital_index = (k_proxies['equity_financing'] + k_proxies['debt_financing']) / 2
        elif 'equity_financing' in k_proxies.columns:
            capital_index = k_proxies['equity_financing']
        elif 'debt_financing' in k_proxies.columns:
            capital_index = k_proxies['debt_financing']
        else:
            raise ValueError("Neither equity nor debt financing columns available")

        capital_index.name = 'nmhc_capital_index'
        return capital_index

    def describe(self) -> dict:
        """Get summary statistics for NMHC data."""
        if self._data is None:
            self.load_data()

        return {
            'date_range': {
                'start': str(self._data.index.min().date()),
                'end': str(self._data.index.max().date()),
            },
            'n_observations': len(self._data),
            'columns': list(self._data.columns),
            'summary_stats': self._data.describe().to_dict(),
        }
