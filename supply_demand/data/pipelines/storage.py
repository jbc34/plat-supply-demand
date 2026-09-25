"""Data storage and database operations.

Handles persistent storage of fetched and processed data.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from supply_demand.config.settings import settings


class DataStorage:
    """Persistent storage for time series data.

    Supports:
    - SQLite for development
    - PostgreSQL for production
    - Parquet files for large datasets

    Example:
        >>> storage = DataStorage()
        >>> storage.save_series("fred_mortgage", mortgage_series)
        >>> loaded = storage.load_series("fred_mortgage")
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        data_dir: Optional[Path] = None
    ):
        """Initialize data storage.

        Args:
            database_url: SQLAlchemy database URL
            data_dir: Directory for file-based storage
        """
        self.database_url = database_url or settings.DATABASE_URL
        self.data_dir = data_dir or settings.ensure_data_dir()
        self._engine: Optional[Engine] = None

    @property
    def engine(self) -> Engine:
        """Lazy initialization of database engine."""
        if self._engine is None:
            self._engine = create_engine(self.database_url)
            self._initialize_schema()
        return self._engine

    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        with self.engine.connect() as conn:
            # Create time series table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS time_series (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    source TEXT,
                    frequency TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name)
                )
            """))

            # Create observations table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS observations (
                    series_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    value REAL,
                    PRIMARY KEY (series_id, date),
                    FOREIGN KEY (series_id) REFERENCES time_series(id)
                )
            """))

            # Create metadata table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            conn.commit()

    def save_series(
        self,
        name: str,
        series: pd.Series,
        source: Optional[str] = None,
        frequency: Optional[str] = None,
        overwrite: bool = True
    ) -> None:
        """Save a time series to storage.

        Args:
            name: Unique name for the series
            series: pandas Series with datetime index
            source: Data source identifier
            frequency: Data frequency
            overwrite: Whether to overwrite existing data
        """
        # Infer frequency if not provided
        if frequency is None and isinstance(series.index, pd.DatetimeIndex):
            frequency = pd.infer_freq(series.index) or "unknown"

        with self.engine.connect() as conn:
            # Get or create series record
            result = conn.execute(
                text("SELECT id FROM time_series WHERE name = :name"),
                {"name": name}
            ).fetchone()

            if result:
                series_id = result[0]
                if overwrite:
                    conn.execute(
                        text("DELETE FROM observations WHERE series_id = :id"),
                        {"id": series_id}
                    )
                    conn.execute(
                        text("""
                            UPDATE time_series
                            SET source = :source, frequency = :freq, updated_at = :now
                            WHERE id = :id
                        """),
                        {"source": source, "freq": frequency,
                         "now": datetime.now(), "id": series_id}
                    )
            else:
                conn.execute(
                    text("""
                        INSERT INTO time_series (name, source, frequency)
                        VALUES (:name, :source, :freq)
                    """),
                    {"name": name, "source": source, "freq": frequency}
                )
                result = conn.execute(
                    text("SELECT id FROM time_series WHERE name = :name"),
                    {"name": name}
                ).fetchone()
                series_id = result[0]

            # Insert observations
            for date, value in series.items():
                if pd.notna(value):
                    conn.execute(
                        text("""
                            INSERT OR REPLACE INTO observations (series_id, date, value)
                            VALUES (:sid, :date, :value)
                        """),
                        {"sid": series_id, "date": date.date(), "value": float(value)}
                    )

            conn.commit()

    def load_series(
        self,
        name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.Series]:
        """Load a time series from storage.

        Args:
            name: Series name
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            pandas Series or None if not found
        """
        query = """
            SELECT o.date, o.value
            FROM observations o
            JOIN time_series t ON o.series_id = t.id
            WHERE t.name = :name
        """
        params = {"name": name}

        if start_date:
            query += " AND o.date >= :start"
            params["start"] = start_date

        if end_date:
            query += " AND o.date <= :end"
            params["end"] = end_date

        query += " ORDER BY o.date"

        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()

        if not rows:
            return None

        dates = [row[0] for row in rows]
        values = [row[1] for row in rows]

        return pd.Series(values, index=pd.to_datetime(dates), name=name)

    def save_dataframe(
        self,
        name: str,
        df: pd.DataFrame,
        format: str = "parquet"
    ) -> Path:
        """Save a DataFrame to file storage.

        Args:
            name: Name for the file
            df: DataFrame to save
            format: File format (parquet, csv)

        Returns:
            Path to saved file
        """
        if format == "parquet":
            filepath = self.data_dir / f"{name}.parquet"
            df.to_parquet(filepath)
        elif format == "csv":
            filepath = self.data_dir / f"{name}.csv"
            df.to_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return filepath

    def load_dataframe(
        self,
        name: str,
        format: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Load a DataFrame from file storage.

        Args:
            name: File name (without extension)
            format: File format (auto-detect if None)

        Returns:
            DataFrame or None if not found
        """
        # Try parquet first
        parquet_path = self.data_dir / f"{name}.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)

        # Try CSV
        csv_path = self.data_dir / f"{name}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path, index_col=0, parse_dates=True)

        return None

    def list_series(self) -> pd.DataFrame:
        """List all stored time series.

        Returns:
            DataFrame with series metadata
        """
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT t.name, t.source, t.frequency, t.updated_at,
                       COUNT(o.date) as obs_count,
                       MIN(o.date) as start_date,
                       MAX(o.date) as end_date
                FROM time_series t
                LEFT JOIN observations o ON t.id = o.series_id
                GROUP BY t.id
                ORDER BY t.name
            """))
            rows = result.fetchall()

        if not rows:
            return pd.DataFrame(columns=[
                "name", "source", "frequency", "updated_at",
                "obs_count", "start_date", "end_date"
            ])

        return pd.DataFrame(rows, columns=[
            "name", "source", "frequency", "updated_at",
            "obs_count", "start_date", "end_date"
        ])

    def delete_series(self, name: str) -> bool:
        """Delete a time series.

        Args:
            name: Series name

        Returns:
            True if deleted, False if not found
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT id FROM time_series WHERE name = :name"),
                {"name": name}
            ).fetchone()

            if not result:
                return False

            series_id = result[0]

            conn.execute(
                text("DELETE FROM observations WHERE series_id = :id"),
                {"id": series_id}
            )
            conn.execute(
                text("DELETE FROM time_series WHERE id = :id"),
                {"id": series_id}
            )
            conn.commit()

        return True

    def get_metadata(self, key: str) -> Optional[str]:
        """Get a metadata value.

        Args:
            key: Metadata key

        Returns:
            Value or None if not found
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT value FROM metadata WHERE key = :key"),
                {"key": key}
            ).fetchone()

        return result[0] if result else None

    def set_metadata(self, key: str, value: str) -> None:
        """Set a metadata value.

        Args:
            key: Metadata key
            value: Metadata value
        """
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT OR REPLACE INTO metadata (key, value, updated_at)
                    VALUES (:key, :value, :now)
                """),
                {"key": key, "value": value, "now": datetime.now()}
            )
            conn.commit()
