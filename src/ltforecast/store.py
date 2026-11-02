"""Parquet storage: raw vendor dumps and year-partitioned core tables, queryable from DuckDB."""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import polars as pl

CORE_TABLES = ("securities", "prices_daily", "fundamentals_pit", "macro_pit", "marketcap_daily", "universe")

# Tables partitioned by year, and the date column that decides the partition.
PARTITION_COL = {
    "prices_daily": "date",
    "fundamentals_pit": "available_date",
    "marketcap_daily": "date",
    "universe": "date",
}


class Store:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir)

    def raw_path(self, vendor: str, table: str) -> Path:
        return self.root / "raw" / vendor / f"{table}.parquet"

    def core_path(self, table: str) -> Path:
        if table not in CORE_TABLES:
            raise KeyError(f"unknown core table {table!r}")
        return self.root / "core" / table

    def write_raw(self, vendor: str, table: str, df: pl.DataFrame) -> Path:
        path = self.raw_path(vendor, table)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return path

    def scan_raw(self, vendor: str, table: str) -> pl.LazyFrame:
        return pl.scan_parquet(self.raw_path(vendor, table))

    def write(self, table: str, df: pl.DataFrame) -> Path:
        """Replace a core table. Partitioned tables are split into year=YYYY directories."""
        path = self.core_path(table)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        col = PARTITION_COL.get(table)
        if col is None:
            df.write_parquet(path / "data.parquet")
            return path
        df = df.with_columns(pl.col(col).dt.year().alias("year"))
        for (year,), part in df.partition_by("year", as_dict=True, maintain_order=True).items():
            d = path / f"year={year}"
            d.mkdir()
            part.drop("year").write_parquet(d / "data.parquet")
        return path

    def scan(self, table: str) -> pl.LazyFrame:
        path = self.core_path(table)
        if not path.exists():
            raise FileNotFoundError(f"core table {table!r} not built yet; run `ltforecast build`")
        return pl.scan_parquet(path / "**" / "*.parquet", hive_partitioning=False)

    def duckdb(self) -> duckdb.DuckDBPyConnection:
        """In-memory DuckDB connection with a view per built core table."""
        con = duckdb.connect()
        for table in CORE_TABLES:
            path = self.root / "core" / table
            if path.exists():
                con.execute(
                    f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path.as_posix()}/**/*.parquet')"
                )
        return con
