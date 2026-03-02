# DATA LAKE MCP: Scalable Caching Layer

**Role**: You are a Data Engineer / Quantitative Developer.
**Objective**: When asked to "Load data", "Cache results", or "Manage datasets" — use THIS specification. This is the reference for High-Performance storage.

---

## 1. Problem Statement

Monolithic file caching (single CSV/Parquet file per symbol) has specific scaling limits:
1.  **Write Amplification**: Appending small data to large files requires rewriting the entire file.
2.  **Memory Constraints**: Loading large datasets (e.g., 50GB tick data) into Pandas can cause OOM errors.
3.  **Concurrency Locking**: Long write operations block readers.
4.  **Storage Inefficiency**: `float64` uses double the memory bandwidth of `float32` which is often sufficient for price data.

**Solution**:
1.  **Partitioned Storage**: Store data in monthly chunks (`year=2024/month=01`).
2.  **Lazy Evaluation**: Use **Polars** (`scan_parquet`) to query datasets without loading them entirely into RAM.
3.  **Hot/Cold Separation**: Immutable history (Cold) vs. Append-only Live Log (Hot).
4.  **Optimized Schema**: Strict `float32` / `Int64` typing.

---

## 2. Core Methodology

### 2.1. Architecture: Hot/Cold Split

To minimize lock contention and write amplification:

*   **Cold Lake (History)**: Immutable Parquet files, partition by Month. Updated once/day (EOD).
*   **Hot Stream (Live)**: Lightweight append-only log (CSV/Redis) for *today's* ticks.
*   **Unified View**: The Reader `loads` = `Cold Lake` + `Hot Stream`.

**Topology**:
```
[IB Fetcher] --(Realtime)--> [Hot Stream: today.csv]
      |
      +-----(EOD Job)------> [Cold Lake: /data/cache/{SYMBOL}/...]
```

### 2.2. Storage Format: Partitioned Parquet

We use **Hive-style Partitioning** with Snappy/Zstd compression.

**Path Structure**:
`data/cache/{SYMBOL}/year={YYYY}/month={MM}/data.parquet`

**Advantages**:
*   **Zero-Copy Appends**: Adding February data means writing a *new* file `month=02`, not touching January.
*   **Predicate Pushdown**: `lake.scan(symbol).filter(pl.col("date") > "2024-01-01")` only reads relevant folders.

### 2.3. Schema Definition (Optimized)

| Column | Type | Notes |
|---|---|---|
| `date` | `Datetime (ns, UTC)` | STRICT UTC. |
| `open` | `Float32` | 7 decimal digits precision. |
| `high` | `Float32` | |
| `low` | `Float32` | |
| `close` | `Float32` | |
| `volume` | `Int64` | Discrete count. |
| `bar_count`| `Int32` | |

---

## 3. Implementation (Polars Standard)

**Why Polars?**
*   **Lazy Execution**: `scan_parquet` builds a query plan. Data flows only on `collect()`.
*   **Streaming**: Capable of processing datasets larger than RAM.

### 3.1. Loading Data

```python
import polars as pl
from multi_strategy_backtest.data import DataLake

lake = DataLake()

# 1. Lazy Scan (Instant return)
lazy_frame = lake.scan_m5("ES")

# 2. Filter & Materialize
# Only reads relevant partitions
df = (
    lazy_frame
    .filter(pl.col("date") >= "2024-01-01")
    .select(["date", "close"])
    .collect()  # <--- IO happens here
    .to_pandas() # Bridge to legacy code if needed
)
```

### 3.2. Live Trading Warm-up (Hot/Cold Merge)

```python
def load_live_context(symbol: str, lookback: int = 1000):
    # 1. Cold History (Lazy)
    cold = pl.scan_parquet(f"data/cache/{symbol}/**/*.parquet")
    
    # 2. Hot Stream (Lazy + Explicit Schema + UTC)
    hot = pl.scan_csv(
        f"data/hot/{symbol}_current.csv",
        dtypes={"close": pl.Float32, "volume": pl.Int64} # Partial example
    ).with_columns(
        pl.col("date").str.to_datetime().dt.replace_time_zone("UTC")
    )
    
    # 3. Merge & Tail
    return (
        pl.concat([cold, hot], how="vertical")
        .sort("date")
        .tail(lookback)
        .collect()
    )
```

---

## 4. Configuration (from `settings.py`)

| Parameter | Default | Description |
|---|---|---|
| `backend` | `polars` | `polars` or `pandas`. |
| `partitioning` | `monthly` | `monthly`, `yearly`, or `none`. |
| `float_precision` | `32` | `32` or `64`. |
| `hot_path` | `data/hot` | Directory for intra-day logs. |

---

## 5. Universal Usage Patterns

### 5.1. Backtesting Engine
*   **Need**: Scan multi-year history for metrics/optimization.
*   **Pattern**:
    ```python
    vol = (
        lake.scan("ES")
        .select(pl.col("close").pct_change().std())
        .collect()
    )
    ```
*   **Benefit**: Faster execution via SIMD and parallel processing.

### 5.2. VaR & Risk Engine
*   **Need**: Rolling window (e.g., 252 days) for regime detection.
*   **Pattern**:
    ```python
    regime_input = (
        lake.scan("ES")
        .tail(252 * 500) # Only load relevant history
        .collect()
    )
    ```

---

## 6. Comparison: Monolithic vs. Partitioned

| Feature | Monolithic (Pandas/CSV) | Partitioned (Polars/Parquet) |
|---|---|---|
| **Storage** | Single File (`ES.csv`) | Partitioned (`ES/2024/01.parquet`) |
| **Update Cost** | $O(N)$ (Rewrite all) | $O(1)$ (Write new partition) |
| **Reading** | Eager (Load all to RAM) | Lazy (Scan & Filter) |
| **Types** | `float64` / Object | `float32` / `Int64` / Categorical |
| **Concurrency** | Read/Write Locks | Writer (Append) + Reader (Snapshot) |
| **Integrity** | Overwrite History | Append-Only (Immutable History) |

---

## 7. Implementation Gotchas (Critical)

Even with good architecture, details matter. Watch out for these three traps:

### 7.1. Schema Mismatch (Parquet vs CSV)
**Risk**: Parquet (Cold) is strictly `Float32`. CSV (Hot) is untyped. `pl.concat` will crash if Polars infers `Float64` or `String` (due to trash data) from CSV.
**Fix**: Explicitly define schema when reading the Hot Log.
```python
# CORRECT way to read Hot Log
hot = pl.scan_csv(
    f"data/hot/{symbol}_current.csv",
    dtypes={
        "open": pl.Float32, "high": pl.Float32, 
        "low": pl.Float32, "close": pl.Float32,
        "volume": pl.Int64
    }
)
```

### 7.2. The "EOD Race Condition"
**Scenario**: At 23:59:59, the Bot writes a tick. At 00:00:00, the Cron Job moves data to Cold storage.
**Risk**: If they access the file simultaneously, you lose data or crash.
**Fix**: **File Rotation**.
1.  Bot always writes to `current.csv`.
2.  Cron Job does `mv current.csv processed_{timestamp}.csv`.
3.  Bot detects missing file and creates a new `current.csv`.
4.  Reader logic: `History` + `processed_*.csv` + `current.csv`.

### 7.3. Timezone Alignment (UTC vs Naive)
**Risk**: Parquet preserves Timezone (`UTC`). CSV loses it (`Naive`). `pl.concat` fails on mismatch.
**Fix**: Force UTC on the Hot stream immediately.
```python
hot = (
    pl.scan_csv(...)
    .with_columns(
        pl.col("date").str.to_datetime().dt.replace_time_zone("UTC")
    )
)
```

---

## 8. Production Checklist

- [ ] **Polars Installed**: Ensure `pip install polars pyarrow`.
- [ ] **Partitioning Job**: Cron script to move "Hot" data to "Cold" partitions at 00:00 UTC.
- [ ] **Schema Validation**: Reject `float64` or string columns in Parquet.
- [ ] **Backup**: Snapshot `data/cache` (Cold) is sufficient.

---

## 9. Credits & References

*   **Polars**: Fast DataFrame library.
*   **Apache Iceberg**: Partitioning logic reference.
