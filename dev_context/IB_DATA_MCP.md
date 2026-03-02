# IB DATA MCP: Interactive Brokers Data Engine

**Role**: You are a Data Engineer / Quantitative Developer.
**Objective**: When asked to "Download data", "Fix data quality", or "Handle contract rolling" — use THIS specification. This is the production reference for Institutional-Grade Data Pipelines.

---

## 1. Problem Statement

Quantitative models are "Garbage In, Garbage Out". Retail data sources often have:
1.  **Bad Rolls**: Splices inconsistent contracts (e.g., Mar -> Jun) without adjustment, causing price jumps.
    *   *Solution*: **Back-Adjustment**. We shift historical prices so the series is continuous at the splice point.
2.  **Missing Bars**: Gaps during high volatility (precisely when you need data).
3.  **Look-Ahead Bias**: Resampling 1-minute bars to 1-hour often leaks future high/low information.

**Solution**:
1.  **Direct Fetching**: Download M5 and H1 *independently* from IB (Exchange native bars).
2.  **Smart Rolling**: Explicitly manage futures expiry (H, M, U, Z) and stitch contracts based on volume/date.
3.  **Strict Validation**: Reject datasets with significant gaps or OHLC violations.

---

## 2. Core Methodology

### 2.1. Dual Timeframe Architecture

We do **NOT** resample M5 data to create H1 data locally.
*   **Why?** IB's server-side aggregation handles tick-level anomalies and trading halts better than local pandas resampling.
*   **Policy**: Always valid to have slightly different closes on M5_last vs H1 due to different sampling boundaries? **NO**, they should match, but we trust IB's H1 for the "Official" hourly view.

### 2.2. Smart Contract Rolling (Futures)

**The Challenge**: Futures expire quarterly. We need a continuous "Back-adjusted" or "Linked" series.
**Our Approach**: "Date-Based Switch" (simplest institutional standard).

**Algorithm**:
1.  Fetch *ALL* quarterly contracts for a symbol (e.g., `ES`).
2.  Sort by Expiry: `ESH2023`, `ESM2023`, `ESU2023`...
3.  For any historical date `t`:
    *   Find contract where `Expiry > t + Buffer`.
    *   If multiple, pick the one closest to `t` (Front Month).
4.  **Auto-Roll**: When `t` reaches `Expiry - 5 days`, switch to Next Month.

**Code Pattern** (`ib_fetcher.py`):
```python
def _get_contract_for_date(self, all_contracts: list, target_date: datetime) -> Future:
    """
    Selects active contract for a specific historical date.
    
    Logic:
    - Iterate through sorted contracts.
    - Pick first contract where Expiry > target_date.
    - This automatically handles 'rolling' backward in time.
    """
    for contract in all_contracts:
        if contract.expiry > target_date:
            return contract
    return all_contracts[-1] # Fallback to newest
```

### 2.3. Back-Adjustment (The Panama Canal Method)

**Concept**:
When we switch from `OldContract` ($4000) to `NewContract` ($4020), there is a +20 gap. This is not market movement; it's just a contract specification change.
To fix this, we shift the *entire history* of `OldContract` by `+20`.

**Algorithm**:
1.  Download data **Backwards** (Newest -> Oldest).
2.  At a rollover point:
    *   `Gap = Price(Newer) - Price(Older)`
    *   `Cumulative_Adjustment += Gap`
3.  Apply: `Price(Adjusted) = Price(Raw) + Cumulative_Adjustment`

**Result**:
A continuous stream suitable for indicators (EMA, RSI) without artificial jumps.

### 2.4. Smart Backfill & Rollover (Backward-Fill)

To build a continuous dataset from "Now" backwards:
1.  **Backwards Loop**: We start at `datetime.now()` and step back by `chunk_days` (e.g., 7 days).
2.  **Dynamic Contract Selection**: At each step `t`, we ask "Which contract was active at `t`?".
3.  **Automatic Splicing**: This naturally handles rollovers.
    -   *Week 1 (Dec 10)*: Fetch `ESZ2023`
    -   *Week 2 (Dec 3)*: Fetch `ESZ2023`
    -   *Week 3 (Nov 25)*: Fetch `ESA2023` (Example rollover point)

**Code Pattern** (`_backfill_loop`):
```python
while current_date > stop_date:
    # 1. Get Active Contract for this week
    contract = self._get_contract_for_date(all_contracts, current_date)
    
    # 2. Fetch Chunk
    df_chunk = self.fetch_chunk(contract, current_date)
    
    # 3. Store & Step Back
    collected_data.append(df_chunk)
    current_date -= timedelta(days=7)
```

### 2.3. Pacing & Rate Limits

IB API imposes strict limits (approx 60 historical requests / 10 mins).
**Violation** = Temporary Ban (Pacing Violation).

**Mechanism**:
*   **Token Bucket**: Track requests made.
*   **Sleep**: If `requests > limit`, sleep until bucket refills.

```python
# settings.py
ib_requests_per_period: int = 60
ib_period_seconds: int = 600

# ib_fetcher.py
def _wait_for_pacing(self):
    elapsed = time.time() - self._last_req
    required_delay = self.settings.ib_period_seconds / self.settings.ib_requests_per_period
    
    if elapsed < required_delay:
        time.sleep(required_delay - elapsed)
```

---

## 3. Data Lake Integration

**Storage Architecture**:
Back-adjusted data is stored in **Apache Parquet**.
For full specifications on Partitioning, Schema, and Polars usage, see **[`DATA_LAKE_MCP.md`](dev_context/DATA_LAKE_MCP.md)**.

**Producer Workflow**:
1.  `IBFetcher` downloads chunks.
2.  Checks `DATA_LAKE_MCP` schema requirements (Float32, etc.).
3.  Writes to the "Cold Lake" or "Hot Stream" as defined in the Data Lake spec.

---

## 4. Validation Protocol (from `data_validator.py`)

Before any backtest, data must pass the **Quality Gate**.

### 4.1. The Checks
1.  **Gaps**: No missing periods > `GAP_THRESHOLD` (30m for M5, 3h for H1).
2.  **OHLC Integrity**:
    *   `High >= Low`
    *   `High >= max(Open, Close)`
    *   `Low <= min(Open, Close)`
3.  **Volume Anomalies**: Z-Score > 5.0 (warn only).

### 4.2. Reporting
We generate a `ValidationResult` object.
*   **Pass**: `quality_score >= 0.95`.
*   **Fail**: `quality_score < 0.95`.

**Code Example**:
```python
penalty = (missing_bars + ohlc_violations * 10) / total_bars
quality_score = max(0.0, 1.0 - penalty)
```

---

## 5. Configuration (from `settings.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ib_host` | 127.0.0.1 | TWS/Gateway IP |
| `ib_port` | 7497 | 7497 (Paper) / 7496 (Live) |
| `max_historical_years` | 2 | How far back to download |
| `delayed_data_minutes` | 15 | Offset for non-subscribed data |
| `timeframes` | ["5m", "1h"] | STRICT list of timeframes to fetch |

---

## 6. Implementation Reference

### 6.1. Fetching Data
```python
# Usage in run.py
from multi_strategy_backtest.data import IBFetcher

fetcher = IBFetcher()
fetcher.connect()

# Fetch M5 (Smart Rolling included)
df_m5 = fetcher.fetch_m5("ES", force_restart=False)

# Fetch H1 (Independent stream)
df_h1 = fetcher.fetch_h1("ES")
```

### 6.2. Validating Data
```python
from multi_strategy_backtest.data import DataLake, DataValidator

lake = DataLake()
validator = DataValidator()

df = lake.load_m5("ES")
result = validator.validate(df, symbol="ES", timeframe="5m")

if not result.is_valid:
    print(f"CRITICAL DATA FAILURE: {result.issues}")
```

---

## 7. Production Checklist

- [ ] **TWS Running**: Ensure IB Gateway/TWS is open and API ports are enabled.
- [ ] **Permissions**: "Enable ActiveX and Socket Clients" must be checked in TWS.
- [ ] **Market Data**: For Live trading, you need real-time subscriptions. For Backtest, delayed data (15m) is fine.
- [ ] **Disk Space**: 2 years of M5 data for 10 symbols ~ 500MB.
- [ ] **No Resampling**: Verify logic does NOT contain `m5.resample('1h')`.

---

## 8. Known Limitations

| Issue | Mitigation |
|-------|------------|
| **Pacing Violation** | If 3 concurrent scripts run, you WILL get banned. Use 1 fetcher instance. |
| **Holiday Gaps** | IB returns no bars for holidays. Validator must ignore weekends/holidays (TODO). |
| **Contract Liquidity** | "Back-month" contracts (e.g., ESU in March) have low volume. Logic prioritizes Front Month. |
| **Timezone** | IB returns Exchange Native Time. We convert everything to **UTC-like** (or raw exchange time depending on `settings`). Standard is naive datetime representing Exchange Local. |

---

## 9. Credits

*   **ib_insync**: The Python library backing the connection.
*   **Project Implementation**: `src/multi_strategy_backtest/data/ib_fetcher.py`
