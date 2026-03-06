"""
src/backtest_engine/analytics/dashboard/components.py

Data loading and Streamlit component rendering helpers.

Responsibility: File I/O (load Parquet / JSON / text from results/) and
reusable Streamlit widgets (e.g. the exit breakdown table).
No chart building happens here — that lives in charts.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


def get_results_dir() -> Path:
    """
    Resolves the results/ directory relative to the project root.

    Returns:
        Absolute path to results/.
    """
    # dashboard/components.py → dashboard/ → analytics/ → backtest_engine/ → src/ → project root
    return Path(__file__).parent.parent.parent.parent.parent / "results"


def load_parquet(filename: str) -> Optional[pd.DataFrame]:
    """
    Loads a Parquet file from the results directory.

    Args:
        filename: File name relative to results/ (e.g. 'history.parquet').

    Returns:
        DataFrame or None if the file does not exist.
    """
    path = get_results_dir() / filename
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_text(filename: str) -> Optional[str]:
    """
    Loads a plain-text file from the results directory.

    Args:
        filename: File name relative to results/ (e.g. 'report.txt').

    Returns:
        String contents or None if the file does not exist.
    """
    path = get_results_dir() / filename
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def load_json(filename: str) -> Optional[dict]:
    """
    Loads a JSON file from the results directory.

    Args:
        filename: File name relative to results/ (e.g. 'metrics.json').

    Returns:
        Parsed dict or None if the file does not exist.
    """
    path = get_results_dir() / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_exit_table(trades: Optional[pd.DataFrame]) -> None:
    """
    Renders a colour-coded exit reason breakdown table using st.dataframe.

    Methodology:
        Groups trades by exit_reason, calculates count and percentage share.
        A TOTAL row is appended so relative proportions are immediately clear.

    Args:
        trades: Trades DataFrame with 'exit_reason' column.
    """
    if trades is None or trades.empty or "exit_reason" not in trades.columns:
        st.caption("No trade data available.")
        return

    total  = len(trades)
    counts = trades["exit_reason"].value_counts()
    rows   = [
        {"Exit Reason": r, "Count": int(c), "%": f"{c / total:.1%}"}
        for r, c in counts.items()
    ]
    rows.append({"Exit Reason": "TOTAL", "Count": total, "%": "100.0%"})

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        height=240,
    )
