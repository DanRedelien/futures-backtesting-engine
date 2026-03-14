from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """
    Resolves the repository root from the dashboard package location.

    Methodology:
        Path discovery lives in a framework-neutral module so FastAPI,
        Streamlit, tests, and async workers can all share the same artifact root
        logic without importing UI code.

    Returns:
        Absolute path to the repository root.
    """
    return Path(__file__).resolve().parents[5]


def get_results_dir() -> Path:
    """
    Resolves the shared results directory under the project root.

    Returns:
        Absolute path to `results/`.
    """
    return get_project_root() / "results"


def get_scenarios_root(create: bool = True) -> Path:
    """
    Resolves the scenario artifact namespace under `results/scenarios/`.

    Args:
        create: Whether to create the directory when it does not exist.

    Returns:
        Absolute path to the scenario root.
    """
    root = get_results_dir() / "scenarios"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root
