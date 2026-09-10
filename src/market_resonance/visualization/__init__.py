"""Publication-quality figures and tables."""

from .treasury_eda import (
    DEFAULT_FIGURE_DIR,
    add_curve_slope,
    daily_yield_changes,
    generate_treasury_eda_figures,
    load_treasury_dataset,
)

__all__ = [
    "DEFAULT_FIGURE_DIR",
    "add_curve_slope",
    "daily_yield_changes",
    "generate_treasury_eda_figures",
    "load_treasury_dataset",
]
