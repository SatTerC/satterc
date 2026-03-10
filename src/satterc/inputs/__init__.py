from .load import input_datasets
from .grid import coordinate_grid
from .daily import daily_inputs
from .weekly import weekly_inputs, aggregate_daily_to_weekly
from .monthly import (
    monthly_inputs,
    aggregate_daily_to_monthly,
    aggregate_weekly_to_monthly,
)
from .static import static_inputs

__all__ = [
    "input_datasets",
    "coordinate_grid",
    "daily_inputs",
    "weekly_inputs",
    "aggregate_daily_to_weekly",
    "monthly_inputs",
    "aggregate_daily_to_monthly",
    "aggregate_weekly_to_monthly",
    "static_inputs",
]
