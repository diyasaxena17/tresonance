"""Sequential 10Y Treasury direction environment.

The environment reframes next-day 10Y Treasury yield movement as a small
chronological decision problem. At each date ``t``, the agent observes only
features that are known at ``t``: the current 10Y level, trailing 10Y yield
changes, and the contemporaneous 2Y-10Y curve spread. It then predicts whether
the next trading row's 10Y yield will FALL, stay FLAT, or RISE.

Leakage prevention is intentionally simple: state rows are built from current
and trailing values only, while the realized direction is computed separately
from the change between row ``t`` and row ``t + 1``. The future row never enters
the state returned for time ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from market_resonance.features import CHANGE_LAGS, create_treasury_features

ACTION_FALL = 0
ACTION_FLAT = 1
ACTION_RISE = 2
VALID_ACTIONS = (ACTION_FALL, ACTION_FLAT, ACTION_RISE)
DEFAULT_DIRECTION_THRESHOLD_BP = 1.0
BASIS_POINTS_PER_PERCENTAGE_POINT = 100.0


def realized_direction(
    next_day_change: float,
    threshold_bp: float = DEFAULT_DIRECTION_THRESHOLD_BP,
) -> int:
    """Convert a next-day 10Y yield change into FALL, FLAT, or RISE.

    The project stores yields in percentage points, so one basis point is
    represented as ``0.01``. A change whose absolute value is less than or equal
    to the configured threshold is labeled FLAT.
    """
    if threshold_bp < 0:
        raise ValueError("threshold_bp must be non-negative.")

    threshold = threshold_bp / BASIS_POINTS_PER_PERCENTAGE_POINT
    if next_day_change < -threshold:
        return ACTION_FALL
    if next_day_change > threshold:
        return ACTION_RISE
    return ACTION_FLAT


@dataclass(frozen=True)
class TreasuryDirectionEnv:
    """Deterministic sequential environment for next-day 10Y direction.

    Args:
        frame: Daily Treasury yield table with ``date``, ``10Y``, and ``2Y``.
            Rows may be unsorted; the environment sorts them chronologically.
        threshold_bp: Basis-point threshold used to map next-day 10Y yield
            changes into FALL, FLAT, and RISE labels.

    The action space is deliberately tiny: ``0`` predicts FALL, ``1`` predicts
    FLAT, and ``2`` predicts RISE. Rewards are ``+1`` for a matching direction
    and ``-1`` otherwise. No Gym dependency is required.
    """

    frame: pd.DataFrame
    threshold_bp: float = DEFAULT_DIRECTION_THRESHOLD_BP

    def __post_init__(self) -> None:
        if self.threshold_bp < 0:
            raise ValueError("threshold_bp must be non-negative.")

        required_columns = ["date", "2Y", "10Y"]
        missing_columns = [
            column for column in required_columns if column not in self.frame.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        sorted_frame = self.frame.loc[:, required_columns].copy()
        sorted_frame["date"] = pd.to_datetime(sorted_frame["date"], errors="coerce")
        if sorted_frame["date"].isna().any():
            raise ValueError("The date column contains invalid dates.")
        if sorted_frame["date"].duplicated().any():
            raise ValueError("The dataset contains duplicate dates.")

        for column in ["2Y", "10Y"]:
            sorted_frame[column] = pd.to_numeric(sorted_frame[column], errors="coerce")
        if sorted_frame[["2Y", "10Y"]].isna().any().any():
            raise ValueError("Treasury yield columns must be complete and numeric.")

        sorted_frame = sorted_frame.sort_values("date").reset_index(drop=True)
        features = create_treasury_features(
            sorted_frame,
            maturities=("10Y",),
            change_lags=CHANGE_LAGS,
        )
        features["2Y_10Y_spread"] = sorted_frame["10Y"] - sorted_frame["2Y"]

        feature_columns = [
            "10Y_level",
            "10Y_chg_1d",
            "10Y_chg_5d",
            "10Y_chg_21d",
            "2Y_10Y_spread",
        ]
        max_lag = max(CHANGE_LAGS, default=0)
        first_index = max_lag
        last_decision_index = len(sorted_frame) - 2
        if first_index > last_decision_index:
            raise ValueError("Not enough observations to create one RL transition.")

        state_values = features.loc[:, feature_columns].to_numpy(dtype=float)
        if np.isnan(state_values[first_index : last_decision_index + 1]).any():
            raise ValueError("State features contain missing values.")

        object.__setattr__(self, "_frame", sorted_frame)
        object.__setattr__(self, "feature_columns", feature_columns)
        object.__setattr__(self, "_states", state_values)
        object.__setattr__(self, "_first_index", first_index)
        object.__setattr__(self, "_last_decision_index", last_decision_index)
        object.__setattr__(self, "_current_index", first_index)

    def reset(self) -> np.ndarray:
        """Return the first chronological state and restart the episode."""
        object.__setattr__(self, "_current_index", self._first_index)
        return self._state_at(self._current_index)

    def step(self, action: int) -> tuple[np.ndarray | None, int, bool, dict[str, Any]]:
        """Advance one chronological transition.

        The reward compares ``action`` with the direction realized between the
        current row and the next row. The returned next state is row ``t + 1``
        only when another decision can be made from it; the terminal transition
        returns ``None`` as the state.
        """
        if action not in VALID_ACTIONS:
            raise ValueError("action must be 0 (FALL), 1 (FLAT), or 2 (RISE).")
        if self._current_index > self._last_decision_index:
            raise RuntimeError("Cannot call step() after the episode is done.")

        current_index = self._current_index
        next_index = current_index + 1
        next_day_change = (
            self._frame.loc[next_index, "10Y"] - self._frame.loc[current_index, "10Y"]
        )
        direction = realized_direction(next_day_change, self.threshold_bp)
        reward = 1 if action == direction else -1

        done = current_index >= self._last_decision_index
        object.__setattr__(self, "_current_index", next_index)
        next_state = None if done else self._state_at(next_index)
        info = {
            "date": self._frame.loc[current_index, "date"],
            "next_date": self._frame.loc[next_index, "date"],
            "next_day_10y_change": float(next_day_change),
            "realized_direction": direction,
            "threshold_bp": self.threshold_bp,
        }
        return next_state, reward, done, info

    @property
    def current_date(self) -> pd.Timestamp:
        """Date of the current decision row."""
        return self._frame.loc[self._current_index, "date"]

    @property
    def n_transitions(self) -> int:
        """Number of available chronological decisions in the episode."""
        return self._last_decision_index - self._first_index + 1

    def _state_at(self, index: int) -> np.ndarray:
        return self._states[index].copy()
