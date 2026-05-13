"""Forecast replay for FedZero.

This module is FedZero's local extension of the Vessim ecosystem. Vessim 0.15
provides `Trace` for actual time-series replay but intentionally has no
forecast support; this file adds an archived-forecast lookup that is used by
FedZero's scheduling LP.

The two types are kept strictly separate:

- `vessim.Trace`: actual data, queried as `at(elapsed)`. One-arg API.
- `Forecast` (this file): archived predictions, queried as
  `window(request_time, start, end, freq, column)`. Two-time API
  (`request_time` selects which forecast batch was used; `start..end` is
  the prediction horizon).

The Vessim docs reference this file as a real-world example of extending
Vessim with custom forecast support.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import vessim as vs

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class Forecast:
    """Archived-forecast table replay.

    Two schemas are accepted:

    - Archived: `DataFrame` with a `MultiIndex` of `(request_time, forecast_time)`.
      `window()` selects the most recent batch issued at or before `request_time`.
    - Static: `DataFrame` indexed by `forecast_time` only. `request_time` is
      ignored; the same predictions are returned regardless of when the query
      is made. Useful for "reserved capacity"-style forecasts.

    Columns become zone/client names matching the corresponding `vs.Trace`.
    """

    def __init__(self, data: pd.DataFrame):
        if isinstance(data.index, pd.MultiIndex):
            req = data.index.get_level_values(0).to_numpy(dtype="datetime64[ns]")
            fc = data.index.get_level_values(1).to_numpy(dtype="datetime64[ns]")
            order = np.lexsort((fc, req))
            self._req_times: Optional[np.ndarray] = req[order]
        else:
            fc = pd.DatetimeIndex(data.index).to_numpy(dtype="datetime64[ns]")
            order = np.argsort(fc)
            self._req_times = None
        self._fc_times = fc[order]
        self._values: dict[str, np.ndarray] = {
            str(c): data[c].to_numpy(dtype=float)[order] for c in data.columns
        }

    @classmethod
    def from_csv(cls, path: str | Path, scale: float = 1.0) -> "Forecast":
        """Load an archived (two-timestamp) forecast CSV.

        Expected schema: first two columns are `request_time, forecast_time`;
        remaining columns are zone names. Matches the Solcast 2022 forecast
        schema bundled in `fedzero/data/`.
        """
        df = pd.read_csv(path, index_col=[0, 1], parse_dates=[0, 1])
        return cls(df * scale if scale != 1.0 else df)

    def columns(self) -> list[str]:
        return list(self._values.keys())

    def window(
        self,
        request_time: datetime,
        *,
        start: datetime,
        end: datetime,
        freq: timedelta | str,
        column: str,
        fill: Literal["ffill", "bfill"] = "bfill",
    ) -> pd.Series:
        """Forecast values at `freq` over `(start, end]`.

        For archived forecasts the most recent batch with
        `request_time <= request_time` is selected before resampling.
        For static forecasts `request_time` is ignored.
        """
        if column not in self._values:
            raise ValueError(f"No forecast available for column {column!r}.")
        np_start, np_end = np.datetime64(pd.Timestamp(start)), np.datetime64(pd.Timestamp(end))
        np_freq = np.timedelta64(pd.Timedelta(freq))

        fc_times, values = self._fc_times, self._values[column]
        if self._req_times is not None:
            np_req = np.datetime64(pd.Timestamp(request_time))
            end_idx = np.searchsorted(self._req_times, np_req, side="right")
            if end_idx <= 0:
                raise ValueError(f"No forecasts available at request_time={request_time}.")
            latest = self._req_times[end_idx - 1]
            start_idx = np.searchsorted(self._req_times, latest, side="left")
            fc_times, values = fc_times[start_idx:end_idx], values[start_idx:end_idx]

        new_times = np.arange(np_start + np_freq, np_end + np.timedelta64(1, "ns"), np_freq, dtype="datetime64[ns]")
        idx = np.searchsorted(fc_times, new_times, side="left" if fill == "bfill" else "right")
        if fill == "ffill":
            idx -= 1
        out = np.full(idx.shape, np.nan)
        valid = (idx >= 0) & (idx < len(values))
        out[valid] = values[idx[valid]]
        return pd.Series(out, index=pd.DatetimeIndex(new_times))


def load_solcast(
    scenario: str,
    scale: float = 1.0,
    use_forecast: bool = True,
) -> tuple[dict[str, vs.Trace], Optional[Forecast]]:
    """Load a bundled Solcast 2022 dataset from `fedzero/data/`.

    Returns one `vs.Trace` per zone plus an optional `Forecast` for the
    scheduling LP.

    Args:
        scenario: Either `"global"` or `"germany"`.
        scale: Multiplier applied to all values (e.g. peak panel size in W).
        use_forecast: If False, the forecast is omitted; callers should
            fall back to actuals as a perfect oracle.
    """
    df = pd.read_csv(_DATA_DIR / f"solcast2022_{scenario}_actual.csv", index_col=0, parse_dates=True) * scale
    actuals = {zone: vs.Trace(df[zone], anchor=df.index[0]) for zone in df.columns}
    forecast = Forecast.from_csv(_DATA_DIR / f"solcast2022_{scenario}_forecast.csv", scale=scale) if use_forecast else None
    return actuals, forecast
