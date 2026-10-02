"""Read-only repository over the historical box-score feature matrix."""

from __future__ import annotations

import os
import threading

import pandas as pd

__all__ = ["BoxScoreRepository", "DatasetError"]


class DatasetError(RuntimeError):
    """Raised when the source file cannot be read into memory."""


class BoxScoreRepository:
    """Loads the feature matrix once and serves filtered slices."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._frame: pd.DataFrame | None = None
        self._lock = threading.RLock()
        self.load()

    # -- lifecycle ----------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """(Re)read the source file, normalising the date column."""
        if not os.path.exists(self.csv_path):
            raise DatasetError(f"dataset not found: {self.csv_path}")

        try:
            frame = pd.read_csv(self.csv_path)
            frame["date"] = pd.to_datetime(frame["date"])
        except Exception as exc:  # noqa: BLE001 - surfaced as a dataset error
            raise DatasetError(f"Failed to load data: {exc}") from exc

        with self._lock:
            self._frame = frame
        return frame

    @property
    def frame(self) -> pd.DataFrame:
        with self._lock:
            if self._frame is None:
                return self.load()
            return self._frame

    # -- slices -------------------------------------------------------------

    def seasons(self) -> list[str]:
        return sorted(self.frame["season"].unique().tolist())

    def season_slice(self, season: str) -> pd.DataFrame:
        return self.frame[self.frame["season"] == season]

    def teams(self, season: str) -> list[str]:
        window = self.season_slice(season)
        home = set(window["h_team_name"].unique())
        away = set(window["o_team_name"].unique())
        return sorted(home.union(away))

    def matchups(self, season: str, home_team: str, away_team: str) -> pd.DataFrame:
        return self.frame[
            (self.frame["season"] == season)
            & (self.frame["h_team_name"] == home_team)
            & (self.frame["o_team_name"] == away_team)
        ]

    def game(self, season: str, home_team: str, away_team: str,
             game_date) -> dict | None:
        window = self.matchups(season, home_team, away_team)
        window = window[window["date"] == pd.to_datetime(game_date)]
        if len(window) == 0:
            return None
        return window.iloc[0].to_dict()

    def team_games(self, team_name: str, before_date, limit: int = 10) -> pd.DataFrame:
        cutoff = pd.to_datetime(before_date)
        window = self.frame[
            ((self.frame["h_team_name"] == team_name)
             | (self.frame["o_team_name"] == team_name))
            & (self.frame["date"] < cutoff)
        ].sort_values("date", ascending=False)
        return window.head(limit)

    def head_to_head(self, team1: str, team2: str, before_date, limit: int = 5) -> pd.DataFrame:
        cutoff = pd.to_datetime(before_date)
        window = self.frame[
            (
                ((self.frame["h_team_name"] == team1) & (self.frame["o_team_name"] == team2))
                | ((self.frame["h_team_name"] == team2) & (self.frame["o_team_name"] == team1))
            )
            & (self.frame["date"] < cutoff)
        ].sort_values("date", ascending=False)
        return window.head(limit)

    # -- shape --------------------------------------------------------------

    def column_names(self) -> list[str]:
        return list(self.frame.columns)

    def size(self) -> int:
        return int(len(self.frame))
