"""
Data processing utilities for NBA game prediction.

Layering: :class:`~nba_core.data.BoxScoreRepository` owns file access and
slicing; :class:`NBADataProcessor` adds the rolling-form aggregation and the
display formatting used by the fusion workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from constants import MODULE2_STAT_COLUMNS
from nba_core.data import BoxScoreRepository

__all__ = ["NBADataProcessor", "DISPLAY_MAP"]

#: Human-readable labels for the differential columns.
DISPLAY_MAP = {
    "diff_PTS": "Points Differential",
    "diff_FGM": "Field Goals Made",
    "diff_FGA": "Field Goal Attempts",
    "diff_FG%": "Field Goal %",
    "diff_3PM": "3-Point Made",
    "diff_3PA": "3-Point Attempts",
    "diff_3P%": "3-Point %",
    "diff_FTM": "Free Throws Made",
    "diff_FTA": "Free Throw Attempts",
    "diff_FT%": "Free Throw %",
    "diff_OREB": "Offensive Rebounds",
    "diff_DREB": "Defensive Rebounds",
    "diff_REB": "Total Rebounds",
    "diff_AST": "Assists",
    "diff_STL": "Steals",
    "diff_BLK": "Blocks",
    "diff_TOV": "Turnovers",
    "diff_PF": "Personal Fouls",
    "win_rate": "Win Rate",
    "games_played": "Games Played",
}


class FormAggregator:
    """Rolling-form statistics for a single team."""

    def __init__(self, columns=MODULE2_STAT_COLUMNS):
        self.columns = tuple(columns)

    def summarise(self, recent_games: pd.DataFrame, team_name: str,
                  window: int) -> dict:
        averages: dict[str, float] = {}

        for column in self.columns:
            if column not in recent_games.columns:
                continue
            averages[column] = float(np.mean(self._oriented(recent_games, team_name, column)))

        averages["win_rate"] = self._win_rate(recent_games, team_name, window)
        averages["games_played"] = int(len(recent_games))
        return averages

    @staticmethod
    def _oriented(recent_games: pd.DataFrame, team_name: str, column: str) -> list[float]:
        """Sign-adjust a differential column so it is always pro-team."""
        values: list[float] = []
        for _, game in recent_games.iterrows():
            if game["h_team_name"] == team_name:
                values.append(game[column])
            else:
                values.append(-game[column])
        return values

    @staticmethod
    def _win_rate(recent_games: pd.DataFrame, team_name: str, window: int) -> float:
        wins = 0
        for _, game in recent_games.iterrows():
            if game["h_team_name"] == team_name and game["home_win"] == 1:
                wins += 1
            elif game["o_team_name"] == team_name and game["home_win"] == 0:
                wins += 1
        return wins / window if window else 0.0


class NBADataProcessor:
    """Facade over the historical dataset used by the fusion workflow."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.repository = BoxScoreRepository(csv_path)
        self.aggregator = FormAggregator()
        self.df = self.repository.frame

    # -- reference data -----------------------------------------------------

    def get_available_seasons(self) -> list[str]:
        return self.repository.seasons()

    def get_teams_by_season(self, season: str) -> list[str]:
        return self.repository.teams(season)

    def get_game_dates(self, season: str, home_team: str, away_team: str) -> list[str]:
        window = self.repository.matchups(season, home_team, away_team)
        return sorted(window["date"].dt.strftime("%Y-%m-%d").tolist())

    def get_game_data(self, season: str, home_team: str, away_team: str,
                      game_date: str) -> dict | None:
        return self.repository.game(season, home_team, away_team, game_date)

    # -- rolling form -------------------------------------------------------

    def get_team_recent_games(self, team_name: str, before_date,
                              num_games: int = 10) -> pd.DataFrame:
        return self.repository.team_games(team_name, before_date, num_games)

    def calculate_team_avg_stats(self, team_name: str, before_date,
                                 num_games: int = 10) -> dict | None:
        recent = self.repository.team_games(team_name, before_date, num_games)
        if len(recent) == 0:
            return None
        return self.aggregator.summarise(recent, team_name, len(recent))

    def get_head_to_head(self, team1: str, team2: str, before_date,
                         num_games: int = 5) -> pd.DataFrame:
        return self.repository.head_to_head(team1, team2, before_date, num_games)

    # -- assembly -----------------------------------------------------------

    def prepare_prediction_data(self, season: str, home_team: str, away_team: str,
                                game_date: str) -> dict:
        return {
            "season": season,
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "game_data": self.get_game_data(season, home_team, away_team, game_date),
            "home_recent_stats": self.calculate_team_avg_stats(home_team, game_date, 10),
            "away_recent_stats": self.calculate_team_avg_stats(away_team, game_date, 10),
        }

    # -- presentation -------------------------------------------------------

    @staticmethod
    def format_stats_for_display(stats: dict | None) -> dict:
        """Render a statistics dictionary with readable labels."""
        if stats is None:
            return {}

        formatted: dict[str, str] = {}
        for key, value in stats.items():
            label = DISPLAY_MAP.get(key, key)
            if key == "win_rate":
                formatted[label] = f"{value * 100:.1f}%"
            elif isinstance(value, float):
                formatted[label] = f"{value:.2f}"
            else:
                formatted[label] = str(value)
        return formatted
