"""Downloadable payload templates.

Both published templates share one column order; only the row semantics differ
(a cross-sectional matchup row versus a two-game form window).
"""

from __future__ import annotations

import pandas as pd

from constants import TEMPLATE_COLUMNS

__all__ = ["TEMPLATE_ROWS", "build_template_frame", "template_filename"]


TEMPLATE_ROWS = {
    "single": [
        {
            "season": "2024-25",
            "season_type": "Regular Season",
            "date": "2024-11-15",
            "h_team_name": "Los Angeles Lakers",
            "o_team_name": "Boston Celtics",
            "diff_MIN": 240.0,
            "diff_FGM": 2.0,
            "diff_FGA": 8.0,
            "diff_FG%": 0.05,
            "diff_3PM": 1.0,
            "diff_3PA": 3.0,
            "diff_3P%": 0.08,
            "diff_FTM": 3.0,
            "diff_FTA": 5.0,
            "diff_FT%": 0.10,
            "diff_OREB": 2.0,
            "diff_DREB": 3.0,
            "diff_REB": 5.0,
            "diff_AST": 3.0,
            "diff_STL": 1.0,
            "diff_BLK": 2.0,
            "diff_TOV": -2.0,
            "diff_PF": 1.0,
        },
        {
            "season": "2024-25",
            "season_type": "Regular Season",
            "date": "2024-11-16",
            "h_team_name": "Golden State Warriors",
            "o_team_name": "Denver Nuggets",
            "diff_MIN": 240.0,
            "diff_FGM": -1.0,
            "diff_FGA": -5.0,
            "diff_FG%": -0.03,
            "diff_3PM": -2.0,
            "diff_3PA": -4.0,
            "diff_3P%": -0.05,
            "diff_FTM": -2.0,
            "diff_FTA": -3.0,
            "diff_FT%": -0.05,
            "diff_OREB": -1.0,
            "diff_DREB": -2.0,
            "diff_REB": -3.0,
            "diff_AST": -2.0,
            "diff_STL": 0.0,
            "diff_BLK": -1.0,
            "diff_TOV": 1.0,
            "diff_PF": -1.0,
        },
    ],
    "time_series": [
        {
            "season": "2024-25",
            "season_type": "Regular Season",
            "date": "2024-11-10",
            "h_team_name": "Los Angeles Lakers",
            "o_team_name": "Boston Celtics",
            "diff_MIN": 240.0,
            "diff_FGM": 2.0,
            "diff_FGA": 8.0,
            "diff_FG%": 0.05,
            "diff_3PM": 1.0,
            "diff_3PA": 3.0,
            "diff_3P%": 0.08,
            "diff_FTM": 3.0,
            "diff_FTA": 5.0,
            "diff_FT%": 0.10,
            "diff_OREB": 2.0,
            "diff_DREB": 3.0,
            "diff_REB": 5.0,
            "diff_AST": 3.0,
            "diff_STL": 1.0,
            "diff_BLK": 2.0,
            "diff_TOV": -2.0,
            "diff_PF": 1.0,
        },
        {
            "season": "2024-25",
            "season_type": "Regular Season",
            "date": "2024-11-12",
            "h_team_name": "Los Angeles Lakers",
            "o_team_name": "Golden State Warriors",
            "diff_MIN": 240.0,
            "diff_FGM": -1.0,
            "diff_FGA": -5.0,
            "diff_FG%": -0.03,
            "diff_3PM": -2.0,
            "diff_3PA": -4.0,
            "diff_3P%": -0.05,
            "diff_FTM": -2.0,
            "diff_FTA": -3.0,
            "diff_FT%": -0.05,
            "diff_OREB": -1.0,
            "diff_DREB": -2.0,
            "diff_REB": -3.0,
            "diff_AST": -2.0,
            "diff_STL": 0.0,
            "diff_BLK": -1.0,
            "diff_TOV": 1.0,
            "diff_PF": -1.0,
        },
    ],
}


def template_filename(kind: str) -> str:
    """Published filename for a template kind."""
    if kind == "single":
        return "template_single_game.csv"
    return "template_time_series.csv"


def build_template_frame(kind: str) -> pd.DataFrame:
    """Materialise a template as a DataFrame in canonical column order."""
    if kind not in TEMPLATE_ROWS:
        raise KeyError(f"unknown template kind: {kind}")
    frame = pd.DataFrame(TEMPLATE_ROWS[kind])
    ordered = [column for column in TEMPLATE_COLUMNS if column in frame.columns]
    return frame[ordered]
