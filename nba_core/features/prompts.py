"""Prompt construction for every gateway-backed workload.

Prompts live here, apart from transport and apart from web handlers, so that the
wording of an instruction set can be versioned independently of the code that
ships it over the wire.
"""

from __future__ import annotations

import json

__all__ = ["PromptLibrary", "GAME_SCORING_SCHEMA", "SUM_100_RULE"]


SUM_100_RULE = "Percentages must be numeric and must add up to 100."

GAME_SCORING_SCHEMA = {
    "winner": "home" or "away",
    "home_win_prob": "<number 0-100>",
    "away_win_prob": "<number 0-100>",
}


_DIFF_LABELS = (
    ("diff_MIN", "Minutes"),
    ("diff_FGM", "Field Goals Made"),
    ("diff_FGA", "Field Goal Attempts"),
    ("diff_FG%", "Field Goal Percentage"),
    ("diff_3PM", "Three-Pointers Made"),
    ("diff_3PA", "Three-Pointer Attempts"),
    ("diff_3P%", "Three-Point Percentage"),
    ("diff_FTM", "Free Throws Made"),
    ("diff_FTA", "Free Throw Attempts"),
    ("diff_FT%", "Free Throw Percentage"),
    ("diff_OREB", "Offensive Rebounds"),
    ("diff_DREB", "Defensive Rebounds"),
    ("diff_REB", "Total Rebounds"),
    ("diff_AST", "Assists"),
    ("diff_STL", "Steals"),
    ("diff_BLK", "Blocks"),
    ("diff_TOV", "Turnovers"),
    ("diff_PF", "Personal Fouls"),
)

_STATUS_LABELS = {
    "home_injury": "Injury Report",
    "away_injury": "Injury Report",
    "home_news": "Team News",
    "away_news": "Team News",
    "home_tactical": "Tactical Profile",
    "away_tactical": "Tactical Profile",
}


def _coerce(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class PromptLibrary:
    """Stateless builders; every method returns a ready-to-send prompt string."""

    # -- shared fragments ---------------------------------------------------

    @staticmethod
    def differential_block(game: dict, heading: str = "Differential Statistics (Home - Away)") -> str:
        lines = [f"{heading}:"]
        for column, label in _DIFF_LABELS:
            lines.append(f"- {label}: {game.get(column, 0)}")
        return "\n".join(lines)

    @staticmethod
    def stats_block(team: str, stats: dict) -> str:
        if not stats:
            return f"### {team}\n- No aggregated form data available.\n"
        lines = [f"### {team}"]
        for key, value in stats.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def json_contract(schema: dict) -> str:
        body = json.dumps(schema, indent=4, ensure_ascii=False)
        return f"Return the assessment as JSON only, with no surrounding commentary:\n{body}"

    # -- module 1: single-game scoring -------------------------------------

    def game_scoring(self, game: dict) -> str:
        home_team = game.get("h_team_name", "Home Team")
        away_team = game.get("o_team_name", "Away Team")

        return (
            "You are an NBA result-modelling specialist. Given the pre-game "
            "differential profile of one matchup, produce a two-way win "
            "probability split.\n\n"
            f"Home Team: {home_team}\n"
            f"Away Team: {away_team}\n\n"
            f"{self.differential_block(game)}\n\n"
            "Weigh possessions, shooting efficiency, rebounding margin and "
            "turnover differential, then account for home-court advantage. "
            f"{SUM_100_RULE}\n\n"
            f"{self.json_contract(GAME_SCORING_SCHEMA)}"
        )

    # -- module 2: evidence retrieval --------------------------------------

    def evidence_query(self, category: str, team: str, opponent: str, season: str, game_date: str) -> str:
        status = _STATUS_LABELS.get(category, "Team Status")
        focus = {
            "injury": (
                "injury reports and player availability, including key absences, "
                "expected return dates and the effect on the rotation"
            ),
            "news": (
                "recent news and team dynamics, including roster moves, "
                "winning or losing streaks and squad morale"
            ),
            "tactical": (
                "tactical profile and playing style, including offensive and "
                "defensive schemes, key contributors and coaching approach"
            ),
        }
        key = category.split("_")[-1]
        return (
            f"Collect the latest {focus.get(key, 'team status')} for the {team} "
            f"around {game_date} in the {season} NBA season. "
            f"The opponent in this matchup is {opponent}. "
            f"Organise the findings under a single '{status}' heading."
        )

    # -- module 2: fusion summary ------------------------------------------

    def fusion_summary(self, season: str, home_team: str, away_team: str, game_date: str,
                       structured_text: str, raw_unstructured: str) -> str:
        return (
            "You are an NBA prediction expert. Based on the following "
            "information, provide a comprehensive game prediction.\n\n"
            f"Matchup: {away_team} at {home_team}, {game_date} ({season} season)\n\n"
            "## Structured Statistical Data:\n"
            f"{structured_text}\n"
            "## Raw Unstructured Information (from web search):\n"
            f"{raw_unstructured}\n\n"
            "Please:\n"
            "1. Summarize the unstructured information into three concise sections:\n"
            "   - Injury Reports (key injuries affecting both teams)\n"
            "   - Pre-game News (recent team dynamics)\n"
            "   - Tactical Analysis (matchup insights)\n\n"
            "2. Predict the game outcome with win probabilities\n\n"
            "3. List Top 5 factors influencing the result (ranked by importance)\n\n"
            "Respond in JSON format:\n"
            "{\n"
            '    "prediction": {\n'
            '        "winner": "home" or "away",\n'
            '        "home_win_probability": number (0-100),\n'
            '        "away_win_probability": number (0-100)\n'
            "    },\n"
            '    "top_factors": [\n'
            "        {\n"
            '            "rank": 1,\n'
            '            "factor": "factor name",\n'
            '            "type": "structured" or "unstructured",\n'
            '            "description": "brief impact description"\n'
            "        }\n"
            "    ],\n"
            '    "unstructured_summary": {\n'
            '        "injury_reports": "concise injury summary",\n'
            '        "pregame_news": "concise news summary",\n'
            '        "tactical_analysis": "concise tactical summary"\n'
            "    }\n"
            "}"
        )

    # -- module 3: roster news digest --------------------------------------

    def roster_digest(self, team: str, players: list[str]) -> str:
        roster = ", ".join(players[:5]) if players else "the current rotation"
        return (
            f"Search for the latest news and updates about {team} NBA team and "
            f"their key players: {roster}.\n"
            "Focus on:\n"
            "1. Recent performance and statistics\n"
            "2. Injury reports and player availability\n"
            "3. Team form and momentum\n"
            "4. Key player highlights\n"
            "5. Any recent trades or roster changes\n\n"
            "Provide a comprehensive summary in JSON format:\n"
            "{\n"
            f'    "team": "{team}",\n'
            '    "overall_status": "brief team status",\n'
            '    "key_points": ["point 1", "point 2", "point 3"],\n'
            '    "player_updates": {"player_name": "update"},\n'
            '    "recent_form": "description of recent games"\n'
            "}"
        )

    # -- module 3: matchup verdict -----------------------------------------

    def matchup_verdict(self, home_team: str, away_team: str,
                        home_news: str, away_news: str) -> str:
        return (
            "You are an expert NBA analyst. Based on the following information "
            "about two teams, predict the match outcome.\n\n"
            f"## {home_team} (Home Team)\n"
            "Recent News and Player Updates:\n"
            f"{home_news}\n\n"
            f"## {away_team} (Away Team)\n"
            "Recent News and Player Updates:\n"
            f"{away_news}\n\n"
            "Please analyze:\n"
            "1. Overall team strength comparison\n"
            "2. Key player availability and form\n"
            "3. Recent performance trends\n"
            "4. Home court advantage\n"
            "5. Head-to-head implications\n\n"
            "Provide your prediction in JSON format:\n"
            "{\n"
            '    "winner": "home" or "away",\n'
            '    "home_probability": number (0-100),\n'
            '    "away_probability": number (0-100),\n'
            '    "analysis": "Detailed analysis explaining your prediction, key '
            'factors, and reasoning (3-5 sentences)",\n'
            '    "confidence": "high", "medium", or "low"\n'
            "}"
        )


LIBRARY = PromptLibrary()


def build(kind: str, **kwargs) -> str:
    """Convenience dispatcher used by service objects."""
    builder = getattr(LIBRARY, kind)
    return builder(**kwargs)
