"""
Data processing utilities for NBA game prediction
Handles CSV data operations and team statistics calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class NBADataProcessor:
    """Process NBA historical data for game predictions"""

    def __init__(self, csv_path):
        """
        Initialize data processor

        Args:
            csv_path: Path to NBA historical data CSV file
        """
        self.csv_path = csv_path
        self.df = None
        self._load_data()

    def _load_data(self):
        """Load CSV data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
        except Exception as e:
            raise Exception(f"Failed to load data: {str(e)}")

    def get_available_seasons(self):
        """Get all available seasons"""
        if self.df is None:
            return []
        return sorted(self.df['season'].unique().tolist())

    def get_teams_by_season(self, season):
        """
        Get all teams in a specific season

        Args:
            season: Season string (e.g., '2024-25')

        Returns:
            List of team names
        """
        if self.df is None:
            return []

        season_data = self.df[self.df['season'] == season]
        home_teams = set(season_data['h_team_name'].unique())
        away_teams = set(season_data['o_team_name'].unique())
        all_teams = sorted(list(home_teams.union(away_teams)))

        return all_teams

    def get_game_dates(self, season, home_team, away_team):
        """
        Get all game dates between two teams in a season

        Args:
            season: Season string
            home_team: Home team name
            away_team: Away team name

        Returns:
            List of game dates
        """
        if self.df is None:
            return []

        games = self.df[
            (self.df['season'] == season) &
            (self.df['h_team_name'] == home_team) &
            (self.df['o_team_name'] == away_team)
        ]

        dates = sorted(games['date'].dt.strftime('%Y-%m-%d').tolist())
        return dates

    def get_game_data(self, season, home_team, away_team, game_date):
        """
        Get specific game data

        Args:
            season: Season string
            home_team: Home team name
            away_team: Away team name
            game_date: Game date string (YYYY-MM-DD)

        Returns:
            Dictionary containing game data or None
        """
        if self.df is None:
            return None

        game_date_dt = pd.to_datetime(game_date)

        game = self.df[
            (self.df['season'] == season) &
            (self.df['h_team_name'] == home_team) &
            (self.df['o_team_name'] == away_team) &
            (self.df['date'] == game_date_dt)
        ]

        if len(game) == 0:
            return None

        return game.iloc[0].to_dict()

    def get_team_recent_games(self, team_name, before_date, num_games=10):
        """
        Get recent games for a team before a specific date

        Args:
            team_name: Team name
            before_date: Date string or datetime object
            num_games: Number of recent games to retrieve (default 10)

        Returns:
            DataFrame containing recent games
        """
        if self.df is None:
            return pd.DataFrame()

        if isinstance(before_date, str):
            before_date = pd.to_datetime(before_date)

        # Get games where team was home or away
        team_games = self.df[
            ((self.df['h_team_name'] == team_name) |
             (self.df['o_team_name'] == team_name)) &
            (self.df['date'] < before_date)
        ].sort_values('date', ascending=False)

        return team_games.head(num_games)

    def calculate_team_avg_stats(self, team_name, before_date, num_games=10):
        """
        Calculate average statistics for a team based on recent games

        Args:
            team_name: Team name
            before_date: Date before which to calculate stats
            num_games: Number of recent games to use (default 10)

        Returns:
            Dictionary containing average statistics
        """
        recent_games = self.get_team_recent_games(team_name, before_date, num_games)

        if len(recent_games) == 0:
            return None

        # Define stat columns to calculate
        stat_cols = [
            'diff_PTS', 'diff_FGM', 'diff_FGA', 'diff_FG%',
            'diff_3PM', 'diff_3PA', 'diff_3P%',
            'diff_FTM', 'diff_FTA', 'diff_FT%',
            'diff_OREB', 'diff_DREB', 'diff_REB',
            'diff_AST', 'diff_STL', 'diff_BLK', 'diff_TOV', 'diff_PF'
        ]

        avg_stats = {}

        for col in stat_cols:
            if col not in recent_games.columns:
                continue

            # Adjust sign based on whether team was home or away
            values = []
            for _, game in recent_games.iterrows():
                if game['h_team_name'] == team_name:
                    # Team was home, use value as is
                    values.append(game[col])
                else:
                    # Team was away, negate the value
                    values.append(-game[col])

            avg_stats[col] = np.mean(values) if values else 0

        # Calculate win rate
        wins = 0
        for _, game in recent_games.iterrows():
            if game['h_team_name'] == team_name and game['home_win'] == 1:
                wins += 1
            elif game['o_team_name'] == team_name and game['home_win'] == 0:
                wins += 1

        avg_stats['win_rate'] = wins / len(recent_games) if len(recent_games) > 0 else 0
        avg_stats['games_played'] = len(recent_games)

        return avg_stats

    def prepare_prediction_data(self, season, home_team, away_team, game_date):
        """
        Prepare structured data for game prediction

        Args:
            season: Season string
            home_team: Home team name
            away_team: Away team name
            game_date: Game date string

        Returns:
            Dictionary containing game data and team statistics
        """
        # Get current game data if exists
        game_data = self.get_game_data(season, home_team, away_team, game_date)

        # Get recent performance for both teams
        home_stats = self.calculate_team_avg_stats(home_team, game_date, num_games=10)
        away_stats = self.calculate_team_avg_stats(away_team, game_date, num_games=10)

        result = {
            'season': season,
            'game_date': game_date,
            'home_team': home_team,
            'away_team': away_team,
            'game_data': game_data,
            'home_recent_stats': home_stats,
            'away_recent_stats': away_stats
        }

        return result

    def format_stats_for_display(self, stats):
        """
        Format statistics dictionary for display

        Args:
            stats: Statistics dictionary

        Returns:
            Formatted dictionary with readable keys
        """
        if stats is None:
            return {}

        display_map = {
            'diff_PTS': 'Points Differential',
            'diff_FGM': 'Field Goals Made',
            'diff_FGA': 'Field Goal Attempts',
            'diff_FG%': 'Field Goal %',
            'diff_3PM': '3-Point Made',
            'diff_3PA': '3-Point Attempts',
            'diff_3P%': '3-Point %',
            'diff_FTM': 'Free Throws Made',
            'diff_FTA': 'Free Throw Attempts',
            'diff_FT%': 'Free Throw %',
            'diff_OREB': 'Offensive Rebounds',
            'diff_DREB': 'Defensive Rebounds',
            'diff_REB': 'Total Rebounds',
            'diff_AST': 'Assists',
            'diff_STL': 'Steals',
            'diff_BLK': 'Blocks',
            'diff_TOV': 'Turnovers',
            'diff_PF': 'Personal Fouls',
            'win_rate': 'Win Rate',
            'games_played': 'Games Played'
        }

        formatted = {}
        for key, value in stats.items():
            display_key = display_map.get(key, key)
            if key == 'win_rate':
                formatted[display_key] = f"{value * 100:.1f}%"
            elif isinstance(value, float):
                formatted[display_key] = f"{value:.2f}"
            else:
                formatted[display_key] = str(value)

        return formatted

    def get_head_to_head(self, team1, team2, before_date, num_games=5):
        """
        Get head-to-head matchup history between two teams

        Args:
            team1: First team name
            team2: Second team name
            before_date: Date before which to look
            num_games: Number of recent matchups (default 5)

        Returns:
            DataFrame of head-to-head games
        """
        if self.df is None:
            return pd.DataFrame()

        if isinstance(before_date, str):
            before_date = pd.to_datetime(before_date)

        h2h_games = self.df[
            (((self.df['h_team_name'] == team1) & (self.df['o_team_name'] == team2)) |
             ((self.df['h_team_name'] == team2) & (self.df['o_team_name'] == team1))) &
            (self.df['date'] < before_date)
        ].sort_values('date', ascending=False)

        return h2h_games.head(num_games)
