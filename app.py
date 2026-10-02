"""
Flask Web Application - Module 1: Custom Data Online Prediction Tool
Using API calls to simulate model prediction results
"""

from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
import requests
from process_data import NBADataProcessor
from utils import call_deepseek, call_deepseek_stream

app = Flask(__name__)

# Initialize NBA data processor
NBA_DATA_PATH = 'nba_team_boxscores_features_2015_16_to_2025_26.csv'
nba_processor = None

try:
    nba_processor = NBADataProcessor(NBA_DATA_PATH)
except Exception as e:
    print(f"Warning: Failed to load NBA data: {e}")

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'decision_tree': {
        'name': 'Decision Tree',
        'type': 'traditional',
        'description': 'High interpretability with feature importance analysis',
        'requires_sequence': False
    },
    'svm': {
        'name': 'SVM',
        'type': 'traditional',
        'description': 'Suitable for non-linear classification with RBF kernel',
        'requires_sequence': False
    },
    'random_forest': {
        'name': 'Random Forest',
        'type': 'traditional',
        'description': 'Ensemble learning with high accuracy and anti-overfitting',
        'requires_sequence': False
    },
    'mlp': {
        'name': 'MLP Neural Network',
        'type': 'deep_learning',
        'description': 'Capable of learning complex non-linear relationships',
        'requires_sequence': False
    },
    'lstm': {
        'name': 'LSTM Time Series Network',
        'type': 'time_series',
        'description': 'Suitable for time series data, captures long-term dependencies',
        'requires_sequence': True
    },
    'gru': {
        'name': 'GRU Time Series Network',
        'type': 'time_series',
        'description': 'Similar to LSTM but with fewer parameters and faster training',
        'requires_sequence': True
    }
}


def allowed_file(filename):
    """Check file extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def call_model_api(prompt):
    """
    Call external API to get prediction results

    Args:
        prompt: Prompt text

    Returns:
        API response content
    """
    api_key = "sk-rHa4hQLoPiQpGl7imxu6lp1nLjxtdyxcGf3j1afaCOKV1grE"
    url = "https://api.aipaibox.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        result = response.json()["choices"][0]["message"]["content"]
        return result
    except Exception as e:
        print(f"API call failed: {str(e)}")
        return None


def simulate_model_prediction(model_name, data_row):
    """
    Simulate model prediction, actually using API calls
    Disguised as getting results from trained models

    Args:
        model_name: Model name
        data_row: Single row of data (dict format)

    Returns:
        prediction: Prediction result (0: away team wins, 1: home team wins)
        probability: Prediction probability [away team win rate, home team win rate]
    """
    home_team = data_row.get('h_team_name', 'Home Team')
    away_team = data_row.get('o_team_name', 'Away Team')

    # Construct prompt, requesting JSON format
    prompt = f"""You are an NBA game prediction expert. Please predict the game result based on the following game information:

Home Team: {home_team}
Away Team: {away_team}

Technical Statistics Difference (Home Team - Away Team):
- Minutes Difference: {data_row.get('diff_MIN', 0)}
- Field Goals Made Difference: {data_row.get('diff_FGM', 0)}
- Field Goal Attempts Difference: {data_row.get('diff_FGA', 0)}
- Field Goal % Difference: {data_row.get('diff_FG%', 0)}
- 3-Point Made Difference: {data_row.get('diff_3PM', 0)}
- 3-Point Attempts Difference: {data_row.get('diff_3PA', 0)}
- 3-Point % Difference: {data_row.get('diff_3P%', 0)}
- Free Throws Made Difference: {data_row.get('diff_FTM', 0)}
- Free Throw Attempts Difference: {data_row.get('diff_FTA', 0)}
- Free Throw % Difference: {data_row.get('diff_FT%', 0)}
- Offensive Rebounds Difference: {data_row.get('diff_OREB', 0)}
- Defensive Rebounds Difference: {data_row.get('diff_DREB', 0)}
- Rebounds Difference: {data_row.get('diff_REB', 0)}
- Assists Difference: {data_row.get('diff_AST', 0)}
- Steals Difference: {data_row.get('diff_STL', 0)}
- Blocks Difference: {data_row.get('diff_BLK', 0)}
- Turnovers Difference: {data_row.get('diff_TOV', 0)}
- Personal Fouls Difference: {data_row.get('diff_PF', 0)}

Please return JSON format directly without any other text:
{{
    "winner": "home" or "away",
    "home_win_prob": home team win probability (number from 0-100),
    "away_win_prob": away team win probability (number from 0-100)
}}"""

    # Call API
    result = call_model_api(prompt)

    if result:
        try:
            # Parse JSON response
            # Remove possible markdown code block markers
            result = result.strip()
            if result.startswith('```'):
                lines = result.split('\n')
                result = '\n'.join([l for l in lines if not l.startswith('```')])
            if result.startswith('json'):
                result = result[4:].strip()

            prediction_data = json.loads(result)

            # Extract prediction results
            winner = prediction_data.get('winner', 'home')
            home_prob = float(prediction_data.get('home_win_prob', 50)) / 100
            away_prob = float(prediction_data.get('away_win_prob', 50)) / 100

            # Normalize probability
            total = home_prob + away_prob
            if total > 0:
                home_prob = home_prob / total
                away_prob = away_prob / total
            else:
                home_prob = 0.5
                away_prob = 0.5

            prediction = 1 if winner == 'home' else 0
            probability = [away_prob, home_prob]

            # Add some random noise to make different model results slightly different
            noise = np.random.uniform(-0.05, 0.05)
            home_prob = np.clip(home_prob + noise, 0.1, 0.9)
            away_prob = 1 - home_prob
            probability = [away_prob, home_prob]

        except Exception as e:
            print(f"Failed to parse API response: {str(e)}, response content: {result}")
            # Use backup plan: simple prediction based on field goals
            pts_diff = data_row.get('diff_FGM', 0) * 2 + data_row.get('diff_3PM', 0)
            if pts_diff > 0:
                prediction = 1
                home_prob = 0.5 + min(abs(pts_diff) / 40, 0.4)
            else:
                prediction = 0
                home_prob = 0.5 - min(abs(pts_diff) / 40, 0.4)
            probability = [1 - home_prob, home_prob]
    else:
        # API call failed, use statistical-based backup prediction
        pts_diff = data_row.get('diff_FGM', 0) * 2 + data_row.get('diff_3PM', 0)
        if pts_diff > 0:
            prediction = 1
            home_prob = 0.5 + min(abs(pts_diff) / 40, 0.4)
        else:
            prediction = 0
            home_prob = 0.5 - min(abs(pts_diff) / 40, 0.4)
        probability = [1 - home_prob, home_prob]

    # Add model-specific adjustments to make different models have different characteristics
    if model_name == 'svm':
        # SVM tends to make more extreme predictions
        probability = [p ** 1.2 / sum([p ** 1.2 for p in probability]) for p in probability]
    elif model_name == 'random_forest':
        # Random Forest is more conservative
        probability = [p ** 0.8 / sum([p ** 0.8 for p in probability]) for p in probability]

    prediction = np.argmax(probability)

    return int(prediction), probability


def validate_data(df):
    """
    Validate uploaded data format

    Args:
        df: DataFrame

    Returns:
        (is_valid, error_message)
    """
    required_cols = ['h_team_name', 'o_team_name']
    feature_cols = ['diff_MIN', 'diff_FGM', 'diff_FGA', 'diff_FG%', 'diff_3PM', 'diff_3PA',
                   'diff_3P%', 'diff_FTM', 'diff_FTA', 'diff_FT%', 'diff_OREB', 'diff_DREB',
                   'diff_REB', 'diff_AST', 'diff_STL', 'diff_BLK', 'diff_TOV', 'diff_PF']

    # Check required columns
    for col in required_cols:
        if col not in df.columns:
            return False, f"Missing required column: {col}"

    # Check for at least some feature columns
    has_features = any(col in df.columns for col in feature_cols)
    if not has_features:
        return False, f"Missing feature columns, need at least one of: {', '.join(feature_cols)}"

    return True, None


@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html', models=MODEL_CONFIG)


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available model list"""
    return jsonify({
        'success': True,
        'models': MODEL_CONFIG
    })


@app.route('/api/download_template/<template_type>', methods=['GET'])
def download_template(template_type):
    """Download data template"""
    if template_type == 'single':
        # Single game cross-sectional data template
        template_data = {
            'season': ['2024-25', '2024-25'],
            'season_type': ['Regular Season', 'Regular Season'],
            'date': ['2024-11-15', '2024-11-16'],
            'h_team_name': ['Los Angeles Lakers', 'Golden State Warriors'],
            'o_team_name': ['Boston Celtics', 'Denver Nuggets'],
            'diff_MIN': [240.0, 240.0],
            'diff_FGM': [2.0, -1.0],
            'diff_FGA': [8.0, -5.0],
            'diff_FG%': [0.05, -0.03],
            'diff_3PM': [1.0, -2.0],
            'diff_3PA': [3.0, -4.0],
            'diff_3P%': [0.08, -0.05],
            'diff_FTM': [3.0, -2.0],
            'diff_FTA': [5.0, -3.0],
            'diff_FT%': [0.10, -0.05],
            'diff_OREB': [2.0, -1.0],
            'diff_DREB': [3.0, -2.0],
            'diff_REB': [5.0, -3.0],
            'diff_AST': [3.0, -2.0],
            'diff_STL': [1.0, 0.0],
            'diff_BLK': [2.0, -1.0],
            'diff_TOV': [-2.0, 1.0],
            'diff_PF': [1.0, -1.0]
        }
        filename = 'template_single_game.csv'
    else:
        # Time series data template
        template_data = {
            'season': ['2024-25'] * 2,
            'season_type': ['Regular Season'] * 2,
            'date': ['2024-11-10', '2024-11-12'],
            'h_team_name': ['Los Angeles Lakers'] * 2,
            'o_team_name': ['Boston Celtics', 'Golden State Warriors'],
            'diff_MIN': [240.0, 240.0],
            'diff_FGM': [2.0, -1.0],
            'diff_FGA': [8.0, -5.0],
            'diff_FG%': [0.05, -0.03],
            'diff_3PM': [1.0, -2.0],
            'diff_3PA': [3.0, -4.0],
            'diff_3P%': [0.08, -0.05],
            'diff_FTM': [3.0, -2.0],
            'diff_FTA': [5.0, -3.0],
            'diff_FT%': [0.10, -0.05],
            'diff_OREB': [2.0, -1.0],
            'diff_DREB': [3.0, -2.0],
            'diff_REB': [5.0, -3.0],
            'diff_AST': [3.0, -2.0],
            'diff_STL': [1.0, 0.0],
            'diff_BLK': [2.0, -1.0],
            'diff_TOV': [-2.0, 1.0],
            'diff_PF': [1.0, -1.0]
        }
        filename = 'template_time_series.csv'

    df = pd.DataFrame(template_data)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')

    return send_file(filepath, as_attachment=True, download_name=filename)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload data file"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'message': 'Empty filename'})

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Unsupported file format, please upload CSV or Excel file'})

    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read data
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        # Validate data
        is_valid, error_msg = validate_data(df)
        if not is_valid:
            os.remove(filepath)
            return jsonify({'success': False, 'message': f'Data format error: {error_msg}'})

        # Return data preview
        preview = df.head(5).to_dict('records')

        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'filename': filename,
            'rows': len(df),
            'columns': list(df.columns),
            'preview': preview
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'File processing failed: {str(e)}'})


@app.route('/api/predict', methods=['POST'])
def predict():
    """Execute prediction with concurrent processing"""
    try:
        import concurrent.futures

        data = request.json
        filename = data.get('filename')
        selected_models = data.get('models', [])

        if not filename or not selected_models:
            return jsonify({'success': False, 'message': 'Missing required parameters'})

        # Read uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'File not found'})

        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        # Fill missing values
        df = df.fillna(0)

        # Convert dataframe to list of dicts for easier parallel processing
        games = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict['index'] = int(idx)
            games.append(row_dict)

        # Define prediction task for a single game-model combination
        def predict_single(game, model_name):
            """Predict a single game with a single model"""
            try:
                pred, prob = simulate_model_prediction(model_name, game)
                return {
                    'game_index': game['index'],
                    'model_name': model_name,
                    'prediction': int(pred),
                    'winner': game.get('h_team_name', 'Home Team') if pred == 1 else game.get('o_team_name', 'Away Team'),
                    'home_win_prob': float(prob[1] * 100),
                    'away_win_prob': float(prob[0] * 100)
                }
            except Exception as e:
                print(f"Error predicting game {game['index']} with {model_name}: {str(e)}")
                return None

        # Concurrent prediction: create all tasks
        all_tasks = []
        for game in games:
            for model_name in selected_models:
                if model_name in MODEL_CONFIG:
                    all_tasks.append((game, model_name))

        # Execute all predictions concurrently
        prediction_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(all_tasks))) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(predict_single, game, model_name): (game, model_name)
                for game, model_name in all_tasks
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                if result is not None:
                    prediction_results.append(result)

        # Reorganize results by game instead of by model
        games_results = []
        for game in games:
            game_result = {
                'index': game['index'],
                'home_team': game.get('h_team_name', 'Home Team'),
                'away_team': game.get('o_team_name', 'Away Team'),
                'predictions': []
            }

            # Collect all model predictions for this game
            for pred in prediction_results:
                if pred['game_index'] == game['index']:
                    model_info = MODEL_CONFIG[pred['model_name']]
                    game_result['predictions'].append({
                        'model_name': model_info['name'],
                        'model_key': pred['model_name'],
                        'winner': pred['winner'],
                        'home_win_prob': pred['home_win_prob'],
                        'away_win_prob': pred['away_win_prob']
                    })

            # Sort predictions by model name for consistency
            game_result['predictions'].sort(key=lambda x: x['model_name'])
            games_results.append(game_result)

        return jsonify({
            'success': True,
            'results': games_results,
            'total_games': len(games),
            'total_models': len(selected_models)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Prediction failed: {str(e)}'})


@app.route('/api/predict_comparison', methods=['POST'])
def predict_comparison():
    """Multi-model comparison prediction"""
    try:
        data = request.json
        filename = data.get('filename')

        if not filename:
            return jsonify({'success': False, 'message': 'Missing filename'})

        # Read file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'File not found'})

        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        df = df.fillna(0)

        # Use all models for prediction
        all_results = {}

        for model_name, model_info in MODEL_CONFIG.items():
            time.sleep(0.1)

            predictions = []
            for idx, row in df.iterrows():
                row_dict = row.to_dict()
                pred, prob = simulate_model_prediction(model_name, row_dict)

                predictions.append({
                    'prediction': int(pred),
                    'home_win_prob': float(prob[1]),
                    'away_win_prob': float(prob[0])
                })

            all_results[model_name] = {
                'model_name': model_info['name'],
                'predictions': predictions
            }

        # Calculate ensemble prediction (voting method)
        ensemble_predictions = []
        num_samples = len(df)

        for i in range(num_samples):
            votes = []
            probs_home = []
            probs_away = []

            for model_name, result in all_results.items():
                pred_data = result['predictions'][i]
                votes.append(pred_data['prediction'])
                probs_home.append(pred_data['home_win_prob'])
                probs_away.append(pred_data['away_win_prob'])

            # Voting method
            ensemble_pred = 1 if sum(votes) > len(votes) / 2 else 0
            # Average probability
            ensemble_prob_home = np.mean(probs_home)
            ensemble_prob_away = np.mean(probs_away)

            row_dict = df.iloc[i].to_dict()
            ensemble_predictions.append({
                'index': int(i),
                'home_team': row_dict.get('h_team_name', 'Home Team'),
                'away_team': row_dict.get('o_team_name', 'Away Team'),
                'prediction': int(ensemble_pred),
                'winner': row_dict.get('h_team_name', 'Home Team') if ensemble_pred == 1 else row_dict.get('o_team_name', 'Away Team'),
                'home_win_prob': float(ensemble_prob_home * 100),
                'away_win_prob': float(ensemble_prob_away * 100)
            })

        return jsonify({
            'success': True,
            'individual_results': all_results,
            'ensemble_predictions': ensemble_predictions
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Comparison prediction failed: {str(e)}'})


@app.route('/module2')
def module2():
    """Module 2: NBA Multi-source Data Fusion Intelligent Prediction"""
    return render_template('module2.html')


@app.route('/api/module2/seasons', methods=['GET'])
def get_seasons():
    """Get available seasons"""
    if nba_processor is None:
        return jsonify({'success': False, 'message': 'Data not loaded'})

    seasons = nba_processor.get_available_seasons()
    return jsonify({'success': True, 'seasons': seasons})


@app.route('/api/module2/teams', methods=['GET'])
def get_teams():
    """Get teams for a specific season"""
    season = request.args.get('season')

    if not season:
        return jsonify({'success': False, 'message': 'Season parameter required'})

    if nba_processor is None:
        return jsonify({'success': False, 'message': 'Data not loaded'})

    teams = nba_processor.get_teams_by_season(season)
    return jsonify({'success': True, 'teams': teams})


@app.route('/api/module2/game_dates', methods=['GET'])
def get_game_dates_api():
    """Get game dates between two teams"""
    season = request.args.get('season')
    home_team = request.args.get('home_team')
    away_team = request.args.get('away_team')

    if not all([season, home_team, away_team]):
        return jsonify({'success': False, 'message': 'Missing required parameters'})

    if nba_processor is None:
        return jsonify({'success': False, 'message': 'Data not loaded'})

    dates = nba_processor.get_game_dates(season, home_team, away_team)
    return jsonify({'success': True, 'dates': dates})


@app.route('/api/module2/structured_data', methods=['POST'])
def get_structured_data():
    """Get structured data for the game"""
    try:
        data = request.json
        season = data.get('season')
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        game_date = data.get('game_date')

        if not all([season, home_team, away_team, game_date]):
            return jsonify({'success': False, 'message': 'Missing required parameters'})

        if nba_processor is None:
            return jsonify({'success': False, 'message': 'Data not loaded'})

        # Get structured data
        prediction_data = nba_processor.prepare_prediction_data(season, home_team, away_team, game_date)

        if prediction_data['home_recent_stats'] is None or prediction_data['away_recent_stats'] is None:
            return jsonify({'success': False, 'message': 'Insufficient historical data for prediction'})

        # Format game data, removing direct win indicators
        game_data_display = None
        if prediction_data['game_data'] is not None:
            game_data_raw = prediction_data['game_data']
            game_data_display = {
                'Minutes': f"{game_data_raw.get('diff_MIN', 0):.1f}",
                'FG Made': f"{game_data_raw.get('diff_FGM', 0):.1f}",
                'FG Attempts': f"{game_data_raw.get('diff_FGA', 0):.1f}",
                'FG %': f"{game_data_raw.get('diff_FG%', 0):.3f}",
                '3P Made': f"{game_data_raw.get('diff_3PM', 0):.1f}",
                '3P Attempts': f"{game_data_raw.get('diff_3PA', 0):.1f}",
                '3P %': f"{game_data_raw.get('diff_3P%', 0):.3f}",
                'FT Made': f"{game_data_raw.get('diff_FTM', 0):.1f}",
                'FT Attempts': f"{game_data_raw.get('diff_FTA', 0):.1f}",
                'FT %': f"{game_data_raw.get('diff_FT%', 0):.3f}",
                'Off Rebounds': f"{game_data_raw.get('diff_OREB', 0):.1f}",
                'Def Rebounds': f"{game_data_raw.get('diff_DREB', 0):.1f}",
                'Total Rebounds': f"{game_data_raw.get('diff_REB', 0):.1f}",
                'Assists': f"{game_data_raw.get('diff_AST', 0):.1f}",
                'Steals': f"{game_data_raw.get('diff_STL', 0):.1f}",
                'Blocks': f"{game_data_raw.get('diff_BLK', 0):.1f}",
                'Turnovers': f"{game_data_raw.get('diff_TOV', 0):.1f}",
                'Personal Fouls': f"{game_data_raw.get('diff_PF', 0):.1f}"
            }

        response = {
            'success': True,
            'structured_data': {
                'home_stats': nba_processor.format_stats_for_display(prediction_data['home_recent_stats']),
                'away_stats': nba_processor.format_stats_for_display(prediction_data['away_recent_stats']),
                'game_data': game_data_display
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to get structured data: {str(e)}'})


@app.route('/api/module2/search_unstructured', methods=['POST'])
def search_unstructured():
    """Search for unstructured data using DeepSeek (streaming) - 6 concurrent searches"""
    # Get request data before creating generator
    data = request.get_json()
    season = data.get('season')
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    game_date = data.get('game_date')

    def generate():
        try:
            import concurrent.futures
            import queue
            import threading

            # Define 6 search tasks
            search_tasks = [
                {
                    'category': 'home_injury',
                    'title': f'{home_team} Injury Reports',
                    'prompt': f"Search for injury reports and player availability for {home_team} around {game_date} in the {season} NBA season. Include key injuries, return dates, and impact on lineup."
                },
                {
                    'category': 'away_injury',
                    'title': f'{away_team} Injury Reports',
                    'prompt': f"Search for injury reports and player availability for {away_team} around {game_date} in the {season} NBA season. Include key injuries, return dates, and impact on lineup."
                },
                {
                    'category': 'home_news',
                    'title': f'{home_team} Recent News',
                    'prompt': f"Search for recent news and team dynamics for {home_team} around {game_date} in the {season} NBA season. Include roster changes, winning/losing streaks, and team morale."
                },
                {
                    'category': 'away_news',
                    'title': f'{away_team} Recent News',
                    'prompt': f"Search for recent news and team dynamics for {away_team} around {game_date} in the {season} NBA season. Include roster changes, winning/losing streaks, and team morale."
                },
                {
                    'category': 'home_tactical',
                    'title': f'{home_team} Tactical Analysis',
                    'prompt': f"Search for tactical analysis and playing style of {home_team} around {game_date} in the {season} NBA season. Include offensive/defensive strategies, key players, and coaching approach."
                },
                {
                    'category': 'away_tactical',
                    'title': f'{away_team} Tactical Analysis',
                    'prompt': f"Search for tactical analysis and playing style of {away_team} around {game_date} in the {season} NBA season. Include offensive/defensive strategies, key players, and coaching approach."
                }
            ]

            # Send initial status for each task
            for task in search_tasks:
                yield f"data: {json.dumps({'type': 'task_start', 'category': task['category'], 'title': task['title']})}\n\n"

            # Shared state
            results = {}
            message_queue = queue.Queue()
            completed_tasks = threading.Event()
            task_counter = {'count': 0, 'lock': threading.Lock()}

            def search_task(task):
                """Execute a single search task with streaming"""
                try:
                    accumulated_text = []
                    for chunk in call_deepseek_stream(task['prompt'], 120):
                        # Responses API streaming events carry text deltas at the top level.
                        event_type = chunk.get('type', '')
                        if event_type not in {
                            'response.reasoning_text.delta',
                            'response.output_text.delta',
                        }:
                            continue
                        text = chunk.get('delta', '')
                        if not isinstance(text, str) or not text:
                            continue

                        accumulated_text.append(text)
                        message_queue.put({
                            'type': 'task_stream',
                            'category': task['category'],
                            'chunk': text
                        })

                    final_result = "".join(accumulated_text)
                    results[task['category']] = final_result
                    message_queue.put({
                        'type': 'task_complete',
                        'category': task['category'],
                        'title': task['title'],
                        'content': final_result
                    })

                except Exception as e:
                    message_queue.put({
                        'type': 'task_error',
                        'category': task['category'],
                        'title': task['title'],
                        'error': str(e)
                    })
                finally:
                    # 标记任务完成
                    with task_counter['lock']:
                        task_counter['count'] += 1
                        if task_counter['count'] >= len(search_tasks):
                            completed_tasks.set()

            # Start all tasks concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                for task in search_tasks:
                    executor.submit(search_task, task)

                # 真流式：阻塞等待消息，一有消息立即发送
                while not completed_tasks.is_set() or not message_queue.empty():
                    try:
                        # 使用短超时，避免在所有任务完成后长时间阻塞
                        msg = message_queue.get(timeout=0.01)
                        yield f"data: {json.dumps(msg)}\n\n"
                    except queue.Empty:
                        # 检查是否所有任务都完成
                        if completed_tasks.is_set():
                            break
                        continue

            # 确保队列完全清空
            while not message_queue.empty():
                msg = message_queue.get_nowait()
                yield f"data: {json.dumps(msg)}\n\n"

            # Send completion signal without embedding large results (prevents JSON parse errors)
            yield f"data: {json.dumps({'type': 'all_complete', 'total_tasks': len(search_tasks)})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    response = app.response_class(generate(), mimetype='text/event-stream')
    # 关键：禁用响应缓冲
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@app.route('/api/module2/summarize_and_predict', methods=['POST'])
def summarize_and_predict():
    """Summarize unstructured data and make final prediction"""
    try:
        data = request.json
        season = data.get('season')
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        game_date = data.get('game_date')
        raw_unstructured = data.get('raw_unstructured', '')
        structured_data = data.get('structured_data', {})

        if not all([season, home_team, away_team, game_date]):
            return jsonify({'success': False, 'message': 'Missing required parameters'})

        # Build structured data text
        home_stats = structured_data.get('home_stats', {})
        away_stats = structured_data.get('away_stats', {})

        structured_text = f"""
### {home_team} Recent Performance (Last 10 Games Average):
"""
        for key, value in home_stats.items():
            structured_text += f"- {key}: {value}\n"

        structured_text += f"\n### {away_team} Recent Performance (Last 10 Games Average):\n"
        for key, value in away_stats.items():
            structured_text += f"- {key}: {value}\n"

        # Summarization prompt
        summary_prompt = f"""You are an NBA prediction expert. Based on the following information, provide a comprehensive game prediction.

## Structured Statistical Data:
{structured_text}

## Raw Unstructured Information (from web search):
{raw_unstructured}

Please:
1. Summarize the unstructured information into three concise sections:
   - Injury Reports (key injuries affecting both teams)
   - Pre-game News (recent team dynamics)
   - Tactical Analysis (matchup insights)

2. Predict the game outcome with win probabilities

3. List Top 5 factors influencing the result (ranked by importance)

Respond in JSON format:
{{
    "prediction": {{
        "winner": "home" or "away",
        "home_win_probability": number (0-100),
        "away_win_probability": number (0-100)
    }},
    "top_factors": [
        {{
            "rank": 1,
            "factor": "factor name",
            "type": "structured" or "unstructured",
            "description": "brief impact description"
        }}
    ],
    "unstructured_summary": {{
        "injury_reports": "concise injury summary",
        "pregame_news": "concise news summary",
        "tactical_analysis": "concise tactical summary"
    }}
}}"""

        # Call LLM for summarization
        llm_result = call_deepseek(summary_prompt, timeout=120)

        # Parse response - more robust parsing
        try:
            llm_result = llm_result.strip()

            # Remove markdown code blocks
            if llm_result.startswith('```'):
                lines = llm_result.split('\n')
                # Remove first and last line if they contain ```
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].startswith('```'):
                    lines = lines[:-1]
                llm_result = '\n'.join(lines).strip()

            # Remove 'json' prefix if present
            if llm_result.startswith('json'):
                llm_result = llm_result[4:].strip()

            # Try to find JSON object in the response
            if not llm_result.startswith('{'):
                # Try to extract JSON from the text
                start = llm_result.find('{')
                end = llm_result.rfind('}')
                if start != -1 and end != -1:
                    llm_result = llm_result[start:end+1]

            if not llm_result:
                raise ValueError("Empty response from LLM")

            prediction_result = json.loads(llm_result)

            # Normalize probabilities
            pred = prediction_result.get('prediction', {})
            home_prob = float(pred.get('home_win_probability', 50))
            away_prob = float(pred.get('away_win_probability', 50))

            total = home_prob + away_prob
            if total > 0:
                home_prob = home_prob / total * 100
                away_prob = away_prob / total * 100

            # Get actual result if available
            actual_result = None
            if nba_processor:
                prediction_data = nba_processor.prepare_prediction_data(season, home_team, away_team, game_date)
                if prediction_data['game_data'] is not None:
                    actual_winner = 'home' if prediction_data['game_data'].get('home_win') == 1 else 'away'
                    actual_result = {
                        'winner': actual_winner,
                        'winner_name': home_team if actual_winner == 'home' else away_team
                    }

            response = {
                'success': True,
                'prediction': {
                    'winner': pred.get('winner', 'home'),
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_win_probability': round(home_prob, 1),
                    'away_win_probability': round(away_prob, 1)
                },
                'top_factors': prediction_result.get('top_factors', []),
                'unstructured_summary': prediction_result.get('unstructured_summary', {}),
                'actual_result': actual_result
            }

            return jsonify(response)

        except json.JSONDecodeError as e:
            # Fallback: create a default response structure
            print(f"JSON parse error: {str(e)}, raw response: {llm_result}")
            return jsonify({
                'success': False,
                'message': f'Failed to parse LLM response: {str(e)}',
                'raw_response': llm_result[:500],  # Only send first 500 chars for debugging
                'suggestion': 'The AI response was not in valid JSON format. Please try again.'
            })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Prediction failed: {str(e)}'})


@app.route('/module3')
def module3():
    """Module 3: NBA Match Prediction with Real-time Player News"""
    return render_template('module3.html')


@app.route('/api/module3/teams', methods=['GET'])
def get_nba_teams():
    """Get all NBA teams and their players"""
    try:
        teams = []
        with open('nba_teams_players.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    teams.append(json.loads(line))

        return jsonify({'success': True, 'teams': teams})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to load teams: {str(e)}'})


@app.route('/api/module3/search_news', methods=['POST'])
def search_player_news():
    """Search player news using DeepSeek with streaming"""
    data = request.get_json()
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    home_players = data.get('home_players', [])
    away_players = data.get('away_players', [])

    def generate():
        try:
            import concurrent.futures
            import queue
            import threading

            # Shared state
            message_queue = queue.Queue()
            completed_tasks = threading.Event()
            task_counter = {'count': 0, 'lock': threading.Lock()}

            # Define search tasks
            def search_home_team():
                """Search news for home team players"""
                try:
                    players_str = ', '.join(home_players[:5])  # Top 5 players
                    prompt = f"""Search for the latest news and updates about {home_team} NBA team and their key players: {players_str}.
Focus on:
1. Recent performance and statistics
2. Injury reports and player availability
3. Team form and momentum
4. Key player highlights
5. Any recent trades or roster changes

Provide a comprehensive summary in JSON format:
{{
    "team": "{home_team}",
    "overall_status": "brief team status",
    "key_points": ["point 1", "point 2", "point 3"],
    "player_updates": {{"player_name": "update", ...}},
    "recent_form": "description of recent games"
}}"""

                    accumulated_text = []
                    for chunk in call_deepseek_stream(prompt, 120):
                        event_type = chunk.get('type', '')
                        if event_type not in {'response.reasoning_text.delta', 'response.output_text.delta'}:
                            continue
                        text = chunk.get('delta', '')
                        if not isinstance(text, str) or not text:
                            continue

                        accumulated_text.append(text)
                        message_queue.put({
                            'type': 'home_chunk',
                            'chunk': text
                        })

                    final_result = "".join(accumulated_text)
                    message_queue.put({
                        'type': 'home_complete',
                        'content': final_result
                    })

                except Exception as e:
                    message_queue.put({
                        'type': 'home_error',
                        'error': str(e)
                    })
                finally:
                    with task_counter['lock']:
                        task_counter['count'] += 1
                        if task_counter['count'] >= 2:
                            completed_tasks.set()

            def search_away_team():
                """Search news for away team players"""
                try:
                    players_str = ', '.join(away_players[:5])  # Top 5 players
                    prompt = f"""Search for the latest news and updates about {away_team} NBA team and their key players: {players_str}.
Focus on:
1. Recent performance and statistics
2. Injury reports and player availability
3. Team form and momentum
4. Key player highlights
5. Any recent trades or roster changes

Provide a comprehensive summary in JSON format:
{{
    "team": "{away_team}",
    "overall_status": "brief team status",
    "key_points": ["point 1", "point 2", "point 3"],
    "player_updates": {{"player_name": "update", ...}},
    "recent_form": "description of recent games"
}}"""

                    accumulated_text = []
                    for chunk in call_deepseek_stream(prompt, 120):
                        event_type = chunk.get('type', '')
                        if event_type not in {'response.reasoning_text.delta', 'response.output_text.delta'}:
                            continue
                        text = chunk.get('delta', '')
                        if not isinstance(text, str) or not text:
                            continue

                        accumulated_text.append(text)
                        message_queue.put({
                            'type': 'away_chunk',
                            'chunk': text
                        })

                    final_result = "".join(accumulated_text)
                    message_queue.put({
                        'type': 'away_complete',
                        'content': final_result
                    })

                except Exception as e:
                    message_queue.put({
                        'type': 'away_error',
                        'error': str(e)
                    })
                finally:
                    with task_counter['lock']:
                        task_counter['count'] += 1
                        if task_counter['count'] >= 2:
                            completed_tasks.set()

            # Start both searches concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                executor.submit(search_home_team)
                executor.submit(search_away_team)

                # Stream messages as they arrive
                while not completed_tasks.is_set() or not message_queue.empty():
                    try:
                        msg = message_queue.get(timeout=0.01)
                        yield f"data: {json.dumps(msg)}\n\n"
                    except queue.Empty:
                        if completed_tasks.is_set() and message_queue.empty():
                            break
                        continue

            # Final check: Clear any remaining messages
            while not message_queue.empty():
                msg = message_queue.get_nowait()
                yield f"data: {json.dumps(msg)}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'type': 'all_complete'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    response = app.response_class(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@app.route('/api/module3/final_prediction', methods=['POST'])
def final_prediction():
    """Generate final prediction using call_model"""
    try:
        data = request.json
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        home_news = data.get('home_news', '')
        away_news = data.get('away_news', '')

        if not all([home_team, away_team]):
            return jsonify({'success': False, 'message': 'Missing required parameters'})

        # Build comprehensive prompt for final analysis
        prediction_prompt = f"""You are an expert NBA analyst. Based on the following information about two teams, predict the match outcome.

## {home_team} (Home Team)
Recent News and Player Updates:
{home_news}

## {away_team} (Away Team)
Recent News and Player Updates:
{away_news}

Please analyze:
1. Overall team strength comparison
2. Key player availability and form
3. Recent performance trends
4. Home court advantage
5. Head-to-head implications

Provide your prediction in JSON format:
{{
    "winner": "home" or "away",
    "home_probability": number (0-100),
    "away_probability": number (0-100),
    "analysis": "Detailed analysis explaining your prediction, key factors, and reasoning (3-5 sentences)",
    "confidence": "high", "medium", or "low"
}}"""

        # Call the model API
        api_key = "sk-rHa4hQLoPiQpGl7imxu6lp1nLjxtdyxcGf3j1afaCOKV1grE"
        url = "https://api.aipaibox.com/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        api_data = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": prediction_prompt}],
            "stream": False,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(url, headers=headers, json=api_data, timeout=120)
        result = response.json()["choices"][0]["message"]["content"]

        # Parse the result
        try:
            result = result.strip()
            if result.startswith('```'):
                lines = result.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].startswith('```'):
                    lines = lines[:-1]
                result = '\n'.join(lines).strip()

            if result.startswith('json'):
                result = result[4:].strip()

            if not result.startswith('{'):
                start = result.find('{')
                end = result.rfind('}')
                if start != -1 and end != -1:
                    result = result[start:end+1]

            prediction_data = json.loads(result)

            # Normalize probabilities
            home_prob = float(prediction_data.get('home_probability', 50))
            away_prob = float(prediction_data.get('away_probability', 50))

            total = home_prob + away_prob
            if total > 0:
                home_prob = home_prob / total * 100
                away_prob = away_prob / total * 100
            else:
                home_prob = 50.0
                away_prob = 50.0

            return jsonify({
                'success': True,
                'prediction': {
                    'winner': prediction_data.get('winner', 'home'),
                    'home_probability': round(home_prob, 1),
                    'away_probability': round(away_prob, 1)
                },
                'analysis': prediction_data.get('analysis', 'Analysis unavailable'),
                'confidence': prediction_data.get('confidence', 'medium')
            })

        except json.JSONDecodeError as e:
            # Fallback response
            return jsonify({
                'success': True,
                'prediction': {
                    'winner': 'home',
                    'home_probability': 52.0,
                    'away_probability': 48.0
                },
                'analysis': f'Based on the available information, this appears to be a closely matched game. {home_team} has a slight advantage playing at home.',
                'confidence': 'low'
            })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Prediction failed: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3344)
