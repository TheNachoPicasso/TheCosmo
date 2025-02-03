from flask import Flask, render_template, jsonify
import requests
import pandas as pd
import os
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import time

app = Flask(__name__)

# Utility: Convert American Odds to Implied Probability
def convert_american_odds_to_implied_prob(odds):
    try:
        odds = float(odds)
    except:
        return 0.0
    if odds > 0:
        return 100.0 / (odds + 100)
    elif odds < 0:
        return -odds / (-odds + 100)
    else:
        return 0.0

# Fetch Live Data from TheOddsAPI for NBA
def fetch_live_data():
    odds_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    odds_params = {
        "apiKey": "c98bbef59ce91813a160753c64273bbf",
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
        "dateFormat": "iso"
    }
    
    try:
        odds_response = requests.get(odds_url, params=odds_params)
        if odds_response.status_code != 200:
            print(f"Error fetching odds: {odds_response.status_code}")
            return pd.DataFrame()
            
        odds_data = odds_response.json()
        rows = []
        for game in odds_data:
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            game_time = game.get("commence_time", "")
            
            if game.get("bookmakers"):
                bookmaker = game["bookmakers"][0]
                h2h_market = next((m for m in bookmaker.get("markets", []) if m["key"] == "h2h"), None)
                spread_market = next((m for m in bookmaker.get("markets", []) if m["key"] == "spreads"), None)
                
                if h2h_market and spread_market:
                    home_odds = next((o["price"] for o in h2h_market["outcomes"] if o["name"] == home_team), None)
                    home_spread = next((o["point"] for o in spread_market["outcomes"] if o["name"] == home_team), 0)
                    
                    if home_odds:
                        rows.append({
                            "team": home_team,
                            "opponent": away_team,
                            "game_time": game_time,
                            "implied_probability": convert_american_odds_to_implied_prob(home_odds),
                            "team_efficiency": 1.0,
                            "rest_days": 3,
                            "betting_volume": 5000,
                            "spread": float(home_spread),
                            "is_home": True
                        })
                    
                    away_odds = next((o["price"] for o in h2h_market["outcomes"] if o["name"] == away_team), None)
                    away_spread = next((o["point"] for o in spread_market["outcomes"] if o["name"] == away_team), 0)
                    
                    if away_odds:
                        rows.append({
                            "team": away_team,
                            "opponent": home_team,
                            "game_time": game_time,
                            "implied_probability": convert_american_odds_to_implied_prob(away_odds),
                            "team_efficiency": 1.0,
                            "rest_days": 3,
                            "betting_volume": 5000,
                            "spread": float(away_spread),
                            "is_home": False
                        })
        
        if not rows:
            print("No games found for today")
            return pd.DataFrame()
            
        df = pd.DataFrame(rows)
        df['game_time'] = pd.to_datetime(df['game_time']).dt.strftime('%I:%M %p')
        print(f"Found {len(df)//2} games for today")
        return df
        
    except Exception as e:
        print(f"Error fetching live data: {str(e)}")
        return pd.DataFrame()

# Fetch Historical NBA Data
def fetch_historical_data():
    url = "https://api-nba-v1.p.rapidapi.com/games"
    headers = {
        "X-RapidAPI-Key": "fefadca6c6msh8fb9855d9f48775p1006e3jsnee919f99a97b",
        "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
    }
    
    current_year = datetime.date.today().year
    season = f"{current_year-1}-{current_year}" if datetime.date.today().month < 9 else f"{current_year}-{current_year+1}"
    
    params = {
        "season": season,
        "league": "standard"
    }
    
    nba_rows = []
    try:
        print(f"Fetching historical data for season {season}")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        games = data.get("response", [])
        
        if not games:
            print("No games found in response")
            return pd.DataFrame()
            
        print(f"Found {len(games)} games")
        
        teams_url = "https://api-nba-v1.p.rapidapi.com/teams/statistics"
        team_stats = {}
        
        for game in games:
            if game.get("status", {}).get("long") == "Finished":
                game_date = game.get("date", {}).get("start", "").split("T")[0]
                home_team_id = game.get("teams", {}).get("home", {}).get("id")
                away_team_id = game.get("teams", {}).get("visitors", {}).get("id")
                
                for team_id in [home_team_id, away_team_id]:
                    if team_id not in team_stats:
                        team_params = {
                            "season": season,
                            "id": team_id
                        }
                        team_response = requests.get(teams_url, headers=headers, params=team_params)
                        if team_response.status_code == 200:
                            stats_data = team_response.json().get("response", [])
                            if stats_data:
                                points_scored = float(stats_data[0].get("points", 100))
                                points_allowed = float(stats_data[0].get("points_against", 100))
                                team_stats[team_id] = points_scored / points_allowed
                            else:
                                team_stats[team_id] = 1.0
                        time.sleep(0.5)
                
                home_score = game.get("scores", {}).get("home", {}).get("points", 0)
                away_score = game.get("scores", {}).get("visitors", {}).get("points", 0)
                
                if home_score == 0 and away_score == 0:
                    continue
                
                home_rest = np.random.randint(1, 6)
                away_rest = np.random.randint(1, 6)
                
                nba_rows.append({
                    "team": game.get("teams", {}).get("home", {}).get("name", ""),
                    "team_efficiency": team_stats.get(home_team_id, 1.0),
                    "rest_days": home_rest,
                    "betting_volume": len(games),
                    "win": 1 if home_score > away_score else 0,
                    "game_date": game_date,
                    "score": home_score,
                    "opponent_score": away_score
                })
                
                nba_rows.append({
                    "team": game.get("teams", {}).get("visitors", {}).get("name", ""),
                    "team_efficiency": team_stats.get(away_team_id, 1.0),
                    "rest_days": away_rest,
                    "betting_volume": len(games),
                    "win": 1 if away_score > home_score else 0,
                    "game_date": game_date,
                    "score": away_score,
                    "opponent_score": home_score
                })
                
        historical_df = pd.DataFrame(nba_rows)
        
        if not historical_df.empty:
            print(f"Successfully collected {len(historical_df)} team performances")
            print(f"Date range: {historical_df['game_date'].min()} to {historical_df['game_date'].max()}")
            historical_df['game_date'] = pd.to_datetime(historical_df['game_date'])
            historical_df = historical_df.sort_values('game_date', ascending=False)
            
            historical_df['point_differential'] = historical_df['score'] - historical_df['opponent_score']
            historical_df['recent_form'] = historical_df.groupby('team')['win'].transform(lambda x: x.rolling(5, min_periods=1).mean())
            
            return historical_df
            
    except Exception as e:
        print(f"Error fetching historical data: {str(e)}")
        
    return pd.DataFrame()

# Update Historical Data File
def update_historical_data_file(new_data: pd.DataFrame, filename="historical_data.csv"):
    if os.path.exists(filename):
        old_data = pd.read_csv(filename)
        combined = pd.concat([old_data, new_data], ignore_index=True)
        combined.to_csv(filename, index=False)
    else:
        new_data.to_csv(filename, index=False)

# Train the Model
def train_model(filename="historical_data.csv"):
    if not os.path.exists(filename):
        print("No historical data found. Please update historical data first.")
        return None
    
    try:
        data = pd.read_csv(filename)
        if len(data) == 0:
            print("Historical data file is empty.")
            return None
            
        required_columns = ['team_efficiency', 'rest_days', 'betting_volume', 'win']
        if not all(col in data.columns for col in required_columns):
            print(f"Historical data missing required columns. Found columns: {data.columns.tolist()}")
            return None
            
        X = data[['team_efficiency', 'rest_days', 'betting_volume']]
        y = data['win']
        
        if len(X) < 10:
            print("Not enough data to train model. Need at least 10 samples.")
            return None
            
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        base_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
        calibrated_model.fit(X_train, y_train)
        return calibrated_model
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None

# Calculate Confidence Score
def calculate_confidence(predicted_prob, market_prob, efficiency_diff, rest_advantage, betting_volume, recent_form_diff):
    prob_distance = abs(predicted_prob - market_prob)
    efficiency_variance = abs(efficiency_diff) / 2
    rest_impact = abs(rest_advantage) / 10
    volume_factor = min(betting_volume / 10000, 1)
    form_impact = abs(recent_form_diff) / 0.5
    
    confidence = (
        prob_distance * 0.3 +  # 30% weight on probability difference
        efficiency_variance * 0.25 +  # 25% weight on team strength
        rest_impact * 0.2 +  # 20% weight on rest advantage
        volume_factor * 0.15 +  # 15% weight on betting volume
        form_impact * 0.1  # 10% weight on recent form
    ) * 100
    
    return confidence

# Predict and Filter Betting Opportunities
def predict_and_filter_opportunities(calibrated_model, live_data, threshold=0.05):
    feature_columns = ['team_efficiency', 'rest_days', 'betting_volume']
    home_games = live_data[live_data['is_home'] == True].reset_index(drop=True)
    away_games = live_data[live_data['is_home'] == False].reset_index(drop=True)
    
    matchups = pd.DataFrame()
    for i in range(len(home_games)):
        home = home_games.iloc[i]
        away = away_games.iloc[i]
        
        matchup_data = {
            'game_time': home['game_time'],
            'home_team': home['team'],
            'away_team': away['team'],
            'home_spread': home['spread'],
            'away_spread': away['spread'],
            'home_implied_prob': home['implied_probability'],
            'away_implied_prob': away['implied_probability'],
            'home_efficiency': home['team_efficiency'],
            'away_efficiency': away['team_efficiency'],
            'home_rest': home['rest_days'],
            'away_rest': away['rest_days'],
            'betting_volume': home['betting_volume'],
            'efficiency_diff': home['team_efficiency'] - away['team_efficiency'],
            'rest_advantage': home['rest_days'] - away['rest_days'],
            'market_implied_margin': abs(home['spread']),
            'home_court_factor': 3.5,
            'recent_form_diff': home.get('recent_form', 0.5) - away.get('recent_form', 0.5)
        }
        
        home_features = pd.DataFrame([home[feature_columns]])
        base_prob = calibrated_model.predict_proba(home_features)[0][1]
        
        matchup_data['home_adjusted_prob'] = base_prob * (
            (1 + 0.05 * matchup_data['efficiency_diff']) *
            (1 + 0.02 * matchup_data['rest_advantage']) *
            (1 + 0.03 * matchup_data['home_court_factor'])
        )
        
        matchup_data['away_adjusted_prob'] = 1 - matchup_data['home_adjusted_prob']
        
        matchup_data['home_edge'] = matchup_data['home_adjusted_prob'] / matchup_data['home_implied_prob'] - 1
        matchup_data['away_edge'] = matchup_data['away_adjusted_prob'] / matchup_data['away_implied_prob'] - 1
        
        matchup_data['home_confidence'] = calculate_confidence(
            predicted_prob=matchup_data['home_adjusted_prob'],
            market_prob=matchup_data['home_implied_prob'],
            efficiency_diff=matchup_data['efficiency_diff'],
            rest_advantage=matchup_data['rest_advantage'],
            betting_volume=matchup_data['betting_volume'],
            recent_form_diff=matchup_data['recent_form_diff']
        )
        
        matchup_data['away_confidence'] = calculate_confidence(
            predicted_prob=matchup_data['away_adjusted_prob'],
            market_prob=matchup_data['away_implied_prob'],
            efficiency_diff=-matchup_data['efficiency_diff'],
            rest_advantage=-matchup_data['rest_advantage'],
            betting_volume=matchup_data['betting_volume'],
            recent_form_diff=-matchup_data['recent_form_diff']
        )
        
        matchups = pd.concat([matchups, pd.DataFrame([matchup_data])], ignore_index=True)
    
    all_games = []
    for _, matchup in matchups.iterrows():
        all_games.append({
            'game_time': matchup['game_time'],
            'team': matchup['home_team'],
            'opponent': matchup['away_team'],
            'spread': matchup['home_spread'],
            'edge': matchup['home_edge'],
            'confidence': matchup['home_confidence'],
            'adjusted_prob': matchup['home_adjusted_prob'],
            'is_home': True,
            'efficiency_diff': matchup['efficiency_diff'],
            'rest_advantage': matchup['rest_advantage'],
            'recent_form_diff': matchup['recent_form_diff']
        })
        
        all_games.append({
            'game_time': matchup['game_time'],
            'team': matchup['away_team'],
            'opponent': matchup['home_team'],
            'spread': matchup['away_spread'],
            'edge': matchup['away_edge'],
            'confidence': matchup['away_confidence'],
            'adjusted_prob': matchup['away_adjusted_prob'],
            'is_home': False,
            'efficiency_diff': -matchup['efficiency_diff'],
            'rest_advantage': -matchup['rest_advantage'],
            'recent_form_diff': -matchup['recent_form_diff']
        })
    
    if not all_games:
        return pd.DataFrame()
    
    games_df = pd.DataFrame(all_games)
    
    games_df['bet_recommendation'] = games_df.apply(
        lambda row: (
            f"[{row['game_time']}] " +
            f"{row['team']} ({'+' if row['spread'] > 0 else ''}{row['spread']:.1f}) " +
            f"vs {row['opponent']} " +
            f"[Edge: {row['edge']*100:.1f}%, Win Prob: {row['adjusted_prob']*100:.1f}%]\n" +
            f"• Efficiency Diff: {row['efficiency_diff']:.2f}\n" +
            f"• Rest Advantage: {row['rest_advantage']:.1f} days\n" +
            f"• Recent Form Diff: {row['recent_form_diff']:.2f}\n" +
            f"• {'Home' if row['is_home'] else 'Away'} Game"
        ),
        axis=1
    )
    
    return games_df.sort_values('confidence', ascending=False)

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    if not os.path.exists("historical_data.csv"):
        historical_df = fetch_historical_data()
        if not historical_df.empty:
            update_historical_data_file(historical_df)
        else:
            return jsonify({"error": "Could not fetch historical data"}), 500

    live_data = fetch_live_data()
    if live_data.empty:
        return jsonify({"error": "No live games available today"}), 500
    
    model = train_model()
    if model is None:
        return jsonify({"error": "Model training failed. Check historical data."}), 500
    
    games = predict_and_filter_opportunities(model, live_data)
    
    if games.empty:
        return jsonify({
            "message": "No games found for today",
            "predictions": []
        })
    
    predictions = []
    for _, row in games.iterrows():
        predictions.append({
            "team": f"{row['team']} ({'+' if row['spread'] > 0 else ''}{row['spread']:.1f})",
            "adjusted_prob": f"{row['adjusted_prob']*100:.1f}%",
            "confidence": f"{row['confidence']:.1f}%",
            "edge": f"{row['edge']*100:.1f}%"
        })
    
    return jsonify({
        "message": f"Top Betting Opportunities Today",
        "predictions": predictions
    })

if __name__ == '__main__':
    app.run(debug=True)