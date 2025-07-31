#!/usr/bin/env python3
"""
Quick Prediction Script

A simple script to make quick predictions for specific soccer matches.
This is useful when you just want to predict a single match without running
the full analysis.
"""

from soccer_prediction_analysis import SoccerResultPredictor
import sys

def quick_predict(home_team, away_team):
    """
    Make a quick prediction for a specific match
    
    Args:
        home_team (str): Name of the home team
        away_team (str): Name of the away team
    """
    print(f"🏟️  Quick Match Prediction: {home_team} vs {away_team}")
    print("=" * 60)
    
    # Initialize and prepare the predictor
    print("Loading data and training models...")
    predictor = SoccerResultPredictor('english_premier_league_dataset.csv')
    predictor.load_and_preprocess_data()
    predictor.create_features()
    
    # Train models (suppress detailed output)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictor.train_and_evaluate_models()
    
    print("\n🔮 Predictions:")
    print("-" * 30)
    
    # Make the prediction
    predictor.predict_match(home_team, away_team)
    
    print("\n" + "=" * 60)
    print("✅ Prediction completed!")

def main():
    """
    Main function to handle command line arguments or interactive input
    """
    if len(sys.argv) == 3:
        # Command line arguments provided
        home_team = sys.argv[1]
        away_team = sys.argv[2]
        quick_predict(home_team, away_team)
    else:
        # Interactive mode
        print("⚽ Soccer Match Prediction Tool")
        print("=" * 40)
        print("\nAvailable teams in the dataset:")
        
        # Load data to show available teams
        import pandas as pd
        df = pd.read_csv('english_premier_league_dataset.csv')
        teams = sorted(list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())))
        
        # Display teams in columns
        for i, team in enumerate(teams):
            if i % 3 == 0:
                print()
            print(f"{team:<20}", end="")
        
        print("\n\n" + "-" * 40)
        
        # Get user input
        try:
            home_team = input("Enter home team name: ").strip()
            away_team = input("Enter away team name: ").strip()
            
            if home_team and away_team:
                print()
                quick_predict(home_team, away_team)
            else:
                print("❌ Please enter both team names.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Usage:")
    print("  python quick_prediction.py                    # Interactive mode")
    print("  python quick_prediction.py 'Arsenal' 'Chelsea'  # Direct prediction")
    print()
    
    main()