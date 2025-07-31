#!/usr/bin/env python3
"""
Demo script for Soccer Result Prediction

This script demonstrates how to use the SoccerResultPredictor class
to analyze soccer match data and make predictions using different ML algorithms.
"""

from soccer_prediction_analysis import SoccerResultPredictor
import pandas as pd

def main():
    print("Soccer Result Prediction Demo")
    print("="*40)
    
    # Initialize the predictor
    predictor = SoccerResultPredictor('english_premier_league_dataset.csv')
    
    # Load and preprocess the data
    predictor.load_and_preprocess_data()
    
    # Create features for machine learning
    predictor.create_features()
    
    # Train and evaluate all models
    print("\nTraining models...")
    results = predictor.train_and_evaluate_models()
    
    # Show visualizations
    predictor.visualize_results(results)
    
    # Get some team names for demonstration
    df = pd.read_csv('english_premier_league_dataset.csv')
    teams = list(set(df['HomeTeam'].unique()))
    
    # Make some example predictions
    print("\n" + "="*50)
    print("EXAMPLE MATCH PREDICTIONS")
    print("="*50)
    
    # Example predictions with popular teams
    example_matches = [
        ('Arsenal', 'Liverpool'),
        ('Man United', 'Chelsea'),
        ('Tottenham', 'Newcastle')
    ]
    
    for home, away in example_matches:
        if home in teams and away in teams:
            predictor.predict_match(home, away)
            print("-" * 30)
    
    print("\nDemo completed! You can now:")
    print("1. Modify the script to test different team combinations")
    print("2. Experiment with different model parameters")
    print("3. Add more features to improve prediction accuracy")
    print("4. Use the predictor.predict_match() method for custom predictions")

if __name__ == "__main__":
    main()