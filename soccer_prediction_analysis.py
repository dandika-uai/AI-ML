import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class SoccerResultPredictor:
    def __init__(self, data_path):
        """
        Initialize the Soccer Result Predictor
        
        Args:
            data_path (str): Path to the English Premier League dataset
        """
        self.data_path = data_path
        self.df = None
        self.features = None
        self.target = None
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the soccer dataset
        """
        print("Loading and preprocessing data...")
        
        # Load the dataset
        self.df = pd.read_csv(self.data_path)
        
        # Display basic information about the dataset
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        
        # Check for missing values
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        
        # Remove rows with missing values
        self.df = self.df.dropna()
        
        print(f"\nDataset shape after removing missing values: {self.df.shape}")
        
    def create_features(self):
        """
        Create features for machine learning models
        """
        print("\nCreating features...")
        
        # Create team performance statistics
        team_stats = self._calculate_team_statistics()
        
        # Create features for each match
        features_list = []
        targets = []
        
        for idx, row in self.df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            result = row['Result']
            
            # Get team statistics
            home_stats = team_stats.get(home_team, self._get_default_stats())
            away_stats = team_stats.get(away_team, self._get_default_stats())
            
            # Create feature vector
            feature_vector = [
                home_stats['avg_goals_scored'],
                home_stats['avg_goals_conceded'],
                home_stats['win_rate'],
                home_stats['home_win_rate'],
                away_stats['avg_goals_scored'],
                away_stats['avg_goals_conceded'],
                away_stats['win_rate'],
                away_stats['away_win_rate'],
                home_stats['avg_goals_scored'] - away_stats['avg_goals_conceded'],  # Home attack vs Away defense
                away_stats['avg_goals_scored'] - home_stats['avg_goals_conceded'],  # Away attack vs Home defense
            ]
            
            features_list.append(feature_vector)
            targets.append(result)
        
        self.features = np.array(features_list)
        self.target = np.array(targets)
        
        # Feature names for reference
        self.feature_names = [
            'Home_Avg_Goals_Scored', 'Home_Avg_Goals_Conceded', 'Home_Win_Rate', 'Home_Home_Win_Rate',
            'Away_Avg_Goals_Scored', 'Away_Avg_Goals_Conceded', 'Away_Win_Rate', 'Away_Away_Win_Rate',
            'Home_Attack_vs_Away_Defense', 'Away_Attack_vs_Home_Defense'
        ]
        
        print(f"Features shape: {self.features.shape}")
        print(f"Target distribution:\n{pd.Series(self.target).value_counts()}")
        
    def _calculate_team_statistics(self):
        """
        Calculate historical statistics for each team
        """
        team_stats = defaultdict(lambda: {
            'goals_scored': [], 'goals_conceded': [], 'results': [],
            'home_results': [], 'away_results': []
        })
        
        # Calculate statistics for each team
        for _, row in self.df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            home_goals = row['HomeGoals']
            away_goals = row['AwayGoals']
            result = row['Result']
            
            # Home team statistics
            team_stats[home_team]['goals_scored'].append(home_goals)
            team_stats[home_team]['goals_conceded'].append(away_goals)
            team_stats[home_team]['results'].append(1 if result == 'H' else 0)
            team_stats[home_team]['home_results'].append(1 if result == 'H' else 0)
            
            # Away team statistics
            team_stats[away_team]['goals_scored'].append(away_goals)
            team_stats[away_team]['goals_conceded'].append(home_goals)
            team_stats[away_team]['results'].append(1 if result == 'A' else 0)
            team_stats[away_team]['away_results'].append(1 if result == 'A' else 0)
        
        # Convert to averages and rates
        processed_stats = {}
        for team, stats in team_stats.items():
            processed_stats[team] = {
                'avg_goals_scored': np.mean(stats['goals_scored']) if stats['goals_scored'] else 0,
                'avg_goals_conceded': np.mean(stats['goals_conceded']) if stats['goals_conceded'] else 0,
                'win_rate': np.mean(stats['results']) if stats['results'] else 0,
                'home_win_rate': np.mean(stats['home_results']) if stats['home_results'] else 0,
                'away_win_rate': np.mean(stats['away_results']) if stats['away_results'] else 0,
            }
        
        return processed_stats
    
    def _get_default_stats(self):
        """
        Return default statistics for teams with no historical data
        """
        return {
            'avg_goals_scored': 1.0,
            'avg_goals_conceded': 1.0,
            'win_rate': 0.33,
            'home_win_rate': 0.33,
            'away_win_rate': 0.33
        }
    
    def train_and_evaluate_models(self):
        """
        Train and evaluate all machine learning models
        """
        print("\nTraining and evaluating models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42, stratify=self.target
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n--- {model_name} ---")
            
            # Train the model
            if model_name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                X_train_model = X_train
                X_test_model = X_test
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_model, y_train, cv=5)
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'y_test': y_test
            }
            
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred))
        
        return results
    
    def visualize_results(self, results):
        """
        Create visualizations for model comparison and results
        """
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Soccer Result Prediction - Model Comparison', fontsize=16, fontweight='bold')
        
        # 1. Model Accuracy Comparison
        model_names = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in model_names]
        cv_means = [results[model]['cv_mean'] for model in model_names]
        
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        axes[0, 0].bar(x_pos + width/2, cv_means, width, label='CV Mean', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Target Distribution
        target_counts = pd.Series(self.target).value_counts()
        axes[0, 1].pie(target_counts.values, labels=['Home Win', 'Away Win', 'Draw'], 
                       autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Match Result Distribution')
        
        # 3. Confusion Matrix for best model
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        cm = confusion_matrix(results[best_model]['y_test'], results[best_model]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Away Win', 'Draw', 'Home Win'],
                    yticklabels=['Away Win', 'Draw', 'Home Win'],
                    ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 4. Feature Importance (for Logistic Regression)
        if 'Logistic Regression' in results:
            lr_model = self.models['Logistic Regression']
            if hasattr(lr_model, 'coef_'):
                # Get average absolute coefficients across classes
                feature_importance = np.mean(np.abs(lr_model.coef_), axis=0)
                feature_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=True)
                
                axes[1, 1].barh(feature_df['feature'], feature_df['importance'])
                axes[1, 1].set_title('Feature Importance (Logistic Regression)')
                axes[1, 1].set_xlabel('Average Absolute Coefficient')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n" + "="*50)
        print("MODEL COMPARISON SUMMARY")
        print("="*50)
        for model_name, result in results.items():
            print(f"{model_name:20} | Accuracy: {result['accuracy']:.4f} | CV Score: {result['cv_mean']:.4f}")
        
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        print(f"\nBest performing model: {best_model} with accuracy: {results[best_model]['accuracy']:.4f}")
    
    def predict_match(self, home_team, away_team):
        """
        Predict the result of a specific match
        
        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team
        """
        print(f"\nPredicting result for {home_team} vs {away_team}")
        
        # Calculate team statistics
        team_stats = self._calculate_team_statistics()
        
        # Get team statistics
        home_stats = team_stats.get(home_team, self._get_default_stats())
        away_stats = team_stats.get(away_team, self._get_default_stats())
        
        # Create feature vector
        feature_vector = np.array([[
            home_stats['avg_goals_scored'],
            home_stats['avg_goals_conceded'],
            home_stats['win_rate'],
            home_stats['home_win_rate'],
            away_stats['avg_goals_scored'],
            away_stats['avg_goals_conceded'],
            away_stats['win_rate'],
            away_stats['away_win_rate'],
            home_stats['avg_goals_scored'] - away_stats['avg_goals_conceded'],
            away_stats['avg_goals_scored'] - home_stats['avg_goals_conceded'],
        ]])
        
        # Make predictions with all models
        predictions = {}
        for model_name, model in self.models.items():
            if model_name == 'Logistic Regression':
                feature_scaled = self.scaler.transform(feature_vector)
                pred = model.predict(feature_scaled)[0]
                prob = model.predict_proba(feature_scaled)[0]
            else:
                pred = model.predict(feature_vector)[0]
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(feature_vector)[0]
                else:
                    prob = None
            
            predictions[model_name] = {'prediction': pred, 'probability': prob}
        
        # Display predictions
        result_mapping = {'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'}
        
        for model_name, pred_info in predictions.items():
            pred_result = result_mapping[pred_info['prediction']]
            print(f"{model_name}: {pred_result}")
            if pred_info['probability'] is not None:
                prob_dict = dict(zip(['A', 'D', 'H'], pred_info['probability']))
                print(f"  Probabilities - Home: {prob_dict['H']:.3f}, Draw: {prob_dict['D']:.3f}, Away: {prob_dict['A']:.3f}")
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("Starting Soccer Result Prediction Analysis")
        print("="*50)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Create features
        self.create_features()
        
        # Train and evaluate models
        results = self.train_and_evaluate_models()
        
        # Visualize results
        self.visualize_results(results)
        
        # Example prediction
        unique_teams = list(set(self.df['HomeTeam'].unique()) | set(self.df['AwayTeam'].unique()))
        if len(unique_teams) >= 2:
            sample_teams = np.random.choice(unique_teams, 2, replace=False)
            self.predict_match(sample_teams[0], sample_teams[1])
        
        print("\nAnalysis completed!")

# Main execution
if __name__ == "__main__":
    # Initialize the predictor
    predictor = SoccerResultPredictor('english_premier_league_dataset.csv')
    
    # Run the complete analysis
    predictor.run_complete_analysis()
    
    # You can also make individual predictions
    # predictor.predict_match('Arsenal', 'Liverpool')