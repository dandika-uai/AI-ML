# Soccer Result Prediction with Machine Learning

This project demonstrates how to predict soccer match results using three different machine learning algorithms: **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Naive Bayes**. The analysis uses historical English Premier League data to train models and make predictions.

## 📁 Project Structure

```
AI-ML/
├── english_premier_league_dataset.csv    # Historical match data
├── players.csv                           # Player information
├── soccer_prediction_analysis.py         # Main analysis class
├── demo_predictions.py                   # Demonstration script
├── requirements.txt                      # Python dependencies
└── README.md                            # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Analysis

```bash
python soccer_prediction_analysis.py
```

### 3. Run the Demo

```bash
python demo_predictions.py
```

## 🔍 What the Analysis Does

### Data Processing
- Loads historical Premier League match data
- Creates team performance statistics (goals scored/conceded, win rates)
- Generates features for machine learning models
- Handles missing data and preprocessing

### Feature Engineering
The model creates the following features for each match:
- **Home team statistics**: Average goals scored/conceded, win rate, home win rate
- **Away team statistics**: Average goals scored/conceded, win rate, away win rate
- **Comparative features**: Attack vs defense matchups

### Machine Learning Models

1. **Logistic Regression**
   - Linear model for classification
   - Provides probability estimates
   - Good baseline performance

2. **K-Nearest Neighbors (KNN)**
   - Instance-based learning
   - Uses similarity to make predictions
   - Non-parametric approach

3. **Naive Bayes**
   - Probabilistic classifier
   - Assumes feature independence
   - Fast and efficient

### Model Evaluation
- **Accuracy scores** on test data
- **Cross-validation** for robust performance estimation
- **Classification reports** with precision, recall, F1-score
- **Confusion matrices** for detailed error analysis

## 📊 Visualizations

The analysis generates several visualizations:
- Model accuracy comparison
- Match result distribution
- Confusion matrix for the best model
- Feature importance (for Logistic Regression)

## 🎯 Making Predictions

You can predict match results using the `predict_match()` method:

```python
from soccer_prediction_analysis import SoccerResultPredictor

# Initialize predictor
predictor = SoccerResultPredictor('english_premier_league_dataset.csv')
predictor.load_and_preprocess_data()
predictor.create_features()

# Train models
predictor.train_and_evaluate_models()

# Make prediction
predictor.predict_match('Arsenal', 'Liverpool')
```

## 📈 Expected Results

Typical performance metrics you might see:
- **Accuracy**: 45-55% (soccer is inherently unpredictable!)
- **Cross-validation**: Consistent performance across folds
- **Best features**: Usually team attack/defense comparisons

## 🔧 Customization Options

### Adding More Features
You can enhance the model by adding:
- Recent form (last 5 matches)
- Head-to-head records
- Player ratings and injuries
- Home advantage factors
- Season timing effects

### Model Tuning
- Adjust KNN neighbors (`n_neighbors`)
- Modify Logistic Regression parameters
- Try different preprocessing techniques

### Example Customization

```python
# Modify the models in the SoccerResultPredictor class
self.models = {
    'Logistic Regression': LogisticRegression(C=0.1, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
    'Naive Bayes': GaussianNB(var_smoothing=1e-8)
}
```

## 📚 Understanding the Data

### Match Data Format
- **League**: Competition name
- **Date**: Match date
- **HomeTeam/AwayTeam**: Team names
- **HomeGoals/AwayGoals**: Goals scored
- **Result**: H (Home win), A (Away win), D (Draw)

### Key Insights
- Home advantage is real in soccer
- Team form and historical performance matter
- Goal difference is a strong predictor
- Some matches are inherently unpredictable

## 🎓 Learning Objectives

This project demonstrates:
- **Data preprocessing** for sports analytics
- **Feature engineering** from historical data
- **Multiple ML algorithms** comparison
- **Model evaluation** techniques
- **Practical prediction** implementation

## 🚨 Important Notes

1. **Soccer Unpredictability**: Even the best models struggle with soccer prediction due to the sport's inherent randomness
2. **Data Quality**: Results depend heavily on data quality and feature engineering
3. **Overfitting**: Be careful not to overfit to historical patterns
4. **Real-world Factors**: Many factors (injuries, weather, motivation) aren't captured in basic statistics

## 🔄 Next Steps

To improve the model:
1. Add more sophisticated features
2. Use ensemble methods
3. Incorporate real-time data
4. Consider deep learning approaches
5. Add player-level statistics

## 📞 Usage Tips

- Run the complete analysis first to understand the data
- Experiment with different team combinations
- Try modifying the feature engineering
- Compare results across different seasons
- Use cross-validation to assess model stability

Happy predicting! ⚽🤖