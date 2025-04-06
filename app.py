import logging
from flask import Flask, render_template, request, jsonify
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    DATA_DIR = Path('./data/ml-latest-small')
    RATING_SCALE = (0.5, 5.0)
    TOP_N_RECOMMENDATIONS = 20
    FINAL_RECOMMENDATIONS = 5
    MODEL_PARAMS = {
        'SVD': {
            'n_factors': 100,
            'lr_all': 0.005,
            'reg_all': 0.02,
            'n_epochs': 20
        },
        'KNNBasic': {
            'sim_options': {
                'name': 'pearson_baseline',
                'user_based': True,
                'min_support': 5
            },
            'k': 40
        }
    }

# Load and prepare the dataset
def load_data() -> Optional[pd.DataFrame]:
    try:
        ratings_path = Config.DATA_DIR / 'ratings.csv'
        data = pd.read_csv(ratings_path)
        logger.info(f"Successfully loaded {len(data)} ratings")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def evaluate_model(model, dataset) -> Tuple[Optional[float], Optional[float]]:
    try:
        # First fit the model on the full dataset for evaluation
        trainset = dataset.build_full_trainset()
        model.fit(trainset)
        
        # Perform cross-validation
        results = cross_validate(model, dataset, measures=['RMSE', 'MAE'], 
                               cv=5, verbose=False, n_jobs=-1)
        
        mean_rmse = results['test_rmse'].mean()
        mean_mae = results['test_mae'].mean()
        
        logger.info(f"Model evaluation - RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}")
        
        # Calculate F1 Score
        testset = trainset.build_testset()
        predictions = model.test(testset)
        
        # Use threshold of 3.5 for binary classification
        y_true = [1 if pred.r_ui >= 3.5 else 0 for pred in predictions]
        y_pred = [1 if pred.est >= 3.5 else 0 for pred in predictions]
        
        f1 = f1_score(y_true, y_pred)
        logger.info(f"F1 Score: {f1:.4f}")
        
        return mean_rmse, f1
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        return None, None

def train_model(data: pd.DataFrame) -> Tuple[Optional[Any], Optional[float], Optional[float]]:
    try:
        reader = Reader(rating_scale=Config.RATING_SCALE)
        dataset = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

        models = {
            'SVD': SVD(**Config.MODEL_PARAMS['SVD']),
            'KNNBasic': KNNBasic(**Config.MODEL_PARAMS['KNNBasic'])
        }

        best_model = None
        best_rmse = float('inf')
        best_f1 = 0

        for name, model in models.items():
            logger.info(f"Training and evaluating {name} model...")
            current_rmse, current_f1 = evaluate_model(model, dataset)
            
            if current_rmse and current_rmse < best_rmse:
                best_rmse = current_rmse
                best_model = model
                best_f1 = current_f1

        if best_model:
            # Retrain the best model on the full dataset
            trainset = dataset.build_full_trainset()
            best_model.fit(trainset)
            logger.info("Best model training completed")
            return best_model, best_rmse, best_f1
        
        return None, None, None

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None, None, None

def get_recommendations(movie_name: str, model) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        ratings_data = load_data()
        movies_path = Config.DATA_DIR / 'movies.csv'
        movies = pd.read_csv(movies_path)

        if ratings_data is None or movies.empty:
            return pd.DataFrame(), "Failed to load necessary data"

        # Case-insensitive search with partial matching
        movie_matches = movies[movies['title'].str.lower().str.contains(movie_name.lower(), na=False)]
        
        if movie_matches.empty:
            return pd.DataFrame(), f"No movies found matching '{movie_name}'"
        
        if len(movie_matches) > 1:
            # If multiple matches, return the closest match
            movie_id = movie_matches.iloc[0]['movieId']
            logger.info(f"Multiple matches found for '{movie_name}', using first match: {movie_matches.iloc[0]['title']}")
        else:
            movie_id = movie_matches.iloc[0]['movieId']

        # Get unique users who rated this movie highly
        similar_users = ratings_data[
            (ratings_data['movieId'] == movie_id) & 
            (ratings_data['rating'] >= 4.0)
        ]['userId'].unique()

        if len(similar_users) == 0:
            return pd.DataFrame(), "Not enough ratings for this movie to make recommendations"

        # Get movies rated by similar users
        predictions = []
        for movie in movies['movieId']:
            if movie != movie_id:
                # Make predictions using multiple similar users for better accuracy
                user_predictions = []
                for user in similar_users[:5]:  # Use top 5 similar users
                    pred = model.predict(uid=user, iid=movie)
                    user_predictions.append(pred.est)
                
                avg_prediction = np.mean(user_predictions)
                movie_info = movies[movies['movieId'] == movie].iloc[0]
                predictions.append({
                    'title': movie_info['title'],
                    'predicted_rating': avg_prediction,
                    'genres': movie_info['genres']
                })

        # Sort by predicted rating and add diversity
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        # Get top N diverse recommendations
        diverse_recommendations = []
        seen_genres = set()
        
        for pred in predictions[:Config.TOP_N_RECOMMENDATIONS]:
            pred_genres = set(pred['genres'].split('|'))
            # Add movie if it introduces at least one new genre
            if pred_genres - seen_genres:
                diverse_recommendations.append(pred)
                seen_genres.update(pred_genres)
            
            if len(diverse_recommendations) >= Config.FINAL_RECOMMENDATIONS:
                break

        recommended_df = pd.DataFrame(diverse_recommendations)
        return recommended_df, None

    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        return pd.DataFrame(), f"An error occurred: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = pd.DataFrame()
    error_message = None
    metrics = {}
    search_query = ''

    if request.method == 'POST':
        search_query = request.form.get('movie_name', '').strip()
        
        if not search_query:
            error_message = "Please enter a movie name"
        else:
            try:
                ratings_data = load_data()
                if ratings_data is None:
                    error_message = "Could not load ratings data"
                else:
                    model, rmse, f1 = train_model(ratings_data)
                    
                    if model is None:
                        error_message = "Failed to train recommendation model"
                    else:
                        metrics = {
                            'rmse': f"{rmse:.4f}" if rmse else "N/A",
                            'f1': f"{f1:.4f}" if f1 else "N/A"
                        }
                        
                        recommendations, rec_error = get_recommendations(search_query, model)
                        if rec_error:
                            error_message = rec_error
                        elif recommendations.empty:
                            error_message = "No recommendations found"
            
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                error_message = "An unexpected error occurred"
    
    return render_template('index.html',
                         recommendations=recommendations,
                         error_message=error_message,
                         metrics=metrics,
                         search_query=search_query)

if __name__ == '__main__':
    app.run(debug=True)
