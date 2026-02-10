"""
Generic inference function for loading models and making predictions
"""

import pandas as pd
import joblib
import numpy as np


def load_model(model_path: str):
    """
    Loads a model from a joblib file
    
    Args:
        model_path: Path to the model file (.joblib)
        
    Returns:
        Loaded model object
    """
    try:
        model = joblib.load(model_path)
        print(f"Modèle chargé avec succès: {model_path}")
        return model
    except FileNotFoundError:
        print(f"Erreur: Fichier modèle non trouvé à {model_path}")
        raise
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        raise


def predict(model, df_test: pd.DataFrame) -> np.ndarray:
    """
    Generic function to make predictions on a preprocessed test dataframe
    
    Note: Data preprocessing (cleaning, imputation, feature selection) 
    should be applied BEFORE calling this function
    
    Args:
        model: Loaded model object
        df_test: Preprocessed test dataframe (ready for prediction)
        
    Returns:
        Predictions array
    """
    print(f"\nPrédiction avec {df_test.shape[0]} samples et {df_test.shape[1]} features")
    
    # Make predictions
    predictions = model.predict(df_test)
    
    print(f"Prédictions complétées")
    print(f"  - Min: {predictions.min():.2f}")
    print(f"  - Max: {predictions.max():.2f}")
    print(f"  - Mean: {predictions.mean():.2f}")
    
    return predictions


if __name__ == "__main__":
    # Example usage
    model_path = r"C:\Users\MSI\housing_proj\model\xgboost.joblib"
    test_data_path = r"C:\Users\MSI\housing_proj\data\test\test_processed.csv"
    
    # Load model
    model = load_model(model_path)
    
    # Load preprocessed test data
    df_test = pd.read_csv(test_data_path)
    
    # Make predictions
    predictions = predict(model, df_test)
    
    # Display results
    print(f"\nRésultats de prédiction:")
    print(f"  - Nombre de prédictions: {len(predictions)}")
    print(f"  - Premières prédictions: {predictions[:5]}")
