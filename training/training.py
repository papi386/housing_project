"""
Script de training pour modèles ML
"""

import pandas as pd
import numpy as np
import joblib
import os


def save_model(model, output_path: str) -> None:
    """
    Fonction générique pour sauvegarder un modèle en format .joblib
    
    Args:
        model: Modèle entraîné (XGBoost, RandomForest, etc.)
        output_path: Chemin complet de sauvegarde du modèle
    """
    # Créer les répertoires s'ils n'existent pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sauvegarder le modèle
    joblib.dump(model, output_path)
    print(f"\nModèle sauvegardé: {output_path}")
    
    # Vérifier la sauvegarde
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"  - Taille: {file_size / 1024:.2f} KB")


def train_xgboost(X, y, params: dict = None):
    """
    Entraîne un modèle XGBoost
    
    Args:
        X: Features (DataFrame ou array)
        y: Target (Series ou array)
        params: Dictionnaire des paramètres XGBoost
        
    Returns:
        Modèle XGBoost entraîné
    """
    import xgboost as xgb
    
    # Paramètres par défaut
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 1
        }
    
    print(f"Entraînement XGBoost avec paramètres: {params}")
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=True)
    
    return model


if __name__ == "__main__":
    # Exemple d'utilisation
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Charger les données
    df_train = pd.read_csv(r"C:\Users\MSI\housing_proj\data\train\trains.csv")
    
    # Préparer les données
    y = df_train['price']
    columns_to_drop = ['price', 'title', 'location']
    X = df_train.drop(columns=[col for col in columns_to_drop if col in df_train.columns])
    X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean())
    
    # Entraîner le modèle
    model = train_xgboost(X, y)
    
    # Sauvegarder le modèle
    save_model(model, r"C:\Users\MSI\housing_proj\model\xgboost.joblib")
    
    # Évaluer
    y_pred = model.predict(X)
    print(f"\nMétriques:")
    print(f"  - RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}")
    print(f"  - MAE: {mean_absolute_error(y, y_pred):.2f}")
    print(f"  - R²: {r2_score(y, y_pred):.4f}")
