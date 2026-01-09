import pandas as pd
import numpy as np
import joblib
import warnings
import time
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder, 
    FunctionTransformer, 
    LabelEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV

# --- 0. Configuration ---
warnings.filterwarnings('ignore')
FILE_PATH = "kdd_train.csv"

# --- 1. Fonctions de base (chargement et pré-traitement) ---

def load_and_clean_data(filepath):
    """Charge les données, supprime les colonnes inutiles et les doublons."""
    print(f"Chargement et nettoyage de {filepath}...")
    df = pd.read_csv(filepath)
    df.drop(['num_outbound_cmds', 'is_host_login'], axis=1, inplace=True, errors='ignore')
    df.drop_duplicates(inplace=True)
    print(f"Données nettoyées, {len(df)} lignes uniques restantes.")
    return df

def get_preprocessor(df_features):
    """Crée et retourne le ColumnTransformer (préprocesseur)."""
    
    categorical_features = ['protocol_type', 'service', 'flag']
    skewed_features = ['src_bytes', 'urgent', 'num_compromised', 'num_root', 'su_attempted'
                       , 'num_file_creations', 'num_failed_logins', 'land', 'dst_bytes', 'num_shells'
                       , 'root_shell', 'num_access_files', 'hot', 'duration', 'is_guest_login'
                       , 'wrong_fragment', 'srv_count', 'dst_host_srv_diff_host_rate', 'diff_srv_rate']
    
    numeric_features = [col for col in df_features.columns 
                        if col not in categorical_features and col not in skewed_features]

    skewed_pipe = Pipeline([
        ('log_transform', FunctionTransformer(np.log1p)),
        ('scaler', StandardScaler())
    ])
    numeric_pipe = Pipeline([('scaler', StandardScaler())])
    categorical_pipe = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe, numeric_features),
            ('skewed', skewed_pipe, skewed_features),
            ('cat', categorical_pipe, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

# --- 2. Fonctions d'entraînement (optimisées) ---

def train_binary_model(df, preprocessor):
    """Optimise et entraîne le modèle binaire (Normal/Attaque)."""
    print("--- Début Optimisation Modèle Binaire (XGBoost) ---")
    start_time = time.time()
    
    # 1. Préparer les données
    df['target_binary'] = df['attack_type'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    y_raw_1 = df['target_binary']
    X1 = df.drop(['target', 'attack_type', 'target_binary'], axis=1)
    le_binary = LabelEncoder()
    y1 = le_binary.fit_transform(y_raw_1)

    # 2. Définir le pipeline (la "recette" de base)
    pipeline_binary = ImbPipeline(steps=[
        ('preprocess', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', xgb.XGBClassifier(
            random_state=42, 
            n_jobs=-1, 
            use_label_encoder=False,
            eval_metric='logloss'
        ))
    ])
    
    # 3. Définir la grille de paramètres à tester
    # 'model__' permet de cibler les paramètres du 'model' dans le pipeline
    param_distributions = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [5, 10, 15],
        'model__learning_rate': [0.01, 0.05, 0.1]
    }
    
    # 4. Configurer la recherche aléatoire
    # n_iter=10 : Teste 10 combinaisons au hasard (rapide)
    # cv=3 : Validation croisée à 3 plis (robuste)
    random_search = RandomizedSearchCV(
        estimator=pipeline_binary, 
        param_distributions=param_distributions, 
        n_iter=10, 
        cv=3, 
        n_jobs=-1, 
        verbose=2, # Affiche la progression
        random_state=42
    )
    
    # 5. Lancer l'optimisation
    print(f"Lancement de RandomizedSearchCV pour le modèle binaire sur {len(X1)} échantillons...")
    random_search.fit(X1, y1)
    
    print("\nMeilleurs paramètres (Binaire) :")
    print(random_search.best_params_)
    
    # 6. Sauvegarder le MEILLEUR modèle trouvé
    joblib.dump(random_search.best_estimator_, "binary_detector_pipeline.pkl")
    joblib.dump(le_binary, "binary_label_encoder.pkl")
    
    end_time = time.time()
    print(f"--- Modèle Binaire Optimisé et Sauvegardé en {end_time - start_time:.2f} secondes ---")

def train_multiclass_model(df, preprocessor):
    """Optimise et entraîne le modèle multiclasse (Types d'attaques)."""
    print("\n--- Début Optimisation Modèle Multiclasse (XGBoost) ---")
    start_time = time.time()
    
    # 1. Préparer les données
    df_multi = df[df['attack_type'] != 'normal'].copy()
    y_raw_2 = df_multi['target']
    X2 = df_multi.drop(['target', 'attack_type'], axis=1)
    le_multi = LabelEncoder()
    y2 = le_multi.fit_transform(y_raw_2)

    # 2. Définir le pipeline
    pipeline_multi = ImbPipeline(steps=[
        ('preprocess', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=1)),
        ('model', xgb.XGBClassifier(
            random_state=42, 
            n_jobs=-1, 
            use_label_encoder=False,
            eval_metric='mlogloss'
        ))
    ])
    
    # 3. Définir la grille de paramètres
    param_distributions = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [5, 10, 15],
        'model__learning_rate': [0.01, 0.05, 0.1]
    }
    
    # 4. Configurer la recherche aléatoire
    random_search_multi = RandomizedSearchCV(
        estimator=pipeline_multi, 
        param_distributions=param_distributions, 
        n_iter=10, 
        cv=3, 
        n_jobs=-1, 
        verbose=2,
        random_state=42
    )
    
    # 5. Lancer l'optimisation
    print(f"Lancement de RandomizedSearchCV pour le modèle multiclasse sur {len(X2)} échantillons...")
    random_search_multi.fit(X2, y2)
    
    print("\nMeilleurs paramètres (Multiclasse) :")
    print(random_search_multi.best_params_)
    
    # 6. Sauvegarder le MEILLEUR modèle
    joblib.dump(random_search_multi.best_estimator_, "multiclass_classifier_pipeline.pkl")
    joblib.dump(le_multi, "multiclass_label_encoder.pkl")
    
    end_time = time.time()
    print(f"--- Modèle Multiclasse Optimisé et Sauvegardé en {end_time - start_time:.2f} secondes ---")

# --- 3. Point d'entrée principal ---

def main():
    """Fonction principale pour orchestrer l'entraînement."""
    print("--- DÉBUT DU SCRIPT D'OPTIMISATION (XGBOOST) ---")
    
    # Étape 1 : Données
    df = load_and_clean_data(FILE_PATH)
    
    # Étape 2 : Préprocesseur
    feature_cols = [col for col in df.columns if col not in ['target', 'attack_type']]
    preprocessor = get_preprocessor(df[feature_cols])
    
    # Étape 3 : Optimiser et entraîner les deux modèles
    train_binary_model(df.copy(), preprocessor)
    train_multiclass_model(df.copy(), preprocessor)
    
    print("\n--- OPTIMISATION TERMINÉE ---")
    print("Tous les modèles ont été optimisés, entraînés et sauvegardés.")

if __name__ == "__main__":
    main()