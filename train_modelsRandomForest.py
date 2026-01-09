import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder, 
    FunctionTransformer, 
    LabelEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# --- 0. Configuration ---
warnings.filterwarnings('ignore')
FILE_PATH = "kdd_train.csv"

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

def train_binary_model(df, preprocessor):
    """Entraîne et sauvegarde le modèle binaire (Normal/Attaque)."""
    print("Début de l'entraînement du modèle binaire (RandomForest)...")
    
    # 1. Préparer les données
    df['target_binary'] = df['attack_type'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    y_raw_1 = df['target_binary']
    X1 = df.drop(['target', 'attack_type', 'target_binary'], axis=1)
    
    # 2. Encodage Y
    le_binary = LabelEncoder()
    y1 = le_binary.fit_transform(y_raw_1)
    
    # 3. Séparation Train/Test (SUPPRIMÉE)
    print(f"Entraînement binaire sur {len(X1)} échantillons.")

    # 4. Pipeline
    pipeline_binary = ImbPipeline(steps=[
        ('preprocess', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    # 5. Entraînement sur TOUTES les données
    pipeline_binary.fit(X1, y1)
    
    # 6. Sauvegarde des artefacts
    joblib.dump(pipeline_binary, "binary_detector_pipeline.pkl")
    joblib.dump(le_binary, "binary_label_encoder.pkl")
    print("Modèle binaire (RandomForest) sauvegardé.")

def train_multiclass_model(df, preprocessor):
    """Entraîne et sauvegarde le modèle multiclasse (Types d'attaques)."""
    print("Début de l'entraînement du modèle multiclasse (RandomForest)...")
    
    # 1. Préparer les données
    df_multi = df[df['attack_type'] != 'normal'].copy()
    y_raw_2 = df_multi['target']
    X2 = df_multi.drop(['target', 'attack_type'], axis=1)

    # 2. Encodage Y
    le_multi = LabelEncoder()
    y2 = le_multi.fit_transform(y_raw_2)
    print(f"Entraînement multiclasse sur {len(X2)} échantillons d'attaques.")

    # 4. Pipeline (avec SMOTE ajusté)
    smote_multi = SMOTE(random_state=42, k_neighbors=1)
    pipeline_multi = ImbPipeline(steps=[
        ('preprocess', preprocessor),
        ('smote', smote_multi),
        ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    # 5. Entraînement sur TOUTES les données
    pipeline_multi.fit(X2, y2)
    
    # 6. Sauvegarde des artefacts
    joblib.dump(pipeline_multi, "multiclass_classifier_pipeline.pkl")
    joblib.dump(le_multi, "multiclass_label_encoder.pkl")
    print("Modèle multiclasse (RandomForest) sauvegardé.")

def main():
    """Fonction principale pour orchestrer l'entraînement."""
    print("--- DÉBUT DU SCRIPT D'ENTRAÎNEMENT (RANDOM FOREST) ---")
    
    df = load_and_clean_data(FILE_PATH)
    feature_cols = [col for col in df.columns if col not in ['target', 'attack_type']]
    preprocessor = get_preprocessor(df[feature_cols])
    
    train_binary_model(df.copy(), preprocessor)
    train_multiclass_model(df.copy(), preprocessor)
    
    print("--- FIN DU SCRIPT D'ENTRAÎNEMENT (RANDOM FOREST) ---")

if __name__ == "__main__":
    main()