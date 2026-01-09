import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score
)

# --- 0. Configuration ---
warnings.filterwarnings('ignore')

# Fichiers des modèles (identique à predict_app.py)
paths = {
    'binary_model': 'binary_detector_pipeline.pkl',
    'multiclass_model': 'multiclass_classifier_pipeline.pkl',
    'binary_encoder': 'binary_label_encoder.pkl',
    'multiclass_encoder': 'multiclass_label_encoder.pkl'
}

# Fichier de test
TEST_DATA_FILE = "kdd_test.csv"

def load_artifacts(paths):
    """Charge tous les modèles et encodeurs."""
    try:
        artifacts = {
            'model_1_binary': joblib.load(paths['binary_model']),
            'model_2_multi': joblib.load(paths['multiclass_model']),
            'le_1_binary': joblib.load(paths['binary_encoder']),
            'le_2_multi': joblib.load(paths['multiclass_encoder'])
        }
        print("Artefacts (modèles et encodeurs) chargés.")
        return artifacts
    except FileNotFoundError as e:
        print(f"Erreur : Fichier manquant. {e}")
        print("Veuillez exécuter 'train_models.py' avant de lancer l'évaluation.")
        return None

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """Génère et sauvegarde une matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Vraie étiquette')
    plt.xlabel('Étiquette prédite')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Matrice de confusion sauvegardée : {filename}")

def evaluate_binary_model(artifacts, df_test):
    """Évalue le modèle binaire."""
    print("\n--- ÉVALUATION : MODÈLE 1 (BINAIRE) ---")
    
    # Préparer les données
    X_test = df_test.drop(['target', 'attack_type'], axis=1, errors='ignore')
    
    # Créer la vérité terrain binaire ('normal' vs 'attack')
    y_true_labels = df_test['attack_type'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    
    # Encoder la vérité terrain
    le_binary = artifacts['le_1_binary']
    y_true_codes = le_binary.transform(y_true_labels)
    
    # 2. Faire les prédictions
    model_binary = artifacts['model_1_binary']
    y_pred_codes = model_binary.predict(X_test)
    
    # 3. Afficher les métriques
    target_names = le_binary.classes_
    print(classification_report(y_true_codes, y_pred_codes, target_names=target_names))
    
    # 4. Générer la matrice de confusion
    plot_confusion_matrix(y_true_codes, y_pred_codes, 
                          labels=le_binary.transform(target_names), 
                          title='Matrice de Confusion - Modèle Binaire', 
                          filename='confusion_matrix_binary.png')

def evaluate_multiclass_model(artifacts, df_test):
    """Évalue le modèle multiclasse (uniquement sur les attaques)."""
    print("\n--- ÉVALUATION : MODÈLE 2 (MULTICLASSE) ---")
    print("(Évaluation uniquement sur le trafic 'attaque' du set de test)")
    
    # 1. Filtrer les données pour ne garder que les attaques
    df_attacks = df_test[df_test['attack_type'] != 'normal'].copy()
    
    if len(df_attacks) == 0:
        print("Aucune attaque trouvée dans le set de test pour l'évaluation multiclasse.")
        return

    X_test_multi = df_attacks.drop(['target', 'attack_type'], axis=1, errors='ignore')
    
    # La vérité terrain est la colonne 'target' (ex: 'dos', 'probe'...)
    y_true_labels = df_attacks['target']
    
    # Encoder la vérité terrain
    le_multi = artifacts['le_2_multi']
    
    # Gérer les étiquettes inconnues dans le set de test
    # (celles que le LabelEncoder n'a jamais vues à l'entraînement)
    known_labels = set(le_multi.classes_)
    y_true_labels_filtered = y_true_labels[y_true_labels.isin(known_labels)]
    X_test_multi_filtered = X_test_multi[y_true_labels.isin(known_labels)]
    
    if len(y_true_labels_filtered) == 0:
        print("Aucune étiquette d'attaque connue dans le set de test.")
        return
        
    y_true_codes = le_multi.transform(y_true_labels_filtered)

    # 2. Faire les prédictions
    model_multi = artifacts['model_2_multi']
    y_pred_codes = model_multi.predict(X_test_multi_filtered)
    
    # 3. Afficher les métriques
    target_names = le_multi.classes_
    # Obtenir les codes numériques (ex: 0, 1, ... 21) pour TOUTES les classes connues
    labels_codes = le_multi.transform(target_names) 

    print(classification_report(
        y_true_codes, 
        y_pred_codes, 
        target_names=target_names,
        labels=labels_codes,  
        zero_division=0 
    ))
    # 4. Générer la matrice de confusion
    plot_confusion_matrix(y_true_codes, y_pred_codes, 
                          labels=le_multi.transform(target_names), 
                          title='Matrice de Confusion - Modèle Multiclasse (Attaques)', 
                          filename='confusion_matrix_multiclass.png')


def main():
    print("--- DÉBUT DE L'ÉVALUATION DES MODÈLES ---")
    
    # 1. Charger les modèles
    artifacts = load_artifacts(paths)
    if artifacts is None:
        return
        
    # 2. Charger les données de test
    try:
        df_test = pd.read_csv(TEST_DATA_FILE)
    except FileNotFoundError:
        print(f"Erreur : Fichier de test '{TEST_DATA_FILE}' non trouvé.")
        return
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier de test : {e}")
        return
    
    print(f"Données de test chargées : {len(df_test)} lignes.")
    
    # 3. Lancer les évaluations
    evaluate_binary_model(artifacts, df_test)
    evaluate_multiclass_model(artifacts, df_test)
    
    print("\n--- ÉVALUATION TERMINÉE ---")
    print("Rapports affichés ci-dessus. Matrices de confusion sauvegardées en .png.")

if __name__ == "__main__":
    main()