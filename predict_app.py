import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

class HierarchicalPredictor:
    """
    Classe pour charger les modèles hiérarchiques et faire des prédictions.
    """
    def __init__(self, model_paths):
        """
        Charge les 4 artefacts (modèles et encodeurs) au démarrage.
        """
        try:
            self.model_1_binary = joblib.load(model_paths['binary_model'])
            self.model_2_multi = joblib.load(model_paths['multiclass_model'])
            self.le_1_binary = joblib.load(model_paths['binary_encoder'])
            self.le_2_multi = joblib.load(model_paths['multiclass_encoder'])
            print("Prédicteur hiérarchique chargé et prêt.")
        except FileNotFoundError as e:
            print(f"Erreur : Fichier modèle manquant. {e}")
            print("Veuillez exécuter 'train_models.py' avant de lancer cette application.")
            self.model_1_binary = None

    def predict(self, data_row_df):
        """
        Prédit le type d'intrusion pour une nouvelle ligne de données.
        :param data_row_df: DataFrame Pandas contenant une ou plusieurs lignes de données.
        :return: Une liste de prédictions (strings).
        """
        if not self.model_1_binary:
            return ["Erreur: Modèles non chargés."]

        # --- Modèle 1 : Binaire (Normal ou Attaque) ---
        pred_1_codes = self.model_1_binary.predict(data_row_df)
        pred_1_labels = self.le_1_binary.inverse_transform(pred_1_codes)
        
        final_predictions = []
        
        # Itérer sur chaque prédiction (au cas où data_row_df a plusieurs lignes)
        for i, label in enumerate(pred_1_labels):
            if label == 'normal':
                final_predictions.append("Trafic Normal")
            else:
                # --- Modèle 2 : Multiclasse (Quel type d'attaque ?) ---
                # Prédire uniquement sur la ligne 'i'
                row = data_row_df.iloc[[i]]
                pred_2_code = self.model_2_multi.predict(row)[0]
                pred_2_label = self.le_2_multi.inverse_transform([pred_2_code])[0]
                final_predictions.append(f"ATTAQUE DÉTECTÉE (Type: {pred_2_label})")
                
        return final_predictions

# --- Point d'entrée pour tester l'application ---
if __name__ == "__main__":
    
    # 1. Définir les chemins des artefacts
    paths = {
        'binary_model': 'binary_detector_pipeline.pkl',
        'multiclass_model': 'multiclass_classifier_pipeline.pkl',
        'binary_encoder': 'binary_label_encoder.pkl',
        'multiclass_encoder': 'multiclass_label_encoder.pkl'
    }
    
    # 2. Initialiser le prédicteur (charge les modèles)
    predictor = HierarchicalPredictor(paths)

    # 3. Créer des données de test (ex: à partir du CSV)
    # Dans une vraie application, ces données viendraient d'une requête réseau, d'un fichier, etc.
    try:
        df_test_data = pd.read_csv("kdd_test.csv")
    except FileNotFoundError:
        print("Fichier kdd_df.csv non trouvé pour le test.")
        exit()

    # (Nous n'avons pas besoin des colonnes cibles ici)
    df_test_data = df_test_data.drop(['target', 'attack_type'], axis=1, errors='ignore')
    
    # Test
    normal_sample = df_test_data.iloc[[145]]
    print(f"\nTest avec un échantillon (index 145):")
    prediction = predictor.predict(normal_sample)
    print(f"Résultat: {prediction[0]}")

    # Test
    attack_sample = df_test_data.iloc[[33]]
    print(f"\nTest avec un échantillon (index 33):")
    prediction = predictor.predict(attack_sample)
    print(f"Résultat: {prediction[0]}")

    # Test
    attack_sample_2 = df_test_data.iloc[[89]]
    print(f"\nTest avec un échantillon (index 89):")
    prediction = predictor.predict(attack_sample_2)
    print(f"Résultat: {prediction[0]}")