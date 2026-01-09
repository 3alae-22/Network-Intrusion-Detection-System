import pandas as pd
import time
import warnings
from predict_app import HierarchicalPredictor

# --- 0. Configuration ---
warnings.filterwarnings('ignore')

# Configuration des chemins (identique à predict_app.py)
paths = {
    'binary_model': 'binary_detector_pipeline.pkl',
    'multiclass_model': 'multiclass_classifier_pipeline.pkl',
    'binary_encoder': 'binary_label_encoder.pkl',
    'multiclass_encoder': 'multiclass_label_encoder.pkl'
}

# Fichier de données pour simuler le flux
DATA_SOURCE_FILE = "kdd_test.csv"

# Vitesse de la simulation (en secondes)
# 0.5 = 2 connexions par seconde
# 0.1 = 10 connexions par seconde
# 0.0 = Vitesse maximale (pour un test rapide)
SIMULATION_SPEED = 0.5 

def start_simulation():
    """
    Lance la simulation de surveillance réseau en temps réel.
    """
    print("--- Démarrage du Moniteur de Sécurité IA ---")
    
    # 1. Initialiser le "cerveau" IA
    try:
        predictor = HierarchicalPredictor(paths)
    except Exception as e:
        print(f"Erreur fatale : Impossible de charger les modèles IA. {e}")
        return

    # 2. Ouvrir la source de données (notre fichier de test)
    try:
        df_stream = pd.read_csv(DATA_SOURCE_FILE)
        # Isoler les caractéristiques (X)
        df_features = df_stream.drop(['target', 'attack_type'], axis=1, errors='ignore')
        
        # Obtenir les vraies étiquettes pour la comparaison (optionnel)
        df_ground_truth = df_stream['target']
        
    except FileNotFoundError:
        print(f"Erreur : Fichier source de données '{DATA_SOURCE_FILE}' non trouvé.")
        return
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier de données : {e}")
        return

    print(f"Source de données chargée. Simulation de {len(df_features)} connexions...")
    print("État : EN LIGNE. Surveillance en cours...")
    print("-" * 40)

    # 3. Boucle de simulation (lecture du flux ligne par ligne)
    attack_count = 0
    for i in range(len(df_features)):
        
        # Isoler la ligne actuelle (les données qui "arrivent")
        # .iloc[[i]] garde le format DataFrame, ce qui est crucial
        current_connection = df_features.iloc[[i]]
        
        # Obtenir la vraie étiquette pour info
        ground_truth = df_ground_truth.iloc[i]
        
        # --- Appel à l'IA ---
        prediction = predictor.predict(current_connection)[0]
        
        # --- Prise de Décision ---
        if prediction != "Trafic Normal":
            # Si l'IA détecte une attaque
            attack_count += 1
            
            # Formater l'alerte
            print(f"*** ALERTE SÉCURITÉ N°{attack_count} (Connexion #{i}) ***")
            print(f"  TYPE DÉTECTÉ : {prediction}")
            print(f"  Vraie étiquette: {ground_truth}")
            print(f"  Source (prot/serv): {current_connection['protocol_type'].values[0]} / {current_connection['service'].values[0]}")
            print("-" * 40)
            
        else:
            # Si le trafic est normal
            # Nous n'imprimons rien pour ne pas polluer la console
            # Sauf si vous voulez un mode "verbeux"
            # print(f"Connexion #{i}: Trafic Normal (OK)")
            pass
            
        # Simuler le temps réel
        if SIMULATION_SPEED > 0:
            time.sleep(SIMULATION_SPEED)

    print("--- FIN DE LA SIMULATION ---")
    print(f"Simulation terminée. {attack_count} alertes générées sur {len(df_features)} connexions.")

# --- Point d'entrée du script ---
if __name__ == "__main__":
    start_simulation()