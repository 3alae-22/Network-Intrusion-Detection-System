import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

FILE_PATH = "kdd_df.csv"
TRAIN_FILE_OUT = "kdd_train.csv"
TEST_FILE_OUT = "kdd_test.csv"

print(f"--- Démarrage de la création des datasets ---")

# 1. Charger les données sources
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Erreur : Fichier '{FILE_PATH}' non trouvé.")
    exit()

print(f"Données sources chargées : {df.shape}")

# 2. Nettoyer les données
df.drop(['num_outbound_cmds', 'is_host_login'], axis=1, inplace=True, errors='ignore')
df.drop_duplicates(inplace=True)

print(f"Données nettoyées, {len(df)} lignes uniques restantes.")

# 3. Séparer en Train (80%) et Test (20%)
# Nous stratifions sur 'attack_type' pour garantir que les deux
# fichiers ont une distribution similaire de classes d'attaques.
print("Séparation en ensembles d'entraînement et de test...")
df_train, df_test = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['attack_type']  # Stratification
)

# 4. Sauvegarder les nouveaux fichiers CSV
df_train.to_csv(TRAIN_FILE_OUT, index=False)
print(f"Ensemble d'entraînement sauvegardé : {TRAIN_FILE_OUT} ({len(df_train)} lignes)")

df_test.to_csv(TEST_FILE_OUT, index=False)
print(f"Ensemble de test sauvegardé : {TEST_FILE_OUT} ({len(df_test)} lignes)")

print("\n--- Création des datasets terminée ---")
