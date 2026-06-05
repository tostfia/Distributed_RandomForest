# PROVVISORIO
import os
import json
import time

from sklearn.ensemble import RandomForestClassifier

from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader

def run_baseline():
    print("=====================================================")
    print("      AVVIO BASELINE NON DISTRIBUITA (LOCAL)         ")
    print("=====================================================\n")

    # 1. Lettura configurazione dal config.json
    config_path = "config.json"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    
    # Estraiamo i parametri
    dataset_type = config.get("dataset_type", "real")
    dataset_url = config.get("dataset_path", "./dataset_completo")
    dataset_seed = config.get("dataset_seed", 123)
    
    hp = config.get("hyperparameters", {})
    n_estimators = hp.get("n_estimators", 100)
    max_depth = hp.get("max_depth", 10)
    
    # ---------------------------------------------------------
    # FASE 1: ETL (Extract, Transform, Load)
    # ---------------------------------------------------------
    print(">>> FASE 1: Estrazione e Pulizia Dati (ETL)")
    etl_start_time = time.time()
    
    if dataset_type == "real":  # Dataset reale con campionamento
        print(f" • Tipo Dataset: Reale (Campionamento 1%, Seed: {dataset_seed})")
        # Extract
        loader = RawCSVDataLoader(
            data_url=dataset_url,
            sample_fraction=0.01,
            dataset_seed=dataset_seed
        )
        df_raw = loader.load()
        
        # Transform
        preprocessor = CICIDSPreprocessor()
        df_clean = preprocessor.process(df_raw)
    else:   # Dataset sintetico per stress test (nessun pre-processing necessario)
        print(f" • Tipo Dataset: Sintetico (Stress Test Task 2)")
        loader = SyntheticDataLoader(n_samples=100000, random_seed=dataset_seed)
        df_clean = loader.load()
        
    etl_time = time.time() - etl_start_time
    
    print(f"\n[OK] ETL completato in {etl_time:.2f} secondi.")
    print(f" • Dimensioni Dataset finale: {df_clean.shape}")
    
    # ---------------------------------------------------------
    # FASE 2: TRAINING LOCALE
    # ---------------------------------------------------------
    print("\n>>> FASE 2: Addestramento Random Forest (Local)")
    print(f" • Alberi (n_estimators): {n_estimators}")
    print(f" • Profondità massima: {max_depth}")
    
    X = df_clean.drop(columns=["Label"])
    y = df_clean["Label"]
    
    # Inizializziamo il modello sfruttando tutti i core della macchina locale (n_jobs=-1)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=hp.get("class_weight", "balanced"),
        n_jobs=-1, 
        random_state=dataset_seed
    )
    
    training_start_time = time.time()
    
    # Addestramento sequenziale su una sola macchina
    model.fit(X, y)
    
    training_time = time.time() - training_start_time
    
    print(f"\n[OK] Training completato in {training_time:.2f} secondi.")
    
    # ---------------------------------------------------------
    # RISULTATI
    # ---------------------------------------------------------
    print("\n=====================================================")
    print("                 RISULTATI BASELINE                  ")
    print("=====================================================")
    print(f" Tempo di Preprocessing (ETL): {etl_time:.2f} s")
    print(f" Tempo di Addestramento:       {training_time:.2f} s")
    print(f" TEMPO TOTALE (End-to-End):    {(etl_time + training_time):.2f} s")
    print("=====================================================\n")

if __name__ == "__main__":
    run_baseline()