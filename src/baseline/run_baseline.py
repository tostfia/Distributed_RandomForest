import os
import json
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_validate

from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
from src.shared.utilities.datasplitter import StratifiedDataSplitter

def run_baseline():
    print("=====================================================")
    print("      AVVIO BASELINE NON DISTRIBUITA (SINGOLO NODO)  ")
    print("=====================================================\n")

    # 1. Tentativo di lettura dal config.json per i soli iperparametri matematici
    config_path = "config.json"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                print("[CONFIG] config.json corrotto o illeggibile. Uso i default della baseline.")
    
    # 2. IPERPARAMETRI MATEMATICI (Estratti se presenti, altrimenti default standard di tesi)
    hp = config.get("hyperparameters", {})
    n_estimators = hp.get("n_estimators", 100)
    max_depth = hp.get("max_depth", None)  # None = illimitata, standard per Random Forest completo
    class_weight = hp.get("class_weight", "balanced")
    
    # 3. FORZATURA LOGICA CENTRALIZZATA SEQUENZIALE
    # La baseline allena un Random Forest standard: servono bootstrap e campionamento dei dati.
    bootstrap = True 
    max_samples = 0.2  # Campionamento classico al 20% per albero, tipico delle baseline sequenziali grandi
    
    # 4. CONFIGURAZIONE DATASET AGNOSTICA
    # Forziamo il caricamento del file locale sul singolo nodo, ignorando cosa dice il cluster.
    dataset_seed = config.get("dataset_seed", 123)
    
    # Verifichiamo se l'utente nel config precedente stava usando un dataset sintetico o reale
    if config.get("dataset_type") == "synthetic":
        dataset_type = "synthetic"
        dataset_url = "/app/data/sintetic_data.csv"
        print("[BASELINE] Rilevato target sintetico. Esecuzione su dati sintetici locali.")
    else:
        dataset_type = "real"
        dataset_url = "/app/data/dataset_finale_binarizzato.csv"
        print("[BASELINE] Esecuzione standard su dataset reale locale.")

    print(f" • Configurazione Modello -> Alberi: {n_estimators} | Profondità: {max_depth} | Bootstrap: {bootstrap} (samples: {max_samples})")
    print(f" • Configurazione Dati    -> Path: {dataset_url} ({dataset_type.upper()})")
    print("-" * 53)
    
    # ---------------------------------------------------------
    # FASE 1: ETL (Extract, Transform, Load)
    # ---------------------------------------------------------
    print(">>> FASE 1: Estrazione e Pulizia Dati (ETL)")
    etl_start_time = time.perf_counter()
    
    if dataset_type == "real":  # Dataset reale con campionamento
        print(f" • Tipo Dataset: Reale (Campionamento 1%, Seed: {dataset_seed})")
        loader = RawCSVDataLoader(
            data_url=dataset_url,
            sample_fraction=0.01,
            dataset_seed=dataset_seed
        )
        df_raw = loader.load()
        
        preprocessor = CICIDSPreprocessor()
        df_clean = preprocessor.process(df_raw)
    else:   # Dataset sintetico per stress test
        print(f" • Tipo Dataset: Sintetico (Stress Test Task 2)")
        loader = SyntheticDataLoader(n_samples=100000, random_seed=dataset_seed)
        df_clean = loader.load()
        
    etl_time = time.perf_counter() - etl_start_time
    
    print(f"\n[OK] ETL completato in {etl_time:.4f} secondi.")
    print(f" • Dimensioni Dataset finale: {df_clean.shape}")
    
    # ---------------------------------------------------------
    # FASE 2: SPLIT STRATIFICATO
    # ---------------------------------------------------------
    splitter = StratifiedDataSplitter(target_column="Label", test_size=0.2, random_state=dataset_seed)
    train_df, test_df = splitter.split(df_clean)

    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]
    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"]

    # ---------------------------------------------------------
    # FASE 3: TRAINING LOCALE E CROSS-VALIDATION
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("  INIZIALIZZAZIONE WORKFLOW: CONFIGURAZIONE METRICHE E VERIFICA MATRICI")
    print("=" * 70)
    print(f"  Features estratte per il training (colonne): {X_train.shape[1]}")
    print(f"  Volume totale campioni di Train (righe):     {X_train.shape[0]:,}".replace(',', '.'))
    print("=" * 70)

    # Inizializzazione modello
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_samples=max_samples,
        bootstrap=bootstrap,
        class_weight=class_weight,
        n_jobs=-1, 
        random_state=dataset_seed
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=dataset_seed)
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

    print("\n[1/4] Calcolo della Cross-Validation a 5 Fold (Multi-Metrica) in corso...")
    start_cv = time.perf_counter()
    cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring_metrics, return_train_score=False)
    tempo_totale_cv = time.perf_counter() - start_cv
    tempo_medio_fold = tempo_totale_cv / 5
    print("Calcolo Cross-Validation completato con successo.")

    print("\n[2/4] Fitting finale sull'intero dataset di Train per estrazione T_seq...")
    start_train_finale = time.perf_counter()
    model.fit(X_train, y_train)
    t_seq = time.perf_counter() - start_train_finale

    print("\n[3/4] Benchmarking dei tempi di inferenza su massa dati (Test Set)...")
    start_inferenza = time.perf_counter()
    _ = model.predict(X_test)
    tempo_inferenza_totale = time.perf_counter() - start_inferenza
    print("Addestramento finale e test di latenza completati.")

    # ---------------------------------------------------------
    # FASE 4: RISULTATI
    # ---------------------------------------------------------
    print("\n═" * 75)
    print("                 REPORT ESTESO DI VALIDAZIONE E BENCHMARK")
    print("═" * 75)

    print("\n1. ANALISI DETTAGLIATA ITERAZIONE PER ITERAZIONE (PER FOLD)")
    print("-" * 75)
    for i in range(5):
        print(f"  [Fold {i+1}/5] -> Acc: {cv_results['test_accuracy'][i]*100:.2f}% | "
              f"Prec: {cv_results['test_precision'][i]*100:.2f}% | "
              f"Rec: {cv_results['test_recall'][i]*100:.2f}% | "
              f"F1: {cv_results['test_f1'][i]*100:.2f}% | "
              f"Tempo Fit: {cv_results['fit_time'][i]:.3f}s")

    print("\n2. METRICHE PREVENTIVE AGGREGATE (MEDIE VIRTUALI +/- DEVIAZIONE STANDARD)")
    print("-" * 75)
    print(f"  ACCURATEZZA GLOBALE:      {cv_results['test_accuracy'].mean() * 100:.2f} %  (+/- {cv_results['test_accuracy'].std() * 100:.2f}%)")
    print(f"  PRECISION MEDIA:          {cv_results['test_precision'].mean() * 100:.2f} %  (+/- {cv_results['test_precision'].std() * 100:.2f}%)")
    print(f"  RECALL MEDIA (Sens.):     {cv_results['test_recall'].mean() * 100:.2f} %  (+/- {cv_results['test_recall'].std() * 100:.2f}%)")
    print(f"  F1-SCORE MEDIO:           {cv_results['test_f1'].mean() * 100:.2f} %  (+/- {cv_results['test_f1'].std() * 100:.2f}%)")

    print("\n3. DIAGNOSTICA TEMPORALE E PROFILAZIONE HARDWARE")
    print("-" * 75)
    print(f"  Tempo di Preprocessing (ETL):         {etl_time:.4f} secondi")
    print(f"  Tempo Totale della Cross-Validation:  {tempo_totale_cv:.4f} secondi")
    print(f"  Latenza Media di un Singolo Fold:     {tempo_medio_fold:.4f} secondi")
    print(f"  ADDESTRAMENTO FINALE (T_seq):         {t_seq:.4f} secondi  <-- BASELINE CLOUD")
    print(f"  Tempo di Inferenza su Test Set:       {tempo_inferenza_totale:.4f} secondi")
    print("═" * 75)

if __name__ == "__main__":
    run_baseline()