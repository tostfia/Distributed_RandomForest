import os
import json
import time
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Import strutturali del sistema
from src.shared.config import SystemConfig

# Import delle utility condivise e del loader con campionamento probabilistico
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
from src.shared.utilities.datasplitter import StratifiedDataSplitter
from src.shared.utilities.featureselection import CICIDSFeatureSelector

def run_baseline():
    print("=====================================================")
    print("   AVVIO FASE TUNING & BASELINE SPECULARE AL CLUSTER ")
    print("=====================================================\n")

    # Seme globale sincronizzato con il cluster e Colab
    RANDOM_SEED = 123
    
    target_col = "Label"
    
    # Inizializziamo il gestore della configurazione di sistema (.env)
    sys_cfg = SystemConfig()
    print(f" • Ambiente infrastrutturale rilevato: {sys_cfg.env.upper()}")
    
    # ---------------------------------------------------------
    # FASE 1: ETL CON CAMPIONAMENTO PROBABILISTICO (Stile Colab)
    # ---------------------------------------------------------
    print(">>> FASE 1: Estrazione e Preprocessing Dati (ETL Speculare)")
    etl_start_time = time.perf_counter()
    
    dataset_type = "real"
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            try:
                tmp_cfg = json.load(f)
                dataset_type = tmp_cfg.get("dataset_type", "real")
            except:
                pass

    if dataset_type == "synthetic":
        print(f" • Tipo Dataset: Sintetico (Stress Test Task 2)")
        loader = SyntheticDataLoader(n_samples=100000, random_seed=RANDOM_SEED, target_column=target_col)
        df_clean = loader.load()
        
        splitter = StratifiedDataSplitter(target_column=target_col, test_size=0.2, random_state=RANDOM_SEED)
        train_df, test_df = splitter.split(df_clean)
    else:
        # 1. Recuperiamo il percorso dal SystemConfig o forziamo la cache locale conosciuta
        data_folder = getattr(sys_cfg, "dataset_path", None)
        
        # Se il config non punta a nulla o a una cartella vuota, forziamo la cartella corretta
        if not data_folder or not os.path.exists(data_folder) or data_folder == "./data":
            if os.path.exists("./dataset_cache"):
                data_folder = "./dataset_cache"
            else:
                data_folder = "./data"

        print(f" • Cartella sorgente identificata per dati reali: '{data_folder}'")
        print(f" • Tipo Dataset: Reale (Campionamento probabilistico 1%, Seed: {RANDOM_SEED})")
        
        # 2. Istanziamo il RawCSVDataLoader passandogli la CARTELLA CACHE.
        # Rileverà i file CSV ordinati, applicando lo skip_logic dell'1%
        loader = RawCSVDataLoader(data_url=data_folder, sample_fraction=0.01, dataset_seed=RANDOM_SEED)
        df_raw = loader.load()
        
        
        
        # Istanziamo i componenti di trasformazione
        preprocessor = CICIDSPreprocessor(target_column=target_col)
        splitter = StratifiedDataSplitter(target_column=target_col, test_size=0.2, random_state=RANDOM_SEED)

        print(" • [ORDINE SPECULARE] Binarizzazione sul dato intero...")
        df_binarized = preprocessor.binarize_target(df_raw)
        
        print(" • [ORDINE SPECULARE] Esecuzione Split Stratificato...")
        train_df, test_df = splitter.split(df_binarized)

        print(" • [ORDINE SPECULARE] Preprocessing indipendente sul Train Set...")
        train_df = preprocessor.process(train_df)
        
        print(" • [ORDINE SPECULARE] Preprocessing indipendente sul Test Set...")
        test_df = preprocessor.process(test_df)

        print(" • [ORDINE SPECULARE] Applicazione Feature Selection Bilaterale...")
        fs = CICIDSFeatureSelector(target_column=target_col, correlation_threshold=0.05)
        train_df = fs.fit_transform(train_df)
        test_df = fs.transform(test_df)
        
    etl_time = time.perf_counter() - etl_start_time
    print(f"[OK] Pipeline ETL speculare completata in {etl_time:.4f} secondi.")
    
    # Separazione delle Feature dalle Label
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(f" • Volume Train: {X_train.shape} | Volume Test: {X_test.shape}")
    print("-" * 60)

    # ---------------------------------------------------------
    # FASE 2: TUNING MULTI-METRICA (RandomizedSearch 5-Fold)
    # ---------------------------------------------------------
    print("\n>>> FASE 2: Esplorazione Spazio Iperparametri (Tuning 5-Fold)...")
    
    param_dist = [
        {
            'n_estimators': [10, 20, 30],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': [None, 'balanced'],
            'bootstrap': [True],
            'max_samples':[0.5,0.7,0.8,1.0]
        },
        {
            'n_estimators': [10, 20, 30],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': [None, 'balanced'],
            'bootstrap': [False],
        },
    ]
            
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_SEED),
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring={'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1'},
        refit='f1',
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_SEED
    )

    search.fit(X_train, y_train)
    best_params = search.best_params_
    best_index = search.best_index_
    print(f"[OK] Tuning completato. Iperparametri ottimali: {best_params}")

    # Configurazione e creazione cartella di output dedicata
    OUTPUT_DIR = "./outputs_baseline"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    config_path_final = os.path.join(OUTPUT_DIR, "config.json")
    pickle_path_final = os.path.join(OUTPUT_DIR, "baseline_random_forest_completa.pkl")

    # ---------------------------------------------------------
    # FASE 3: SCRITTURA MANIFESTO CONFIG.JSON
    # ---------------------------------------------------------
    best_booststrap = bool(best_params.get("bootstrap", True))
    config_data = {
        "mode": "distributed",
        "dataset_type": dataset_type,
        "dataset_path": data_folder if dataset_type == "real" else "synthetic",
        "hyperparameters": {
            "n_estimators": int(best_params.get("n_estimators", 100)),
            "max_depth": best_params.get("max_depth") ,
            "min_samples_split": int(best_params.get("min_samples_split", 2)),
            "class_weight": best_params.get("class_weight", None),
            "bootstrap": best_booststrap,
            "max_samples": float(best_params.get("max_samples", 1.0)) if best_booststrap else 1.0,
            "tree_type": "classifier",
            "target_column": target_col,
            "random_state": int(RANDOM_SEED)
        }
    }
    
    with open(config_path_final, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"[OK] Manifesto 'config.json' salvato correttamente in: '{config_path_final}'")

    # Estrazione metriche storiche aggregate
    cv_results_extracted = {
        'test_accuracy': [search.cv_results_[f'split{i}_test_accuracy'][best_index] for i in range(5)],
        'test_precision': [search.cv_results_[f'split{i}_test_precision'][best_index] for i in range(5)],
        'test_recall': [search.cv_results_[f'split{i}_test_recall'][best_index] for i in range(5)],
        'test_f1': [search.cv_results_[f'split{i}_test_f1'][best_index] for i in range(5)],
        'fit_time': [search.cv_results_['mean_fit_time'][best_index]] * 5
    }
    tempo_medio_fold_tuning = search.cv_results_['mean_fit_time'][best_index]
    tempo_totale_tuning_config = tempo_medio_fold_tuning * 5

    # ---------------------------------------------------------
    # FASE 4: ADDESTRAMENTO FINALE (T_seq) & INFERENZA LOCALE
    # ---------------------------------------------------------
    print("\n>>> FASE 4: Addestramento Finale Monolitico per estrazione T_seq...")
    rf_kwargs = dict(
        n_estimators=config_data["hyperparameters"]["n_estimators"],
        max_depth=config_data["hyperparameters"]["max_depth"],
        min_samples_split=config_data["hyperparameters"]["min_samples_split"],
        bootstrap=config_data["hyperparameters"]["bootstrap"],
        class_weight=config_data["hyperparameters"]["class_weight"],
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    if config_data["hyperparameters"]["bootstrap"]:
        rf_kwargs["max_samples"] = config_data["hyperparameters"]["max_samples"]

    tree_clf  = RandomForestClassifier(**rf_kwargs)
    
    start_train_finale = time.perf_counter()
    tree_clf.fit(X_train, y_train)
    t_seq = time.perf_counter() - start_train_finale
    print(f"[OK] Fitting completato. T_seq ottenuto: {t_seq:.4f} secondi.")

    print("\n[LOCAL] Calcolo delle predizioni e latenza sul Test Set indipendente...")
    start_inferenza = time.perf_counter()
    local_preds = tree_clf.predict(X_test)
    tempo_inferenza_totale = time.perf_counter() - start_inferenza

    # Valutazione puntuale reale sul Test Set speculare
    test_accuracy = np.mean(local_preds == y_test)
    test_precision = precision_score(y_test, local_preds, zero_division=0)
    test_recall = recall_score(y_test, local_preds, zero_division=0)
    test_f1 = f1_score(y_test, local_preds, zero_division=0)
    cm = confusion_matrix(y_test, local_preds)

    # Persistenza coordinata dei metadati locali
    metadata_pipeline = {
        "modello_addestrato": tree_clf,
        "features_mappate": list(X_train.columns),
        "baseline_tempi_locali": {
            "tempo_totale_cv": tempo_totale_tuning_config,
            "tempo_medio_fold": tempo_medio_fold_tuning,
            "t_seq": t_seq,
            "tempo_inferenza_totale": tempo_inferenza_totale
        }
    }
    with open(pickle_path_final, "wb") as f:
        pickle.dump(metadata_pipeline, f)
    print(f"[OK] Pipeline locale (file .pkl) salvata in: '{pickle_path_final}'")

    # ---------------------------------------------------------
    # FASE 5: OUTPUT REPORT COMPLETO
    # ---------------------------------------------------------
    print("\n" + "═" * 75)
    print("                  REPORT ESTESO DI VALIDAZIONE E BENCHMARK")
    print("═" * 75)

    print("\n1. ANALISI DETTAGLIATA ITERAZIONE PER ITERAZIONE (ESTRATTA DA TUNING)")
    print("-" * 75)
    for i in range(5):
        print(f"  [Fold {i+1}/5] -> "
              f"Acc: {cv_results_extracted['test_accuracy'][i]*100:.2f}% | "
              f"Prec: {cv_results_extracted['test_precision'][i]*100:.2f}% | "
              f"Rec: {cv_results_extracted['test_recall'][i]*100:.2f}% | "
              f"F1: {cv_results_extracted['test_f1'][i]*100:.2f}% | "
              f"Tempo Fit Medio: {cv_results_extracted['fit_time'][i]:.3f}s")

    print("\n2. METRICHE PREVENTIVE AGGREGATE DAL TUNING (MEDIE +/- DEVIAZIONE STANDARD)")
    print("-" * 75)
    print(f"  ACCURATEZZA GLOBALE CV:   {np.mean(cv_results_extracted['test_accuracy']) * 100:.2f} %  (+/- {np.std(cv_results_extracted['test_accuracy']) * 100:.2f}%)")
    print(f"  PRECISION MEDIA CV:       {np.mean(cv_results_extracted['test_precision']) * 100:.2f} %  (+/- {np.std(cv_results_extracted['test_precision']) * 100:.2f}%)")
    print(f"  RECALL MEDIA CV:          {np.mean(cv_results_extracted['test_recall']) * 100:.2f} %  (+/- {np.std(cv_results_extracted['test_recall']) * 100:.2f}%)")
    print(f"  F1-SCORE MEDIO CV:        {np.mean(cv_results_extracted['test_f1']) * 100:.2f} %  (+/- {np.std(cv_results_extracted['test_f1']) * 100:.2f}%)")

    print("\n3. PERFORMANCE REALI SUL TEST SET INDIPENDENTE (TERRENO DI CONFRONTO CLUSTER)")
    print("-" * 75)
    print(f"  ACCURACY SUL TEST SET:    {test_accuracy * 100:.2f} %")
    print(f"  PRECISION SUL TEST SET:   {test_precision * 100:.2f} %")
    print(f"  RECALL SUL TEST SET:      {test_recall * 100:.2f} %")
    print(f"  F1-SCORE SUL TEST SET:    {test_f1 * 100:.2f} %")
    print("\n  Matrice di Confusione sul Test Set:")
    print(cm)

    print("\n4. DIAGNOSTICA TEMPORALE E PROFILAZIONE HARDWARE")
    print("-" * 75)
    print(f"  Tempo di Preprocessing (ETL Speculare):            {etl_time:.4f} secondi")
    print(f"  Tempo Totale Stimato dei Fit CV Config. Ottimale:  {tempo_totale_tuning_config:.4f} secondi")
    print(f"  Latenza Media di un Singolo Fold nel Tuning:       {tempo_medio_fold_tuning:.4f} secondi")
    print(f"  ADDESTRAMENTO FINALE LOCALE MONOLITICO (T_seq):    {t_seq:.4f} secondi  <-- BASELINE")
    print(f"  Tempo di Inferenza Locale su Test Set:             {tempo_inferenza_totale:.4f} secondi")
    print("═" * 75)

if __name__ == "__main__":
    run_baseline()