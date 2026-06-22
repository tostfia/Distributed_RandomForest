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
    # --- CONFIGURAZIONE STILISTICA REPORT ---
    LUNGHEZZA_LINEA = 80
    DOPPIA_LINEA = "═" * LUNGHEZZA_LINEA
    LINEA_SINGOLA = "─" * LUNGHEZZA_LINEA

    print("=====================================================")
    print("   AVVIO FASE TUNING & BASELINE SPECULARE AL CLUSTER ")
    print("=====================================================\n")

    # Seme globale sincronizzato con il cluster e Colab
    RANDOM_SEED = 123
    target_col = "Label"
    
    sys_cfg = SystemConfig()
    print(f" • Ambiente infrastrutturale rilevato: {sys_cfg.env.upper()}")
    
    # ---------------------------------------------------------
    # FASE 1: ETL CON CAMPIONAMENTO PROBABILISTICO (Stile Colab)
    # ---------------------------------------------------------
    print(">>> FASE 1: Estrazione e Preprocessing Dati")
    
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
        data_folder = getattr(sys_cfg, "dataset_path", None)
        
        if not data_folder or not os.path.exists(data_folder) or data_folder == "./data":
            if os.path.exists("./dataset_cache"):
                data_folder = "./dataset_cache"
            else:
                data_folder = "./data"

        print(f" • Cartella sorgente identificata per dati reali: '{data_folder}'")
        print(f" • Tipo Dataset: Reale (Campionamento probabilistico 1%, Seed: {RANDOM_SEED})")

        # [TIMER 1]: Misura l'I/O di caricamento dei file grezzi dal disco
        io_start_time = time.perf_counter()
        # 2. Istanziamo il RawCSVDataLoader passandogli la CARTELLA CACHE.
        loader = RawCSVDataLoader(data_url=data_folder, sample_fraction=0.01, dataset_seed=RANDOM_SEED)
        df_raw = loader.load()
        io_time = time.perf_counter() - io_start_time
        print(f"[OK] Caricamento dati (I/O) completato in {io_time:.4f} secondi.")

        # [TIMER 2]: Parte esattamente prima di istanziare i componenti di trasformazione computazionale
        preprocess_start_time = time.perf_counter()
        
        # Istanziamo i componenti di trasformazione
        preprocessor = CICIDSPreprocessor(target_column=target_col)
        splitter = StratifiedDataSplitter(target_column=target_col, test_size=0.2, random_state=RANDOM_SEED)

        print(" • Binarizzazione sul dato intero...")
        df_binarized = preprocessor.binarize_target(df_raw)
        
        print(" • Esecuzione Split Stratificato...")
        train_df, test_df = splitter.split(df_binarized)

        print(" • Preprocessing indipendente sul Train Set...")
        train_df = preprocessor.process(train_df)
        
        print(" • Preprocessing indipendente sul Test Set...")
        test_df = preprocessor.process(test_df)

        print(" • Applicazione Feature Selection Bilaterale...")
        fs = CICIDSFeatureSelector(target_column=target_col, correlation_threshold=0.05)
        train_df = fs.fit_transform(train_df)
        test_df = fs.transform(test_df)
        
        etl_time = time.perf_counter() - preprocess_start_time
        print(f"[OK] Trasformazione e Feature Selection completate in {etl_time:.4f} secondi.")
    
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
        n_jobs=1,
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
    # FASE 4: ADDESTRAMENTO FINALE & INFERENZA LOCALE
    # ---------------------------------------------------------
    print("\n>>> FASE 4: Addestramento Finale Monolitico per estrazione T_seq...")
    rf_kwargs = dict(
        n_estimators=config_data["hyperparameters"]["n_estimators"],
        max_depth=config_data["hyperparameters"]["max_depth"],
        min_samples_split=config_data["hyperparameters"]["min_samples_split"],
        bootstrap=config_data["hyperparameters"]["bootstrap"],
        class_weight=config_data["hyperparameters"]["class_weight"],
        n_jobs=1,
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
    print("\n" + DOPPIA_LINEA)
    print(f"{'REPORT ESTESO DI VALIDAZIONE E BENCHMARK':^{LUNGHEZZA_LINEA}}")
    print(DOPPIA_LINEA)

    print(f"\n1. ANALISI DETTAGLIATA ITERAZIONE PER ITERAZIONE (CROSS-VALIDATION)")
    print(LINEA_SINGOLA)
    print(f"  {'Fold':<10} | {'Accuratezza':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Tempo Fit':<10}")
    print(LINEA_SINGOLA)
    for i in range(5):
        print(f"  Fold {i+1:02d}/05  | "
              f"{cv_results_extracted['test_accuracy'][i]*100:10.2f}% | "
              f"{cv_results_extracted['test_precision'][i]*100:10.2f}% | "
              f"{cv_results_extracted['test_recall'][i]*100:10.2f}% | "
              f"{cv_results_extracted['test_f1'][i]*100:10.2f}% | "
              f"{cv_results_extracted['fit_time'][i]:.3f}s")
    print(LINEA_SINGOLA)

    print(f"\n2. METRICHE AGGREGATE DA TUNING (MEDIE ± DEVIAZIONE STANDARD)")
    print(LINEA_SINGOLA)
    metriche_cv = [
        ("ACCURATEZZA GLOBALE", cv_results_extracted['test_accuracy']),
        ("PRECISION MEDIA", cv_results_extracted['test_precision']),
        ("RECALL MEDIA", cv_results_extracted['test_recall']),
        ("F1-SCORE MEDIO", cv_results_extracted['test_f1'])
    ]
    for nome, array in metriche_cv:
        media = np.mean(array) * 100
        dev_std = np.std(array) * 100
        print(f"  ▸ {nome:<25} : {media:6.2f}%  (± {dev_std:.2f}%)")

    print(f"\n3. PERFORMANCE REALI SUL TEST SET INDIPENDENTE")
    print(LINEA_SINGOLA)
    print(f"  ▸ ACCURACY SUL TEST SET   : {test_accuracy * 100:6.2f}%")
    print(f"  ▸ PRECISION SUL TEST SET  : {test_precision * 100:6.2f}%")
    print(f"  ▸ RECALL SUL TEST SET     : {test_recall * 100:6.2f}%")
    print(f"  ▸ F1-SCORE SUL TEST SET   : {test_f1 * 100:6.2f}%")
    print("\n  Matrice di Confusione sul Test Set:")
    for riga in cm:
        print(" " * 6 + " ".join(f"[{val:4d}]" for val in riga))

    print(f"\n4. DIAGNOSTICA TEMPORALE E PROFILAZIONE HARDWARE")
    print(LINEA_SINGOLA)
    print(f"  • Tempo Totale di Cross-Validation     : {tempo_totale_tuning_config:8.4f} s")
    print(f"  • Tempo Medio per Singolo Fold (CV)    : {tempo_medio_fold_tuning:8.4f} s")
    print(f"  • Tempo di Caricamento Dati (I/O)      : {io_time:8.4f} s")
    print(f"  • Tempo di Trasformazione (Process)    : {etl_time:8.4f} s")
    print(f"  • Tempo Totale di Addestramento (Training Set)  : {t_seq:8.4f} s")
    print(f"  • Tempo Totale di Inferenza (Testing Set) : {tempo_inferenza_totale:8.4f} s")

if __name__ == "__main__":
    run_baseline()