import os
import json
import time
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
from src.shared.config import SystemConfig

# Import delle utility condivise e del loader con campionamento probabilistico
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
from src.shared.utilities.datasplitter import StratifiedDataSplitter
from src.shared.utilities.featureselection import CICIDSFeatureSelector

# ---------------------------------------------------------------------------
# Configurazione di riferimento FISSA per il task sintetico di REGRESSIONE.
#
# NOTA METODOLOGICA: a differenza del task sintetico di classificazione (dove
# ereditare gli iperparametri dal tuning sul dataset reale ha senso, perché è
# lo stesso algoritmo/task), per la regressione non esiste alcuna garanzia che
# gli iperparametri ottimizzati per un RandomForestClassifier su un problema
# di classificazione binaria sbilanciata siano sensati per un
# RandomForestRegressor su dati sintetici continui e bilanciati.
#
# Si usa quindi una configurazione dichiarata a priori (non derivata da
# tuning) e tenuta IDENTICA in ogni esperimento di scalabilità — baseline
# locale e ogni run del cluster distribuito, a qualunque numero di worker —
# in modo da isolare l'effetto della scalabilità (numero di nodi, dimensione
# del dataset) dalla complessità del modello.
# ---------------------------------------------------------------------------
SYNTHETIC_REGRESSOR_REFERENCE_HP = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "bootstrap": True,
    "max_samples": 1.0,
}


def run_baseline():
    # --- CONFIGURAZIONE STILISTICA REPORT ---
    LUNGHEZZA_LINEA = 80
    DOPPIA_LINEA = "═" * LUNGHEZZA_LINEA
    LINEA_SINGOLA = "─" * LUNGHEZZA_LINEA

    print("=====================================================")
    print("   AVVIO FASE TUNING & BASELINE SPECULARE AL CLUSTER ")
    print("=====================================================\n")

    # Variabili di configurazione
    RANDOM_SEED = 123
    TEST_SIZE = 0.2
    SAMPLE_FRACTION = 0.01
    CORRELATION_THRESHOLD = 0.05

    target_col = "Label"
    BOOT_CONFIG_PATH = os.path.join("./.local_storage", "config.json")
    OUTPUT_DIR = "./outputs_baseline"
    REAL_CONFIG_PATH = os.path.join(OUTPUT_DIR, "config_real.json")
    SYNTHETIC_CONFIG_PATH = os.path.join(OUTPUT_DIR, "config_synthetic.json")
    dataset_type = "real"
    user_tree_type = "classifier"
    
    sys_cfg = SystemConfig()
    print(f" • Ambiente infrastrutturale rilevato: {sys_cfg.env.upper()}")
    
    # ---------------------------------------------------------
    # FASE 1: ETL CON CAMPIONAMENTO PROBABILISTICO 
    # ---------------------------------------------------------
    print(">>> FASE 1: Estrazione e Preprocessing Dati")
    
    if os.path.exists(BOOT_CONFIG_PATH):
        with open(BOOT_CONFIG_PATH, "r") as f:
            try:
                tmp_cfg = json.load(f)
                dataset_type = tmp_cfg.get("dataset_type", "real")
                user_tree_type = tmp_cfg.get("tree_type", "classifier")
                print(f" [INFO] Configurazione di boot letta con successo da '{BOOT_CONFIG_PATH}'")
            except Exception as e:
                print(f" [ATTENZIONE] Errore nel parsing di {BOOT_CONFIG_PATH}: {e}")
                pass
    else:
        print(f" [INFO] Nessun file di boot trovato in '{BOOT_CONFIG_PATH}'. Scalo sul dataset reale di default.")

    if dataset_type == "synthetic":
        if not os.path.exists(REAL_CONFIG_PATH):
            raise FileNotFoundError(
                f"Per eseguire la baseline sintetica serve aver già eseguito il tuning sul "
                f"dataset reale: '{REAL_CONFIG_PATH}' non trovato. Eseguire prima la baseline "
                f"sul dataset reale (opzione 1)."
            )
        else:
            print(f" [INFO] '{REAL_CONFIG_PATH}' non trovato. Uso iperparametri di fallback di default.")
            best_hp_reale = {
                "n_estimators": 10,
                "max_depth": 10,
                "min_samples_split": 2,
                "bootstrap": True,
                "max_samples": 1.0
            }
        with open(REAL_CONFIG_PATH, "r") as f:
            real_config = json.load(f)
        best_hp_reale = real_config["hyperparameters"]
        print(f" [INFO] Iperparametri di riferimento caricati dal tuning sul reale: '{REAL_CONFIG_PATH}'")

        if os.path.exists(SYNTHETIC_CONFIG_PATH):
            with open(SYNTHETIC_CONFIG_PATH, "r") as f:
                try:
                    tmp_cfg = json.load(f)
                    print(f" [INFO] Configurazione sintetica letta con successo da '{SYNTHETIC_CONFIG_PATH}'")
                except Exception as e:
                    tmp_cfg = {}
                    print(f" [ATTENZIONE] Errore nel parsing di {SYNTHETIC_CONFIG_PATH}: {e}")
        else:
            tmp_cfg = {}
        
        target_col = "Target"
        n_samples = tmp_cfg.get("n_samples", 500000)
        n_features = tmp_cfg.get("n_features", 30)
        n_informative_reg = tmp_cfg.get("n_informative_reg", int(n_features * 0.5))
        noise = tmp_cfg.get("noise", 10.0)

        # Conserviamo TUTTI i parametri di generazione già presenti nel file
        # (inclusi quelli usati solo per la classificazione, es. n_informative,
        # n_redundant...) in modo da poterli riscrivere intatti a fine run,
        # dato che ora config_synthetic.json è l'unica fonte di verità sia
        # per l'input (ricetta dataset) sia per l'output (manifesto).
        dataset_gen_params = {
            k: v for k, v in tmp_cfg.items()
            if k in (
                "n_samples", "n_features", "n_informative", "n_redundant",
                "n_clusters_per_class", "flip_y", "weight",
                "n_informative_reg", "noise",
            )
        }

        task_str = "regression" if user_tree_type == "regressor" else "classification"
        print(f" • Tipo Dataset: Sintetico (Stress Test Task - {user_tree_type.upper()})")

        # Corretto il typo "ttask" -> "task"
        loader = SyntheticDataLoader(
            task=task_str,
            n_samples=n_samples,
            n_features=n_features,
            random_seed=RANDOM_SEED,
            target_column=target_col,
            n_informative_reg=n_informative_reg,
            noise=noise,
        )
        df_clean = loader.load()

        train_df, test_df = train_test_split(df_clean, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        io_time = 0.0 
        etl_time = 0.0
    else:
        data_folder = getattr(sys_cfg, "dataset_path", None)
        
        if not data_folder or not os.path.exists(data_folder) or data_folder == "./data":
            if os.path.exists("./dataset_cache"):
                data_folder = "./dataset_cache"
            else:
                data_folder = "./data"

        print(f" • Cartella sorgente identificata per dati reali: '{data_folder}'")
        print(f" • Tipo Dataset: Reale (Campionamento probabilistico 1%, Seed: {RANDOM_SEED})")

        io_start_time = time.perf_counter()
        loader = RawCSVDataLoader(data_url=data_folder, sample_fraction=SAMPLE_FRACTION, dataset_seed=RANDOM_SEED)
        df_raw = loader.load()
        io_time = time.perf_counter() - io_start_time
        print(f"[OK] Caricamento dati (I/O) completato in {io_time:.4f} secondi.")

        preprocess_start_time = time.perf_counter()
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
        fs = CICIDSFeatureSelector(target_column=target_col, correlation_threshold=CORRELATION_THRESHOLD)
        train_df = fs.fit_transform(train_df)
        test_df = fs.transform(test_df)
        dizionario_feature = fs.feature_summary_
        etl_time = time.perf_counter() - preprocess_start_time

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    if dataset_type == "synthetic":
        dizionario_feature = {"eliminate": [], "salvate": list(X_train.columns)}

    print(f" • Volume Train: {X_train.shape} | Volume Test: {X_test.shape}")
    print("-" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config_path_final = os.path.join(OUTPUT_DIR, "config_real.json")
    pickle_path_final = os.path.join(
        OUTPUT_DIR,
        f"baseline_random_forest_{user_tree_type}.pkl" if dataset_type == "synthetic" else "baseline_random_forest_completa.pkl"
    )

    if dataset_type == "real":
    # ---------------------------------------------------------
    # FASE 2: TUNING MULTI-METRICA (RandomizedSearch 5-Fold)
    # ---------------------------------------------------------
        print("\n>>> FASE 2: Esplorazione Spazio Iperparametri (Tuning 5-Fold)...")
        param_dist = [
            {
                'n_estimators': [10, 20, 30],
                'max_depth': [10, 25, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': [None, 'balanced'],
                'bootstrap': [True],
                'max_samples':[0.5,0.7,0.8,1.0]
            },
            {
                'n_estimators': [10, 20, 30],
                'max_depth': [10, 25, None],
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
        start_tuning = time.perf_counter()
        search.fit(X_train, y_train)
        tempo_tuning = time.perf_counter() - start_tuning
        best_params = search.best_params_
        best_index = search.best_index_
        print(f"[OK] Tuning completato. Iperparametri ottimali: {best_params}")

        # Configurazione e creazione cartella di output dedicata
        OUTPUT_DIR = "./outputs_baseline"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        config_path_final = os.path.join(OUTPUT_DIR, "config_real.json")
        pickle_path_final = os.path.join(OUTPUT_DIR, "baseline_random_forest_completa.pkl")

        # ---------------------------------------------------------
        # FASE 3: SCRITTURA MANIFESTO CONFIG_REAL.JSON
        # ---------------------------------------------------------
        best_booststrap = bool(best_params.get("bootstrap", True))
        config_data = {
            "mode": "distributed",
            "dataset_type": dataset_type,
            "dataset_path": data_folder if dataset_type == "real" else "synthetic",
            "feature_eliminata" : dizionario_feature["eliminate"],
            "feature_selezionate" : dizionario_feature["salvate"],
            "hyperparameters": {
                "n_estimators": int(best_params.get("n_estimators", 10)),
                "max_depth": best_params.get("max_depth") ,
                "min_samples_split": int(best_params.get("min_samples_split", 2)),
                "class_weight": best_params.get("class_weight", None),
                "bootstrap": best_booststrap,
                "max_samples": float(best_params.get("max_samples", 1.0)) if best_booststrap else 1.0,
                "tree_type": user_tree_type,
                "target_column": target_col,
                "random_state": int(RANDOM_SEED)
            }
        }
        
        with open(config_path_final, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print(f"[OK] Manifesto 'config_real.json' salvato correttamente in: '{config_path_final}'")

        cv_results_extracted = {
            'test_accuracy': [search.cv_results_[f'split{i}_test_accuracy'][best_index] for i in range(5)],
            'test_precision': [search.cv_results_[f'split{i}_test_precision'][best_index] for i in range(5)],
            'test_recall': [search.cv_results_[f'split{i}_test_recall'][best_index] for i in range(5)],
            'test_f1': [search.cv_results_[f'split{i}_test_f1'][best_index] for i in range(5)]
        }
        tempo_medio_fold_tuning = search.cv_results_['mean_fit_time'][best_index]
    
    else:
        print("\n>>> FASE 2: SALTATA — riuso iperparametri ottenuti dal tuning sul dataset reale")
        tempo_tuning = 0.0
        tempo_medio_fold_tuning = 0.0
        cv_results_extracted = None  

        if user_tree_type == "classifier":
            # Stesso algoritmo/task del tuning reale (classificazione binaria):
            # ereditare gli iperparametri è metodologicamente corretto.
            best_bootstrap = bool(best_hp_reale.get("bootstrap", True))
            hp_sintetici = {
                "n_estimators": int(best_hp_reale.get("n_estimators", 10)),
                "max_depth": best_hp_reale.get("max_depth"),
                "min_samples_split": int(best_hp_reale.get("min_samples_split", 2)),
                "bootstrap": best_bootstrap,
                "max_samples": float(best_hp_reale.get("max_samples", 1.0)) if best_bootstrap else 1.0,
            }
        else:
            # Task REGRESSOR: non ereditiamo gli iperparametri del classificatore
            # reale (algoritmo e distribuzione dei dati diversi). Si usa la
            # configurazione di riferimento fissa definita a inizio file.
            print(" [INFO] Task REGRESSOR: uso configurazione di riferimento fissa "
                  "(non ereditata dal tuning sul dataset reale).")
            hp_sintetici = dict(SYNTHETIC_REGRESSOR_REFERENCE_HP)

        config_data = {
            "mode": "distributed",
            "dataset_type": "synthetic",
            "dataset_path": "synthetic",
            **dataset_gen_params,
            "feature_eliminata": dizionario_feature["eliminate"],
            "feature_selezionate": dizionario_feature["salvate"],
            "hyperparameters": {
                **hp_sintetici,
                "tree_type": user_tree_type,
                "target_column": target_col,
                "random_state": int(RANDOM_SEED)
            }
        }

        config_path_synthetic = os.path.join(OUTPUT_DIR, "config_synthetic.json")
        with open(config_path_synthetic, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print(f"[OK] Manifesto sintetico ({user_tree_type}) salvato in: '{config_path_synthetic}'")

    # ---------------------------------------------------------
    # FASE 4: ADDESTRAMENTO FINALE & INFERENZA LOCALE
    # ---------------------------------------------------------
    print("\n>>> FASE 4: Addestramento Finale Monolitico per estrazione T_seq...")
    hp = config_data["hyperparameters"]
    rf_kwargs = dict(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        min_samples_split=hp["min_samples_split"],
        bootstrap=hp["bootstrap"],
        n_jobs=1,
        random_state=RANDOM_SEED
    )
    
    if hp["bootstrap"]:
        rf_kwargs["max_samples"] = hp["max_samples"]

    if user_tree_type == "classifier":
        if dataset_type == "real":
            rf_kwargs["class_weight"] = hp["class_weight"]
        tree_clf = RandomForestClassifier(**rf_kwargs)
    else:
        tree_clf = RandomForestRegressor(**rf_kwargs)
    
    start_train_finale = time.perf_counter()
    tree_clf.fit(X_train, y_train)
    t_seq = time.perf_counter() - start_train_finale
    print(f"[OK] Fitting completato. T_seq ottenuto: {t_seq:.4f} secondi.")

    print("\n[LOCAL] Calcolo delle predizioni e latenza sul Test Set indipendente...")
    start_inferenza = time.perf_counter()
    local_preds = tree_clf.predict(X_test)
    
    # Gestione delle probabilità legata al TASK, non al dataset
    if user_tree_type == "classifier":
        local_proba = tree_clf.predict_proba(X_test)[:, 1]
    else:
        local_proba = None
    tempo_inferenza_totale = time.perf_counter() - start_inferenza

    # ---------------------------------------------------------
    # Valutazione, diramata per tipo di task (user_tree_type)
    # ---------------------------------------------------------
    if user_tree_type == "classifier":
        test_accuracy = np.mean(local_preds == y_test)
        test_precision = precision_score(y_test, local_preds, zero_division=0)
        test_recall = recall_score(y_test, local_preds, zero_division=0)
        test_f1 = f1_score(y_test, local_preds, zero_division=0)
        test_roc_auc = roc_auc_score(y_test, local_proba) if local_proba is not None else 0.0
        cm = confusion_matrix(y_test, local_preds)

        metriche_test = {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "roc_auc": test_roc_auc
        }
    else:
        test_mse = mean_squared_error(y_test, local_preds)
        test_rmse = float(np.sqrt(test_mse))
        test_mae = mean_absolute_error(y_test, local_preds)
        test_r2 = r2_score(y_test, local_preds)

        metriche_test = {
            "mse": test_mse,
            "rmse": test_rmse,
            "mae": test_mae,
            "r2": test_r2
        }

    metadata_pipeline = {
        "modello_addestrato": tree_clf,
        "features_mappate": list(X_train.columns),
        "baseline_tempi_locali": {
            "tempo_totale_cv": tempo_tuning,
            "tempo_medio_fold": tempo_medio_fold_tuning,
            "t_seq": t_seq,
            "tempo_inferenza_totale": tempo_inferenza_totale
        },
        "metriche_test": metriche_test
    }

    with open(pickle_path_final, "wb") as f:
        pickle.dump(metadata_pipeline, f)
    print(f"[OK] Pipeline locale (file .pkl) salvata in: '{pickle_path_final}'")

    # ---------------------------------------------------------
    # FASE 5: OUTPUT REPORT COMPLETO Condizionato sul Task
    # ---------------------------------------------------------
    print("\n" + DOPPIA_LINEA)
    print(f"{'REPORT ESTESO DI VALIDAZIONE E BENCHMARK':^{LUNGHEZZA_LINEA}}")
    print(DOPPIA_LINEA)

    if dataset_type == "real" and cv_results_extracted is not None:
        print(f"\n1. ANALISI DETTAGLIATA ITERAZIONE PER ITERAZIONE (CROSS-VALIDATION)")
        print(LINEA_SINGOLA)
        print(f"  {'Fold':<10} | {'Accuratezza':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}")
        print(LINEA_SINGOLA)
        for i in range(5):
            print(f"  Fold {i+1:02d}/05  | "
                  f"{cv_results_extracted['test_accuracy'][i]*100:10.2f}% | "
                  f"{cv_results_extracted['test_precision'][i]*100:10.2f}% | "
                  f"{cv_results_extracted['test_recall'][i]*100:10.2f}% | "
                  f"{cv_results_extracted['test_f1'][i]*100:10.2f}%")
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
    else:
        print(f"\n1-2. TUNING SALTATO — iperparametri riusati dal tuning sul dataset reale")
        print(LINEA_SINGOLA)
        print(f"  ▸ Iperparametri applicati: {hp}")

    print(f"\n3. PERFORMANCE REALI SUL TEST SET ({user_tree_type.upper()})")
    print(LINEA_SINGOLA)
    if user_tree_type == "classifier":
        print(f"  ▸ ACCURACY SUL TEST SET   : {metriche_test['accuracy'] * 100:6.2f}%")
        print(f"  ▸ PRECISION SUL TEST SET  : {metriche_test['precision'] * 100:6.2f}%")
        print(f"  ▸ RECALL SUL TEST SET     : {metriche_test['recall'] * 100:6.2f}%")
        print(f"  ▸ F1-SCORE SUL TEST SET   : {metriche_test['f1'] * 100:6.2f}%")
        print(f"  ▸ ROC-AUC SUL TEST SET    : {metriche_test['roc_auc']:6.4f}")
        print("\n  Matrice di Confusione sul Test Set:")
        for riga in cm:
            print(" " * 6 + " ".join(f"[{val:4d}]" for val in riga))
    else:
        print(f"  ▸ MSE SUL TEST SET   : {metriche_test['mse']:.4f}")
        print(f"  ▸ RMSE SUL TEST SET  : {metriche_test['rmse']:.4f}")
        print(f"  ▸ MAE SUL TEST SET   : {metriche_test['mae']:.4f}")
        print(f"  ▸ R² SUL TEST SET    : {metriche_test['r2']:.4f}")

    print(f"\n4. DIAGNOSTICA TEMPORALE E PROFILAZIONE HARDWARE")
    print(LINEA_SINGOLA)
    print(f"  • Tempo Totale di Cross-Validation     : {tempo_tuning:8.4f} s")
    print(f"  • Tempo Medio per Singolo Fold (CV)    : {tempo_medio_fold_tuning:8.4f} s")
    print(f"  • Tempo di Caricamento Dati (I/O)      : {io_time:8.4f} s")
    print(f"  • Tempo di Trasformazione (Process)    : {etl_time:8.4f} s")
    print(f"  • Tempo Totale di Addestramento (Training Set)  : {t_seq:8.4f} s")
    print(f"  • Tempo Totale di Inferenza (Testing Set) : {tempo_inferenza_totale:8.4f} s")

if __name__ == "__main__":
    run_baseline()