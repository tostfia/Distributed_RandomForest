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
    "n_estimators": 40,
    "max_depth": None,
    "min_samples_split": 2,
    # Esplicito perché il default sklearn per RandomForestRegressor è 1.0
    # (usa TUTTE le feature ad ogni split, cioè bagging puro): senza questo
    # override anche il regressore "di riferimento" non sarebbe un vero
    # Random Forest. 1/3 è il valore storicamente suggerito da Breiman (2001).
    "max_features": 1 / 3,
    "criterion": "squared_error",
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
    # Default storico: partizionamento IID, invariato. Sovrascritti sotto se il
    # boot config specifica una strategia non-IID per l'esperimento federato.
    partition_strategy = "dirichlet"
    federated_alpha = 0.5
    
    sys_cfg = SystemConfig()
    print(f" • Ambiente infrastrutturale rilevato: {sys_cfg.env.upper()}")
    
    # ---------------------------------------------------------
    # FASE 1: ETL CON CAMPIONAMENTO PROBABILISTICO 
    # ---------------------------------------------------------
    print(">>> FASE 1: Estrazione e Preprocessing Dati")
    
    if os.path.exists(BOOT_CONFIG_PATH):
        with open(BOOT_CONFIG_PATH, "r") as f:
            try:
                raw_state = json.load(f)
                if not isinstance(raw_state, dict):
                    raise ValueError("Il contenuto del file di stato locale non è un oggetto JSON valido.")

                if "baseline_boot" in raw_state:
                    # Nuovo formato strutturato: {"baseline_boot": {...}, "last_training_request": {...}}
                    boot_cfg = raw_state["baseline_boot"]
                elif "dataset_type" in raw_state and "hyperparameters" not in raw_state:
                    # Retrocompatibilità: vecchio formato piatto scritto direttamente come boot config.
                    boot_cfg = raw_state
                    print(f" [INFO] '{BOOT_CONFIG_PATH}' è nel formato precedente (piatto). Letto comunque per retrocompatibilità.")
                else:
                    # Il file esiste ma contiene solo (o principalmente) una last_training_request:
                    # non c'è una boot config valida, si scala sui default.
                    boot_cfg = {}
                    print(f" [INFO] '{BOOT_CONFIG_PATH}' non contiene una sezione 'baseline_boot' valida. Uso i default.")

                dataset_type = boot_cfg.get("dataset_type", "real")
                user_tree_type = boot_cfg.get("tree_type", "classifier")
                # Iperparametro dell'ESPERIMENTO (partizionamento tra worker federati),
                # non del modello: registrato qui nel manifesto così che
                # provision_local_shards.py / provision_federated_shards.py possano
                # essere lanciati con la stessa strategia usata per generare la
                # baseline, invece di doverla ripetere a mano.
                partition_strategy = boot_cfg.get("partition_strategy", "iid")
                federated_alpha = boot_cfg.get("alpha", 0.5)
                if boot_cfg:
                    print(f" [INFO] Configurazione di boot letta con successo da '{BOOT_CONFIG_PATH}'")
            except Exception as e:
                print(f" [ATTENZIONE] Errore nel parsing di {BOOT_CONFIG_PATH}: {e}")
                pass
    else:
        print(f" [INFO] Nessun file di boot trovato in '{BOOT_CONFIG_PATH}'. Scalo sul dataset reale di default.")

    if dataset_type == "synthetic":
        # Il tuning sul dataset reale serve SOLO al task sintetico di
        # CLASSIFICAZIONE (stesso algoritmo e stesso tipo di problema, quindi
        # ereditarne gli iperparametri è metodologicamente difendibile). Per il
        # REGRESSOR non viene mai usato — si va su SYNTHETIC_REGRESSOR_REFERENCE_HP
        # — quindi pretenderlo bloccava la baseline di regressione senza motivo.
        #
        # La versione precedente sollevava FileNotFoundError se il file NON
        # esisteva e, nel ramo 'else' (cioè quando ESISTEVA), stampava
        # "non trovato" assegnando un fallback che veniva comunque sovrascritto
        # due righe dopo: messaggio fuorviante e codice irraggiungibile.
        best_hp_reale = None
        if user_tree_type == "classifier":
            if not os.path.exists(REAL_CONFIG_PATH):
                raise FileNotFoundError(
                    f"Per eseguire la baseline sintetica di CLASSIFICAZIONE serve aver già "
                    f"eseguito il tuning sul dataset reale: '{REAL_CONFIG_PATH}' non trovato. "
                    f"Eseguire prima la baseline sul dataset reale (opzione 1)."
                )
            with open(REAL_CONFIG_PATH, "r") as f:
                real_config = json.load(f)
            best_hp_reale = real_config["hyperparameters"]
            print(f" [INFO] Iperparametri di riferimento caricati dal tuning sul reale: '{REAL_CONFIG_PATH}'")
        else:
            print(" [INFO] Task REGRESSOR: il tuning sul dataset reale non è richiesto "
                  "(si usa la configurazione di riferimento fissa dichiarata a inizio file).")

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
        
        # Stessa convenzione di CentralizedOrchestrator._prepare_data
        # ("Target" per la regressione, "Label" per la classificazione). Prima
        # era fissa a "Target" anche in classificazione, quindi baseline e
        # cluster producevano CSV con la colonna target chiamata diversamente.
        target_col = "Target" if user_tree_type == "regressor" else "Label"

        # Default allineato a SyntheticDataLoader (n_samples=300000). Prima qui
        # il default era 500000: in assenza di un manifesto la baseline generava
        # un dataset 1.67x più grande di quello del cluster, e i tempi di
        # addestramento non erano confrontabili.
        n_samples = tmp_cfg.get("n_samples", 300000)
        n_features = tmp_cfg.get("n_features", 30)
        n_informative_reg = tmp_cfg.get("n_informative_reg", int(n_features * 0.5))
        noise = tmp_cfg.get("noise", 10.0)

        # Registriamo i parametri EFFETTIVAMENTE usati per generare il dataset,
        # non quelli riletti dal file di input. Prima il dizionario veniva
        # costruito filtrando 'tmp_cfg': al primo run il manifesto non esiste,
        # tmp_cfg è vuoto e quindi n_samples/n_features non venivano MAI
        # persistiti. Il manifesto restava privo della ricetta, SyntheticDataLoader
        # continuava a usare i propri default e le due parti non convergevano
        # mai su una configurazione comune.
        # I parametri specifici della classificazione vengono invece conservati
        # da tmp_cfg, perché in quel ramo è il loader a risolverli dal manifesto.
        dataset_gen_params = {
            k: v for k, v in tmp_cfg.items()
            if k in ("n_informative", "n_redundant", "n_clusters_per_class", "flip_y", "weight")
        }
        dataset_gen_params.update({
            "n_samples": n_samples,
            "n_features": n_features,
            "target_column": target_col,
            "random_seed": RANDOM_SEED,
        })
        if user_tree_type == "regressor":
            dataset_gen_params["n_informative_reg"] = n_informative_reg
            dataset_gen_params["noise"] = noise

        task_str = "regression" if user_tree_type == "regressor" else "classification"
        print(f" • Tipo Dataset: Sintetico (Stress Test Task - {user_tree_type.upper()})")

        loader = SyntheticDataLoader(
            task=task_str,
            n_samples=n_samples,
            n_features=n_features,
            random_seed=RANDOM_SEED,
            target_column=target_col,
            n_informative_reg=n_informative_reg,
            noise=noise,
        )
        # Generazione e split cronometrati come I/O + ETL. Prima erano azzerati
        # (io_time = etl_time = 0.0): la baseline sintetica dichiarava zero costo
        # di preparazione dati mentre il lato distribuito conteggia l'intero
        # _prepare_data (30-40s su AWS per via di S3), quindi un confronto sui
        # tempi TOTALI risultava sistematicamente sfavorevole al cluster.
        io_start_time = time.perf_counter()
        df_clean = loader.load()
        io_time = time.perf_counter() - io_start_time
        print(f"[OK] Generazione dataset sintetico completata in {io_time:.4f} secondi.")

        preprocess_start_time = time.perf_counter()
        if user_tree_type == "regressor":
            train_df, test_df = train_test_split(df_clean, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        else:
            # Split STRATIFICATO come in _prepare_data: sul sintetico di
            # classificazione le classi sono sbilanciate (weights [0.9, 0.1]),
            # quindi uno split non stratificato darebbe alla baseline una
            # ripartizione train/test diversa da quella vista dal cluster.
            train_df, test_df = StratifiedDataSplitter(
                target_column=target_col, test_size=TEST_SIZE, random_state=RANDOM_SEED
            ).split(df_clean)
        etl_time = time.perf_counter() - preprocess_start_time
    else:
        # Cartella sorgente del dataset REALE.
        #
        # La versione precedente era:
        #     data_folder = getattr(sys_cfg, "dataset_path", None)
        #     if not data_folder or not os.path.exists(data_folder) or data_folder == "./data":
        #         data_folder = "./dataset_cache" if os.path.exists("./dataset_cache") else "./data"
        # con due difetti indipendenti:
        #
        # 1) SystemConfig non espone alcun attributo 'dataset_path' (vedi
        #    config.py: definisce solo mode, env, aws_region, le due code SQS e
        #    s3_bucket_name). Quindi il getattr restituiva SEMPRE None, la prima
        #    condizione era SEMPRE vera e nessun percorso configurato veniva mai
        #    onorato: la configurazione dava l'illusione di essere letta.
        #
        # 2) Il fallback finale su './data' faceva puntare RawCSVDataLoader a
        #    una cartella che poteva contenere CSV estranei (è lì che si trovava
        #    'sintetic_data.csv', un dataset SINTETICO): sarebbero stati
        #    ingeriti e processati dal CICIDSPreprocessor come se fossero
        #    traffico di rete reale, corrompendo la baseline in silenzio.
        #
        # Ora il percorso è configurabile davvero, tramite DATASET_LOCAL_PATH,
        # e in assenza della cartella si fallisce subito con un errore
        # esplicito invece di ripiegare su una directory arbitraria.
        data_folder = os.environ.get("DATASET_LOCAL_PATH", "./dataset_cache")
        if not os.path.exists(data_folder):
            raise FileNotFoundError(
                f"Cartella del dataset reale non trovata: '{data_folder}'. "
                f"Posiziona lì i CSV del CICIDS, oppure indica un percorso diverso "
                f"con la variabile d'ambiente DATASET_LOCAL_PATH. "
                f"(Nessun fallback automatico: leggere CSV da una cartella non "
                f"prevista produrrebbe una baseline sbagliata senza segnalarlo.)"
            )

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
                'max_features': ['sqrt'],
                'criterion': ['gini', 'entropy'],
                'class_weight': [None, 'balanced'],
                'bootstrap': [True],
                'max_samples':[0.5,0.7,0.8,1.0]
            },
            {
                'n_estimators': [10, 20, 30],
                'max_depth': [10, 25, None],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt'],
                'criterion': ['gini', 'entropy'],
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
            "correlation_threshold": CORRELATION_THRESHOLD,
            "feature_eliminata" : dizionario_feature["eliminate"],
            "feature_selezionate" : dizionario_feature["salvate"],
            # Iperparametro dell'ESPERIMENTO federato (non del modello): tenuto
            # separato da "hyperparameters" perché descrive come i dati vengono
            # ripartiti tra i worker, non l'algoritmo di training. Va passato
            # tal quale a provision_local_shards.py / provision_federated_shards.py.
            "federated_partitioning": {
                "strategy": partition_strategy,
                "alpha": federated_alpha if partition_strategy == "dirichlet" else None,
            },
            "hyperparameters": {
                "n_estimators": int(best_params.get("n_estimators", 10)),
                "max_depth": best_params.get("max_depth") ,
                "min_samples_split": int(best_params.get("min_samples_split", 2)),
                "max_features": best_params.get("max_features", "sqrt"),
                "criterion": best_params.get("criterion", "gini"),
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
                "max_features": best_hp_reale.get("max_features", "sqrt"),
                "criterion": best_hp_reale.get("criterion", "gini"),
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
            "federated_partitioning": {
                "strategy": partition_strategy,
                "alpha": federated_alpha if partition_strategy == "dirichlet" else None,
            },
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

        # 'config_synthetic.json' è il manifesto ATTIVO: è quello letto sia da
        # SyntheticDataLoader (ricetta del dataset) sia dal client
        # (load_hyperparameters_from_config). Essendo unico, un run di
        # classificazione sovrascrive quello di regressione e viceversa — ma la
        # traccia richiede ENTRAMBI i task. Ne salviamo quindi anche una copia
        # per-task, così i due esperimenti restano documentati e ricostruibili
        # (il .pkl del modello era già differenziato per tree_type, il manifesto no).
        config_path_synthetic = os.path.join(OUTPUT_DIR, "config_synthetic.json")
        with open(config_path_synthetic, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print(f"[OK] Manifesto sintetico ATTIVO ({user_tree_type}) salvato in: '{config_path_synthetic}'")

        config_path_synthetic_task = os.path.join(OUTPUT_DIR, f"config_synthetic_{user_tree_type}.json")
        with open(config_path_synthetic_task, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print(f"[OK] Copia per-task archiviata in: '{config_path_synthetic_task}'")
        print(f"[NOTA] Il cluster legge SEMPRE '{config_path_synthetic}': per passare all'altro "
              f"task sintetico va rilanciata la baseline, non basta avere la copia per-task.")

    # ---------------------------------------------------------
    # FASE 4: ADDESTRAMENTO FINALE & INFERENZA LOCALE
    # ---------------------------------------------------------
    print("\n>>> FASE 4: Addestramento Finale Monolitico per estrazione T_seq...")
    hp = config_data["hyperparameters"]
    rf_kwargs = dict(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        min_samples_split=hp["min_samples_split"],
        # Esplicito e letto dal manifesto invece di affidarsi al default sklearn
        # (che per il Regressor è 1.0 = nessun subsampling delle feature, cioè
        # bagging puro, non vero Random Forest).
        max_features=hp.get("max_features", "sqrt" if user_tree_type == "classifier" else 1 / 3),
        bootstrap=hp["bootstrap"],
        n_jobs=1,
        random_state=RANDOM_SEED
    )
    
    if hp["bootstrap"]:
        rf_kwargs["max_samples"] = hp["max_samples"]

    if hp.get("criterion"):
        rf_kwargs["criterion"] = hp["criterion"]

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

    # ---------------------------------------------------------
    # SECONDA BASELINE: stessa macchina, TUTTI i core (n_jobs=-1)
    #
    # Perché serve: T_seq è monocore (n_jobs=1), mentre ogni worker del cluster
    # addestra i propri alberi con un Pool multiprocesso (BaseWorker:
    # allocated_cores = cpu_count-1 su AWS). Se in relazione si presenta
    # T_seq / T_distribuito come "speedup della distribuzione", quel numero
    # include anche il guadagno del semplice multicore locale, che con
    # l'architettura distribuita non c'entra nulla.
    #
    # Con entrambi i riferimenti l'analisi diventa onesta e più ricca:
    #  • T_seq          -> speedup ed efficienza confrontabili con la teoria;
    #  • T_1node_par    -> "quanto guadagno DAVVERO distribuendo, rispetto a
    #                       usare al meglio una macchina sola?"
    # ---------------------------------------------------------
    cpu_disponibili = os.cpu_count() or 1
    print(f"\n>>> FASE 4b: Baseline su singola macchina MULTICORE ({cpu_disponibili} core logici, n_jobs=-1)...")
    rf_kwargs_par = dict(rf_kwargs)
    rf_kwargs_par["n_jobs"] = -1
    if user_tree_type == "classifier":
        tree_clf_par = RandomForestClassifier(**rf_kwargs_par)
    else:
        tree_clf_par = RandomForestRegressor(**rf_kwargs_par)

    start_train_par = time.perf_counter()
    tree_clf_par.fit(X_train, y_train)
    t_1node_parallel = time.perf_counter() - start_train_par
    speedup_multicore = (t_seq / t_1node_parallel) if t_1node_parallel > 0 else 1.0
    print(f"[OK] Fitting multicore completato: {t_1node_parallel:.4f} s "
          f"(speedup del solo multicore locale: {speedup_multicore:.2f}x)")
    # Il modello usato per le metriche resta quello sequenziale: n_jobs cambia
    # solo COME viene calcolato il fit, non il risultato (stesso random_state),
    # quindi tree_clf_par serve unicamente da riferimento temporale.

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
        # Scomposizione completa dei tempi: senza io_time/etl_time non è
        # possibile confrontare correttamente con il cluster, il cui tempo
        # totale include sempre la preparazione dati (vedi
        # CentralizedOrchestrator.last_etl_seconds).
        "baseline_tempi_locali": {
            "tempo_totale_cv": tempo_tuning,
            "tempo_medio_fold": tempo_medio_fold_tuning,
            "io_time": io_time,
            "etl_time": etl_time,
            "t_seq": t_seq,
            "t_1node_parallel": t_1node_parallel,
            "speedup_multicore_locale": speedup_multicore,
            "cpu_count": cpu_disponibili,
            "tempo_inferenza_totale": tempo_inferenza_totale
        },
        # Dimensioni effettive: servono a verificare a colpo d'occhio che
        # baseline e cluster abbiano lavorato sullo stesso volume di dati.
        "dataset_shape": {
            "train": list(X_train.shape),
            "test": list(X_test.shape),
            "dataset_type": dataset_type,
            "tree_type": user_tree_type,
        },
        "iperparametri_usati": dict(hp),
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
    print(f"  • T_seq  - Addestramento MONOCORE (n_jobs=1)   : {t_seq:8.4f} s")
    print(f"  • T_1node - Addestramento MULTICORE (n_jobs=-1): {t_1node_parallel:8.4f} s  "
          f"[{cpu_disponibili} core, speedup locale {speedup_multicore:.2f}x]")
    print(f"  • Tempo Totale di Inferenza (Testing Set) : {tempo_inferenza_totale:8.4f} s")
    print(f"  • Volume dati: train={X_train.shape}  test={X_test.shape}")

    print(f"\n5. COME CONFRONTARE QUESTI NUMERI COL CLUSTER")
    print(LINEA_SINGOLA)
    print("  ▸ Confronta il tempo di ADDESTRAMENTO del cluster al NETTO dell'ETL")
    print("    (CentralizedOrchestrator.last_etl_seconds) contro T_seq / T_1node:")
    print("    il totale del cluster include sempre la preparazione dati, la baseline no.")
    print("  ▸ Usa T_seq per speedup ed efficienza 'da manuale', T_1node per rispondere")
    print("    alla domanda pratica 'conviene distribuire invece di usare una macchina sola?'.")
    print("  ▸ Verifica che 'Volume dati' qui sopra coincida con lo shape stampato dal cluster:")
    print("    se differiscono, il confronto NON è valido (controlla config_synthetic.json).")

if __name__ == "__main__":
    run_baseline()