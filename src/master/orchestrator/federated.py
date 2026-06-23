import pickle
import os
import time
import rpyc
import traceback
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.extmath import weighted_mode
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from src.shared.utilities.datasplitter import StratifiedDataSplitter
from src.shared.config import SystemConfig
from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.factory import DatasetDAOFactory
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.featureselection import CICIDSFeatureSelector


class FederatedOrchestrator(BaseOrchestrator):
    def __init__(self):
        # 1. Recuperiamo la configurazione dal file .env tramite SystemConfig
        self.cfg = SystemConfig()
        
        # Recuperiamo il Process ID per distinguere le repliche nei log
        pid = os.getpid()
        
        # Inizializziamo la classe base in ascolto sulla coda federata
        super().__init__(
            orchestrator_name=f"Orchestrator-Federato-{pid}",
            queue_name="federated_queue"
        )
        self.current_job_id = None
        self.worker_shards_paths = {}  # Mappa per tracciare i path dei dataset partizionati per ciascun worker

    def _resolve_dataset_type(self, payload: dict) -> str:
        dataset_type = payload.get("dataset_type")
        if dataset_type:
            return str(dataset_type).strip().lower()
        return "real"

    def _prepare_data_federated(self, payload: dict, base_seed: int, worker_names: list):
        """
        Scarica il dataset dall'URL, lo binarizza, lo splitta in 80/20 (ordine esatto Colab), 
        applica il preprocessing e la feature selection separatamente, e infine lo frammenta in N shard.
        """
        self.current_job_id = payload.get("job_id", "unknown_job")
        dataset_path = payload.get("dataset_path")
        dataset_type = self._resolve_dataset_type(payload)
        target_col = "Label"
        num_workers = len(worker_names)

        print(f"\n[{self.orchestrator_name}] Avvio ETL FEDERATO. Tipo dataset: {dataset_type} (Ordine Speculare a Colab)")

        # --- ESTRAZIONE (Se Sintetico la partizione nativa avviene a livello worker) ---
        if dataset_type == "synthetic":
            print(f"[{self.orchestrator_name}] Dataset sintetico rilevato. La generazione avverrà nativamente sui nodi.")
            for i, name in enumerate(worker_names):
                self.worker_shards_paths[name] = f"NATIVE_PARTITIONED||{i}"
            return

        # --- ESTRAZIONE DATASET REALE (Da URL) ---
        if not dataset_path: 
            raise ValueError("dataset_path (URL) mancante nel payload per la modalità reale.")
        
        loader = RawCSVDataLoader(data_url=dataset_path, sample_fraction=0.01, dataset_seed=base_seed)
        df_raw = loader.load()

        # Istanziamo il nuovo preprocessor modificato e lo splitter
        preprocessor = CICIDSPreprocessor(target_column=target_col)
        splitter = StratifiedDataSplitter(target_column=target_col, test_size=0.2, random_state=base_seed)

        # ─── FASE 1: BINARIZZAZIONE SUL DATO INTERO (Previene il crash delle classi rare) ───
        df_binarized = preprocessor.binarize_target(df_raw)

        # ─── FASE 2: SPLIT STRATIFICATO SU DATO BINARIZZATO (Esatto ordine Colab) ───
        print(f"[{self.orchestrator_name}] Esecuzione Split Stratificato...")
        train_df, test_df = splitter.split(df_binarized)

        # ─── FASE 3 & 4: PREPROCESAMENTO INDIPENDENTE SUI DUE PEZZI (Metadata + NaN/inf) ───
        print(f"\n[{self.orchestrator_name}] === PREPROCESSING FEDERATO SUL TRAIN SET ===")
        train_df = preprocessor.process(train_df)
        
        print(f"\n[{self.orchestrator_name}] === PREPROCESSING FEDERATO SUL TEST SET ===")
        test_df = preprocessor.process(test_df)

        # ─── FASE 5: FEATURE SELECTION CON FIT/TRANSFORM (Previene Data Leakage) ───
        fs = CICIDSFeatureSelector(target_column=target_col, correlation_threshold=0.05)
        train_df = fs.fit_transform(train_df)
        test_df = fs.transform(test_df)

        # ─── FASE 6: PARTIZIONAMENTO E DISTRIBUZIONE SULLE CACHE DEI WORKER VIA DAO ───
        print(f"\n[{self.orchestrator_name}] Partizionamento dei DataFrame in {num_workers} shard bilanciati...")
        
        # Divisione matematica delle righe dei due set pronti e allineati
        train_shards = np.array_split(train_df, num_workers)
        test_shards = np.array_split(test_df, num_workers)
        
        dao = DatasetDAOFactory.get_dao(self.environment)

        for i, name in enumerate(worker_names):
            if self.environment == "aws":
                w_train_path = f"s3://my-cluster-datasets-bucket/federated_cache/{name}/train_{self.current_job_id}.csv"
                w_test_path = f"s3://my-cluster-datasets-bucket/federated_cache/{name}/test_{self.current_job_id}.csv"
            else:
                w_train_path = f"./.local_storage/{name}_cache/train_{self.current_job_id}.csv"
                w_test_path = f"./.local_storage/{name}_cache/test_{self.current_job_id}.csv"
                os.makedirs(os.path.dirname(w_train_path), exist_ok=True)

            # Salvataggio degli shard ripuliti e allineati al millimetro
            dao.save_dataset(path=w_train_path, df=train_shards[i])
            dao.save_dataset(path=w_test_path, df=test_shards[i])
            
            self.worker_shards_paths[name] = w_train_path
            print(f" [{self.orchestrator_name}] Shard inviato alla cache di {name} -> {w_train_path}")
            
    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int) -> bool:
        self.current_job_id = payload.get("job_id", "unknown_job")
        total_step_trees = target_alberi - start_alberi
        round_num = (start_alberi // total_step_trees) + 1
        
        print(f"\n [{self.orchestrator_name}] === AVVIO ROUND FEDERATO RPC {round_num} ===")

        # --- SCOPERTA DEI WORKER ---
        while True:
            available_workers = ServiceRegistry.get_available_workers(self.environment)
            if available_workers:
                break
            print(f" [{self.orchestrator_name}] Nessun worker federato disponibile. In Attesa...")
            time.sleep(10)

        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)

        # --- PREPARAZIONE E PARTIZIONAMENTO DATI (Se non ancora effettuato per questo Job) ---
        if not self.worker_shards_paths:
            self._prepare_data_federated(payload, seed, worker_names)

        trees_per_worker = total_step_trees // num_workers
        remainder = total_step_trees % num_workers

        hp = payload.get("hyperparameters", {})
        max_depth = hp.get("max_depth", None)
        tree_type = hp.get("tree_type", "classifier")

        all_trained_trees = []
        current_seed_offset = seed
        connessioni_attive = []

        # --- FUNZIONE INTERNA DI CHIAMATA RPC ADATTATA ---
        def _federated_rpc_call(w_name, w_info, n_trees, w_seed):
            print(f" [RPC -> {w_name}] Invio comando training FEDERATO locale ({n_trees} alberi)...")
            conn = rpyc.connect(w_info["host"], w_info["port"], config={'allow_pickle': True})
            connessioni_attive.append(conn)  
            try: 
                # Il worker riceve il path specifico della propria cache locale/S3
                federated_source_info = self.worker_shards_paths[w_name]

                remote_trees = conn.root.train_subset_forest(
                    source_info=federated_source_info, 
                    num_trees=n_trees, 
                    base_seed=w_seed, 
                    max_depth=max_depth
                )
                return remote_trees
            except Exception as e:
                print(f"   [ERRORE RPC FEDERATO] Connessione o calcolo fallito con {w_name}: {e}")
                raise e

        # --- DISTRIBUZIONE PARALLELA TRAMITE THREAD POOL ---
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_worker = {}
            for i, name in enumerate(worker_names):
                info = available_workers[name]
                allocated_trees = trees_per_worker + (1 if i < remainder else 0)

                if allocated_trees == 0:
                    continue

                f = executor.submit(_federated_rpc_call, name, info, allocated_trees, current_seed_offset)
                future_to_worker[f] = name
                current_seed_offset += allocated_trees

            for future in as_completed(future_to_worker):
                w_name = future_to_worker[future]
                try:
                    result_raw = future.result()
                    result_trees = pickle.loads(result_raw) if isinstance(result_raw, bytes) else result_raw
                    all_trained_trees.extend(result_trees)
                    print(f"   [RPC <- {w_name}] Ricevuti con successo {len(result_trees)} alberi federati.")
                except Exception as e:
                    print(f"   [ERRORE CRITICO FEDERATO] Il worker '{w_name}' ha fallito nel round: {e}")
                    raise e  

        print(f"[*] Pulizia risorse: chiusura di {len(connessioni_attive)} connessioni RPyC attive...")
        for conn in connessioni_attive:
            try: conn.close()
            except Exception: pass

        # --- RICOMPOSIZIONE E AGGREGAZIONE DEL MODELLO GLOBALE ---
        if len(all_trained_trees) > 0:
            if target_alberi == hp.get("n_estimators", 100):
                print(f"   [{self.orchestrator_name}] Ricomposizione foresta globale federata...")
                try:
                    n_features = all_trained_trees[0].n_features_in_

                    if tree_type == "classifier":
                        global_model = RandomForestClassifier(n_estimators=len(all_trained_trees))
                        global_model.classes_ = np.array([0, 1]) 
                        global_model.n_classes_ = 2
                    else:
                        global_model = RandomForestRegressor(n_estimators=len(all_trained_trees))
                    
                    global_model.estimators_ = all_trained_trees
                    global_model.n_features_in_ = n_features
                    global_model.n_outputs_ = 1
                    
                    TARGET_DIR = "./saved_models"
                    os.makedirs(TARGET_DIR, exist_ok=True)
                    model_path = os.path.join(TARGET_DIR, f"model_federated_{self.current_job_id}.pkl")
                    
                    with open(model_path, "wb") as f:
                        pickle.dump(global_model, f)
                    
                    print(f"   [{self.orchestrator_name}] [OK] Modello finale federato salvato in '{model_path}'.")
                except Exception as e:
                    print(f"   [ERRORE AGGREGAZIONE FEDERATA] Impossibile creare il modello finale: {e}")
                    traceback.print_exc()
                    return False
            return True

        return False
    
    def _execute_inference_step(self, payload: dict):
        job_id = payload.get("job_id")
        hp = payload.get("hyperparameters", {})
        tree_type = hp.get("tree_type", "classifier")

        print(f"\n[{self.orchestrator_name}] === AVVIO INFERENZA DISTRIBUITA FEDERATA ===")

        model_path = os.path.join("./saved_models", f"model_federated_{job_id}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modello globale federato non trovato in '{model_path}'.")

        with open(model_path, "rb") as f:
            global_model = pickle.load(f)
        
        all_trees = global_model.estimators_
        total_trees = len(all_trees)

        available_workers = ServiceRegistry.get_available_workers(self.environment)
        if not available_workers:
            raise RuntimeError("Nessun worker federato disponibile nel Service Registry per l'inferenza.")

        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)

        trees_per_worker = total_trees // num_workers
        remainder = total_trees % num_workers

        connessioni_attive = []
        # Qui terremo traccia delle metriche aggregate pesate
        acc_pesata_tot, prec_pesata_tot, rec_pesata_tot, f1_pesata_tot, mae_pesato_tot = 0, 0, 0, 0, 0
        total_global_samples = 0

        # --- FUNZIONE DI INF-CALL MODIFICATA CON CALCOLO DI METRICHE AVANZATE SUL NODO ---
        def _rpc_federated_inference_call(w_name, w_info, subset_trees_chunk):
            print(f" [RPC FED-INFERENZA -> {w_name}] Invio di {len(subset_trees_chunk)} alberi globali...")
            conn = rpyc.connect(w_info["host"], w_info["port"], config={'allow_pickle': True})
            connessioni_attive.append(conn)

            # Inoltro degli alberi lasciando i dati a None -> Forza il worker a caricare il proprio shard locale di test
            serialized_chunk = pickle.dumps(subset_trees_chunk)
            raw_response = conn.root.predict_subset_forest(serialized_chunk, None)
            sub_predictions = pickle.loads(raw_response) 

            if hasattr(conn.root, "get_local_y_test"):
                raw_y_test = conn.root.get_local_y_test()
            elif hasattr(conn.root, "y_test"):
                raw_y_test = conn.root.y_test
            else:
                raise AttributeError(f"Il worker {w_name} non espone y_test.")
            
            y_test_locale = pickle.loads(raw_y_test) if isinstance(raw_y_test, bytes) else np.array(raw_y_test)
            matrix_pred_locale = np.array(sub_predictions)
            size_test = len(y_test_locale)

            if tree_type == "classifier":
                uniform_weights = np.ones(matrix_pred_locale.shape[0])
                final_local_pred, _ = weighted_mode(matrix_pred_locale, uniform_weights, axis=0)
                final_local_pred = final_local_pred.ravel()
                
                # Calcolo metriche locali complete per il singolo worker
                acc = np.mean(final_local_pred == y_test_locale)
                prec = precision_score(y_test_locale, final_local_pred, zero_division=0)
                rec = recall_score(y_test_locale, final_local_pred, zero_division=0)
                f1_s = f1_score(y_test_locale, final_local_pred, zero_division=0)
                cm = confusion_matrix(y_test_locale, final_local_pred)
                rep = classification_report(y_test_locale, final_local_pred, zero_division=0)
                
                return w_name, "classifier", size_test, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1_s, "cm": cm, "report": rep}
            else:
                final_local_pred = np.mean(matrix_pred_locale, axis=0)
                mae = np.mean(np.abs(final_local_pred - y_test_locale))
                return w_name, "regressor", size_test, {"mae": mae}

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_worker = {}
            current_tree_idx = 0

            for i, name in enumerate(worker_names):
                allocated_trees = trees_per_worker + (1 if i < remainder else 0)
                if allocated_trees == 0:
                    continue

                chunk = all_trees[current_tree_idx : current_tree_idx + allocated_trees]
                current_tree_idx += allocated_trees

                f = executor.submit(_rpc_federated_inference_call, name, available_workers[name], chunk)
                future_to_worker[f] = name

            for future in as_completed(future_to_worker):
                w_name = future_to_worker[future]
                try:
                    name_w, m_type, size_test, res = future.result()
                    total_global_samples += size_test
                    
                    print(f"\n   [RPC <- {name_w}] Validazione completata sullo Shard Locale ({size_test} campioni).")
                    print("   " + "-"*60)
                    
                    if m_type == "classifier":
                        # Log del singolo nodo federato
                        print(f"         Accuracy Locale:      {res['accuracy'] * 100:.2f} %")
                        print(f"         Precision Locale:     {res['precision'] * 100:.2f} %")
                        print(f"         Recall Locale:        {res['recall'] * 100:.2f} %")
                        print(f"         F1-Score Locale:      {res['f1'] * 100:.2f} %")
                        print("         Matrice Confusione Locale:")
                        print(res['cm'])
                        
                        # Accumulo pesato per la metrica globale finale
                        acc_pesata_tot += res['accuracy'] * size_test
                        prec_pesata_tot += res['precision'] * size_test
                        rec_pesata_tot += res['recall'] * size_test
                        f1_pesata_tot += res['f1'] * size_test
                    else:
                        print(f"         MAE Locale:           {res['mae']:.4f}")
                        mae_pesato_tot += res['mae'] * size_test
        
                except Exception as e:
                    print(f"   [ERRORE CRITICO FED-INFERENZA] Fallimento computazione sul worker '{w_name}': {e}")
                    raise e

        for conn in connessioni_attive:
            try: conn.close()
            except Exception: pass

        print("\n" + "═" * 75)
        print(f"  VALUTAZIONE PERFORMANCE MODELLO FEDERATO GLOBALE (JOB: {job_id[:8]})")
        print("═" * 75)

        if tree_type == "classifier":
            # Calcolo delle medie globali pesate sulla dimension dei test set dei nodi
            global_accuracy = (acc_pesata_tot / total_global_samples) if total_global_samples > 0 else 0
            global_precision = (prec_pesata_tot / total_global_samples) if total_global_samples > 0 else 0
            global_recall = (rec_pesata_tot / total_global_samples) if total_global_samples > 0 else 0
            global_f1 = (f1_pesata_tot / total_global_samples) if total_global_samples > 0 else 0
            
            print(f"  Tipo di Modello:                        CLASSIFICATORE")
            print(f"  Testing Set Globale (Nodi Sommati):     {total_global_samples} campioni")
            print("-" * 75)
            print(f"  ACCURACY MEDIA PESATA SUI NODI:         {global_accuracy * 100:.2f} %")
            print(f"  PRECISION MEDIA PESATA SUI NODI:        {global_precision * 100:.2f} %")
            print(f"  RECALL MEDIA PESATA SUI NODI:           {global_recall * 100:.2f} %")
            print(f"  F1-SCORE MEDIA PESATA SUI NODI:         {global_f1 * 100:.2f} %")
        else:
            global_mae = (mae_pesato_tot / total_global_samples) if total_global_samples > 0 else 0
            print(f"  Tipo di Modello:                        REGRESSORE")
            print(f"  Testing Set Globale (Nodi Sommati):     {total_global_samples} campioni")
            print("-" * 75)
            print(f"  MAE MEDIO PESATO SUI NODI:              {global_mae:.4f}")

        print("═" * 75 + "\n")


if __name__ == "__main__":
    print("[BOOT] Avvio del nodo Orchestratore Federato...")
    orchestrator = FederatedOrchestrator()
    orchestrator.start()