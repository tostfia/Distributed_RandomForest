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
import src.shared.utilities.datasplitter
from src.shared.config import SystemConfig
from src.shared.factory import DatasetDAOFactory
from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.featureselection import CICIDSFeatureSelector


class CentralizedOrchestrator(BaseOrchestrator):
    def __init__(self):
        # 1. Recuperiamo la configurazione dal file .env
        self.cfg = SystemConfig()
        
        # Recuperiamo il Process ID per distinguere le repliche nei log
        pid = os.getpid()
        
        # Inizializziamo la classe base senza passare l'ambiente (lo legge da sé via config)
        super().__init__(
            orchestrator_name=f"Orchestrator-Centralizzato-{pid}",
            queue_name="centralized_queue"
        )
        self.current_job_id = None
        self.train_data_path = None
        self.test_data_path = None

    def _resolve_dataset_type(self, payload: dict) -> str:
        """Determina il tipo di dataset basandosi sul payload inviato dal Client."""
        dataset_type = payload.get("dataset_type")
        if dataset_type:
            return str(dataset_type).strip().lower()
        return "real"
    
    def _prepare_data(self, payload: dict, base_seed: int):
        self.current_job_id = payload.get("job_id", "unknown_job")
        dataset_path = payload.get("dataset_path")
        dataset_type = self._resolve_dataset_type(payload)
        target_col = "Label"

        print(f"\n[{self.orchestrator_name}] Avvio ETL. Tipo: {dataset_type} (Ordine Speculare a Colab)")

        # Inizializziamo lo splitter
        splitter = src.shared.utilities.datasplitter.StratifiedDataSplitter(
            target_column=target_col, test_size=0.2, random_state=base_seed
        )

        # --- ESTRAZIONE ---
        if dataset_type == "synthetic":
            loader = SyntheticDataLoader()
            train_df, test_df = splitter.split(loader.load())
        else:
            if not dataset_path: 
                raise ValueError("dataset_path mancante.")
            loader = RawCSVDataLoader(data_url=dataset_path, sample_fraction=0.01, dataset_seed=base_seed)
            df_raw = loader.load()
            
            # Istanziamo il nuovo preprocessor modificato
            preprocessor = CICIDSPreprocessor(target_column=target_col)

            # ─── FASE 1: BINARIZZAZIONE SUL DATO INTERO ───
            df_binarized = preprocessor.binarize_target(df_raw)
            
            # ─── FASE 2: SPLIT STRATIFICATO ADESSO SICURO ───
            print(f"[{self.orchestrator_name}] Esecuzione Split Stratificato...")
            train_df, test_df = splitter.split(df_binarized)

            # ─── FASE 3 & 4: PREPROCESAMENTO INDIPENDENTE (Metadata + NaN/inf) ───
            print(f"\n[{self.orchestrator_name}] === PREPROCESSING SUL TRAIN SET ===")
            train_df = preprocessor.process(train_df)
            
            print(f"\n[{self.orchestrator_name}] === PREPROCESSING SUL TEST SET ===")
            test_df = preprocessor.process(test_df)

        # --- FEATURE SELECTION (Solo Real) ---
        if dataset_type == "real":
            fs = CICIDSFeatureSelector(target_column=target_col, correlation_threshold=0.05)
            train_df = fs.fit_transform(train_df)
            test_df = fs.transform(test_df)

        # --- SALVATAGGIO COORDINATO DAI DAO ---
        if self.environment == "aws":
            self.train_data_path = f"s3://my-cluster-datasets-bucket/distributed_trains/shared_train_{self.current_job_id}.csv"
            self.test_data_path = f"s3://my-cluster-datasets-bucket/distributed_tests/shared_test_{self.current_job_id}.csv"
        else:
            self.train_data_path = f"./.local_storage/shared_train_{self.current_job_id}.csv"
            self.test_data_path = f"./.local_storage/shared_test_{self.current_job_id}.csv"
            
        print(f"\n[{self.orchestrator_name}] Delega salvataggio a DatasetDAOFactory...")
        try:
            dao = DatasetDAOFactory.get_dao(self.environment)
            dao.save_dataset(path=self.train_data_path, df=train_df)
            dao.save_dataset(path=self.test_data_path, df=test_df)
            print(f"[{self.orchestrator_name}] [OK] Dataset di Train e Test archiviati correttamente.")
        except Exception as e:
            raise IOError(f"[{self.orchestrator_name}] Errore critico nel salvataggio dei dataset tramite DAO: {e}")

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int):
        if self.train_data_path is None:
            self._prepare_data(payload, seed)
        
        total_step_trees = target_alberi - start_alberi
        print(f"\n [{self.orchestrator_name}] Distribuzione carico: {total_step_trees} alberi da generare...")

        while True:
            available_workers = ServiceRegistry.get_available_workers(self.environment)
            if available_workers:
                print(f"[{self.orchestrator_name}] Worker rilevati: {list(available_workers.keys())}. Procedo...")
                break
            
            print(f"[{self.orchestrator_name}] Nessun worker disponibile. In Attesa...")
            time.sleep(10)

        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)

        trees_per_worker = total_step_trees // num_workers
        remainder = total_step_trees % num_workers
        source_info = self.train_data_path 
        
        hp = payload.get("hyperparameters", {})
        max_depth = hp.get("max_depth", None)

        all_trained_trees = []
        current_seed_offset = seed
        connessioni_attive = []

        def _rpc_call(w_name, w_info, n_trees, w_seed):
            print(f" [RPC -> {w_name}] Invio richiesta per {n_trees} alberi su {w_info['host']}:{w_info['port']}...")
            conn = rpyc.connect(w_info["host"], w_info["port"], config={'allow_pickle': True,'sync_request_timeout': 600})
            connessioni_attive.append(conn)  
            try: 
                remote_trees = conn.root.train_subset_forest(source_info=source_info, num_trees=n_trees, base_seed=w_seed, max_depth=max_depth)
                return remote_trees
            except Exception as e:
                print(f"   [ERRORE RPC] Connessione fallita con {w_name}: {e}")
                raise e

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_worker = {}
            for i, name in enumerate(worker_names):
                info = available_workers[name]
                allocated_trees = trees_per_worker + (1 if i < remainder else 0)

                if allocated_trees == 0:
                    continue

                f = executor.submit(_rpc_call, name, info, allocated_trees, current_seed_offset)
                future_to_worker[f] = name
                current_seed_offset += allocated_trees

            for future in as_completed(future_to_worker):
                w_name = future_to_worker[future]
                try:
                    result_raw = future.result()
                    result_trees = pickle.loads(result_raw) if isinstance(result_raw, bytes) else result_raw
                    all_trained_trees.extend(result_trees)
                    print(f"   [RPC <- {w_name}] Ricevuti con successo {len(result_trees)} alberi.")
                except Exception as e:
                    print(f"   [ERRORE CRITICO] Il worker '{w_name}' ha fallito o si è disconnesso: {e}")
                    traceback.print_exc()  
                    raise e  

        print(f"[*] Pulizia risorse: chiusura di {len(connessioni_attive)} connessioni RPyC attive...")
        for conn in connessioni_attive:
            try: conn.close()
            except Exception: pass

        # --- RICOMPOSIZIONE E SALVATAGGIO ---
        if len(all_trained_trees) > 0:
            print(f"   [{self.orchestrator_name}] Ricomposizione foresta globale conforme a Scikit-Learn...")
            tree_type = hp.get("tree_type", "classifier")
            
            try:
                # Estraiamo il numero di feature di input dal primo albero valido per blindare il modello
                n_features = all_trained_trees[0].n_features_in_
                
                if tree_type == "classifier":
                    global_model = RandomForestClassifier(n_estimators=len(all_trained_trees))
                    global_model.classes_ = np.array([0, 1]) 
                    global_model.n_classes_ = 2
                else:
                    global_model = RandomForestRegressor(n_estimators=len(all_trained_trees))
                
                # Iniezione parametri strutturali per la piena compatibilità esterna
                global_model.estimators_ = all_trained_trees
                global_model.n_features_in_ = n_features
                global_model.n_outputs_ = 1
                
                TARGET_DIR = "./saved_models"
                os.makedirs(TARGET_DIR, exist_ok=True)
                model_path = os.path.join(TARGET_DIR, f"model_{self.current_job_id}.pkl")
                
                with open(model_path, "wb") as f:
                    pickle.dump(global_model, f)
                
                print(f"   [{self.orchestrator_name}] Modello {tree_type} integrato correttamente in '{model_path}'.")
                return True
                
            except Exception as e:
                print(f"   [ERRORE AGGREGAZIONE] Impossibile creare il modello {tree_type}: {e}")
                traceback.print_exc()
                return False

        print(f"   [{self.orchestrator_name}] Nessun albero ricevuto. Fallimento.")
        return False
    
    def _execute_inference_step(self, payload: dict):
        job_id = payload.get("job_id")
        hp = payload.get("hyperparameters", {})
        tree_type = hp.get("tree_type", "classifier")
        target_col = "Label"

        print(f"\n[{self.orchestrator_name}] === AVVIO INFERENZA DISTRIBUITA CENTRALIZZATA ===")

        inference_start_time = time.perf_counter()

        model_path = os.path.join("./saved_models", f"model_{job_id}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modello globale non trovato in '{model_path}'.")

        print(f"[{self.orchestrator_name}] Caricamento della foresta da {model_path}...")
        with open(model_path, "rb") as f:
            global_model = pickle.load(f)
        
        all_trees = global_model.estimators_
        total_trees = len(all_trees)
        print(f"[{self.orchestrator_name}] Foresta caricata. Numero totale di alberi: {total_trees}")

        # RECOVERY DEL PATH: Ricostruiamo il path se l'istanza è stata sostituita dal failover
        if self.test_data_path is None:
            if self.environment == "aws":
                self.test_data_path = f"s3://my-cluster-datasets-bucket/distributed_tests/shared_test_{job_id}.csv"
            else:
                # Sostituisci questa riga:
                self.test_data_path = f"./.local_storage/shared_test_{job_id}.csv"

        print(f"[{self.orchestrator_name}] Caricamento Testing Set persistito via DAO: {self.test_data_path}")
        dao = DatasetDAOFactory.get_dao(self.environment)
        test_df = dao.load_dataset(self.test_data_path)

        print(f"[{self.orchestrator_name}] Preparazione della matrice di test (Shape: {test_df.shape})...")
        X_test = test_df.drop(columns=[target_col]).to_numpy(dtype=np.float64)
        y_test = test_df[target_col].to_numpy()
        serialized_X_test = pickle.dumps(X_test)

        available_workers = ServiceRegistry.get_available_workers(self.environment)
        if not available_workers:
            raise RuntimeError("Nessun worker disponibile nel Service Registry per l'inferenza.")

        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        print(f"[{self.orchestrator_name}] Worker pronti per l'inferenza: {num_workers} -> {worker_names}")

        trees_per_worker = total_trees // num_workers
        remainder = total_trees % num_workers

        all_worker_predictions = []
        connessioni_attive = []

        def _rpc_inference_call(w_name, w_info, subset_trees_chunk):
            print(f" [RPC INFERENZA -> {w_name}] Invio di {len(subset_trees_chunk)} alberi...")
            conn = rpyc.connect(w_info["host"], w_info["port"], config={'allow_pickle': True,'sync_request_timeout': 600})
            connessioni_attive.append(conn)
            
            serialized_chunk = pickle.dumps(subset_trees_chunk)
            raw_response = conn.root.predict_subset_forest(serialized_chunk, serialized_X_test)
            return pickle.loads(raw_response)
        
        rpc_start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_worker = {}
            current_tree_idx = 0

            for i, name in enumerate(worker_names):
                allocated_trees = trees_per_worker + (1 if i < remainder else 0)
                if allocated_trees == 0:
                    continue

                chunk = all_trees[current_tree_idx : current_tree_idx + allocated_trees]
                current_tree_idx += allocated_trees

                f = executor.submit(_rpc_inference_call, name, available_workers[name], chunk)
                future_to_worker[f] = name

            for future in as_completed(future_to_worker):
                w_name = future_to_worker[future]
                try:
                    sub_predictions = future.result()
                    all_worker_predictions.extend(sub_predictions)
                    print(f"   [RPC INFERENZA <- {w_name}] Ricevute correttamente predizioni parziali.")
                except Exception as e:
                    print(f"   [ERRORE CRITICO INFERENZA] Il worker '{w_name}' ha fallito: {e}")
                    raise e

        print(f"[*] Chiusura di {len(connessioni_attive)} connessioni RPC attive...")
        for conn in connessioni_attive:
            try: conn.close()
            except Exception: pass

        rpc_inference_time = time.perf_counter() - rpc_start_time

        predictions_matrix = np.array(all_worker_predictions)
        print(f"[{self.orchestrator_name}] Matrice complessiva predizioni generata: {predictions_matrix.shape}")

        print("\n" + "═" * 75)
        print(f"  VALUTAZIONE PRESTAZIONI MODELLO DISTRIBUITO (JOB: {job_id[:8]})")
        print("═" * 75)

        total_inference_time = time.perf_counter() - inference_start_time

        print("═" * 75)
        print(f"  TEMPO TOTALE DI INFERENZA:              {total_inference_time:.4f} secondi")
        print("═" * 75 + "\n")
        print(f"  TEMPO INFERENZA DISTRIBUITA RPC:        {rpc_inference_time:.4f} secondi")

        if tree_type == "classifier":
            print("[DEBUG ORCHESTRATORE] Sto eseguendo il NUOVO codice con ones_like!")
            # Calcoliamo il voto di maggioranza esente da bug tramite weighted_mode ad un solo peso uniforme (1.0)
            uniform_weights = np.ones_like(predictions_matrix)
            final_predictions, _ = weighted_mode(predictions_matrix, uniform_weights, axis=0)
            final_predictions = final_predictions.ravel().astype(int)
            y_test = y_test.astype(int)
            
            # --- CALCOLO METRICHE DETTAGLIATE ALLINEATE A COLAB ---
            accuracy = np.mean(final_predictions == y_test)
            precision = precision_score(y_test, final_predictions, zero_division=0)
            recall = recall_score(y_test, final_predictions, zero_division=0)
            f1 = f1_score(y_test, final_predictions, zero_division=0)
            cm = confusion_matrix(y_test, final_predictions)
            
            print(f"  Tipo di Modello:                        CLASSIFICATORE")
            print(f"  Testing Set size:                       {X_test.shape[0]} campioni")
            print("-" * 75)
            print(f"  ACCURACY FINALE DISTRIBUITA:            {accuracy * 100:.2f} %")
            print(f"  PRECISION DISTRIBUITA:                  {precision * 100:.2f} %")
            print(f"  RECALL DISTRIBUITA:                     {recall * 100:.2f} %")
            print(f"  F1-SCORE DISTRIBUITO:                   {f1 * 100:.2f} %")
            print("-" * 75)
            print("  Matrice di Confusione:")
            print(cm)
            print("\n  Classification Report Completo:")
            print(classification_report(y_test, final_predictions, zero_division=0))
            
        else:
            final_predictions = np.mean(predictions_matrix, axis=0)
            mae = np.mean(np.abs(final_predictions - y_test))
            print(f"  Tipo di Modello:                        REGRESSORE")
            print(f"  Testing Set size:                       {X_test.shape[0]} campioni")
            print(f"  MAE FINALE DISTRIBUITO:                 {mae:.4f}")

        print("═" * 75 + "\n")


if __name__ == "__main__":
    print("[BOOT] Avvio del nodo Orchestratore Centralizzato...")
    orchestrator = CentralizedOrchestrator()
    orchestrator.start()