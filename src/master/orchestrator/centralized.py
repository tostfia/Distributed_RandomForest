import pickle
import os
import time
import rpyc
import queue
import threading
import traceback
from rpyc.utils.classic import obtain
import numpy as np
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

        print(f"\n[{self.orchestrator_name}] Avvio ETL. Tipo: {dataset_type}")

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

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int) -> bool:
        """
        Esegue lo step di addestramento distribuito centralizzato.
        Versione allineata e verificata con le firme di BaseWorker.
        """

        # 1. Preparazione dei dati (se non ancora pronti)
        if self.train_data_path is None:
            self._prepare_data(payload, seed)
        
        total_step_trees = target_alberi - start_alberi
        print(f"\n [{self.orchestrator_name}] Distribuzione carico: {total_step_trees} alberi da generare...")

        # 2. Recupero dei Worker disponibili dal ServiceRegistry
        while True:
            available_workers = ServiceRegistry.get_available_workers(self.environment)
            if available_workers:
                print(f"[{self.orchestrator_name}] Worker rilevati: {list(available_workers.keys())}. Procedo...")
                break
            
            print(f"[{self.orchestrator_name}] Nessun worker disponibile. In Attesa...")
            time.sleep(10)

        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        source_info = self.train_data_path 
        
        # Estrazione iperparametri dal payload
        hp = payload.get("hyperparameters", {})
        max_depth = hp.get("max_depth", None)
        tree_type = hp.get("tree_type", "classifier")

        # 3. CALCOLO DINAMICO DELLA DIMENSIONE DEL CHUNK
        CHUNK_SIZE = max(1, total_step_trees // (num_workers * 2))
        print(f"[{self.orchestrator_name}] Calcolo dinamico: {num_workers} worker rilevati -> CHUNK_SIZE impostata a {CHUNK_SIZE} alberi per task.")

        # 4. Configurazione della Coda di Sotto-Task locale
        task_queue = queue.Queue()
        sub_start = start_alberi
        task_id_counter = 0
        
        while sub_start < target_alberi:
            sub_end = min(sub_start + CHUNK_SIZE, target_alberi)
            # Ogni sotto-task associa un seed specifico calcolato sull'offset cumulativo
            task_queue.put((task_id_counter, sub_start, sub_end, seed + (sub_start - start_alberi)))
            task_id_counter += 1
            sub_start = sub_end

        all_trained_trees = []
        results_lock = threading.Lock()
        connessioni_attive = []
        connessioni_lock = threading.Lock()
        
        active_worker_names = list(worker_names)

        # 5. Definizione della funzione consumatrice per i thread
        def worker_thread_consumer(w_name):
            w_info = available_workers[w_name]
            worker_conn = None
            try:
                print(f" [RPC -> {w_name}] Apertura connessione su {w_info['host']}:{w_info['port']}...")
                worker_conn = rpyc.connect(
                    w_info["host"], 
                    w_info["port"], 
                    config={
                        'allow_pickle': True,
                        'sync_request_timeout': 600,
                        'keepalive': True
                    }
                )
                with connessioni_lock:
                    connessioni_attive.append(worker_conn)
                
                while True:
                    try:
                        task_id, start_t, end_t, chunk_seed = task_queue.get(timeout=1)
                    except queue.Empty:
                        break

                    quota_chunk = end_t - start_t
                    print(f"[{self.orchestrator_name}-Thread] Assegnazione Task {task_id} ({quota_chunk} alberi: {start_t}-{end_t}) a {w_name}")
                    
                    try:
                        
                        result_raw = worker_conn.root.train_subset_forest(
                            source_info=source_info,
                            num_trees=quota_chunk,       
                            base_seed=chunk_seed,    
                            max_depth=max_depth
                        )
                        
                        # Deserializzazione sicura dei byte trasmessi via rete
                        result_trees = pickle.loads(obtain(result_raw))
                        
                        with results_lock:
                            all_trained_trees.extend(result_trees)
                            
                        print(f"   [RPC <- {w_name}] Task {task_id} completato. Ricevuti {len(result_trees)} alberi.")
                        task_queue.task_done()
                        
                    except Exception as e:
                        print(f"   [ERRORE RPC] Fallimento o disconnessione del worker {w_name} durante il Task {task_id}: {e}")
                        
                        # Reinserimento immediato del chunk per la fault tolerance
                        task_queue.put((task_id, start_t, end_t, chunk_seed))
                        print(f"[{self.orchestrator_name}-Thread] Task {task_id} riaccodato per il failover.")
                        
                        with results_lock:
                            if w_name in active_worker_names:
                                active_worker_names.remove(w_name)
                        break  
                        
            except Exception as conn_err:
                print(f"   [ERRORE CRITICO] Impossibile connettersi a {w_name}: {conn_err}")
                with results_lock:
                    if w_name in active_worker_names:
                        active_worker_names.remove(w_name)
            finally:
                if worker_conn:
                    try:
                        worker_conn.close()
                        with connessioni_lock:
                            if worker_conn in connessioni_attive:
                                connessioni_attive.remove(worker_conn)
                    except:
                        pass

        # 6. Avvio dei thread
        threads = []
        for name in worker_names:
            t = threading.Thread(target=worker_thread_consumer, args=(name,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # 7. Monitoraggio fallimento totale dello step
        if not task_queue.empty() and len(active_worker_names) == 0:
            print(f"   [{self.orchestrator_name}] Tutti i worker sono crashati. SQS gestirà il failover macro.")
            raise RuntimeError("Sotto-sistema Fault Tolerance interrotto: Nessun worker disponibile rimasto.")

        # Chiusura pulita delle connessioni
        print(f"[*] Pulizia risorse: chiusura di {len(connessioni_attive)} connessioni RPyC residue...")
        with connessioni_lock:
            for conn in connessioni_attive:
                try: conn.close()
                except Exception: pass

        # 8. Ricomposizione della foresta globale
        if len(all_trained_trees) > 0:
            print(f"   [{self.orchestrator_name}] Ricomposizione foresta globale conforme a Scikit-Learn...")
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
                model_path = os.path.join(TARGET_DIR, f"model_{self.current_job_id}.pkl")
                
                with open(model_path, "wb") as f:
                    pickle.dump(global_model, f)
                
                print(f"   [{self.orchestrator_name}] Modello {tree_type} salvato con successo in '{model_path}'.")
                return True
                
            except Exception as e:
                print(f"   [ERRORE AGGREGAZIONE] Fallimento durante l'unione dei sotto-modelli: {e}")
                traceback.print_exc()
                return False

        print(f"   [{self.orchestrator_name}] Nessun albero collezionato.")
        return False
    
    def _execute_inference_step(self, payload: dict):
        """
        Esegue l'inferenza distribuita centralizzata in modalità Fault-Tolerant
        sfruttando una task queue concorrente per riallocare dinamicamente i blocchi in caso di crash.
        """
        job_id = payload.get("job_id")
        hp = payload.get("hyperparameters", {})
        tree_type = hp.get("tree_type", "classifier")
        target_col = "Label"

        print(f"\n[{self.orchestrator_name}] === AVVIO INFERENZA DISTRIBUITA CENTRALIZZATA FAULT-TOLERANT ===")
        inference_start_time = time.perf_counter()

        # 1. RISOLUZIONE DINAMICA FILE MODELLO (.pkl) E TESTING SET (.csv) IN BASE ALL'AMBIENTE
        if self.environment == "aws":
            model_path = f"s3://my-cluster-datasets-bucket/saved_models/model_{job_id}.pkl"
            self.test_data_path = f"s3://my-cluster-datasets-bucket/distributed_tests/shared_test_{job_id}.csv"
        else:
            model_path = os.path.join("./saved_models", f"model_{job_id}.pkl")
            self.test_data_path = f"./.local_storage/shared_test_{job_id}.csv"

        print(f"[{self.orchestrator_name}] [AUTO-RESOLVE] Risoluzione asset logici per il Job ID: {job_id}")
        print(f"[{self.orchestrator_name}] Path Modello calcolato: {model_path}")
        print(f"[{self.orchestrator_name}] Path Dataset calcolato: {self.test_data_path}")

        # 2. CARICAMENTO DELLA FORESTA (MODELLO GLOBALE AGGREGATO)
        if self.environment == "local":
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modello globale non trovato in '{model_path}'.")
            print(f"[{self.orchestrator_name}] Caricamento della foresta locale da {model_path}...")
            with open(model_path, "rb") as f:
                global_model = pickle.load(f)
        else:
            print(f"[{self.orchestrator_name}] Ambiente AWS: caricamento foresta...")
            local_fallback_path = os.path.join("./saved_models", f"model_{job_id}.pkl")
            with open(local_fallback_path, "rb") as f:
                global_model = pickle.load(f)
        
        all_trees = global_model.estimators_
        total_trees = len(all_trees)
        print(f"[{self.orchestrator_name}] Foresta caricata. Numero totale di alberi: {total_trees}")

        # 3. CARICAMENTO E PREPARAZIONE DEL DATASET DI TEST TRAMITE DAO
        print(f"[{self.orchestrator_name}] Caricamento Testing Set persistito via DAO: {self.test_data_path}")
        dao = DatasetDAOFactory.get_dao(self.environment)
        test_df = dao.load_dataset(self.test_data_path)

        print(f"[{self.orchestrator_name}] Preparazione della matrice di test (Shape: {test_df.shape})...")
        X_test = test_df.drop(columns=[target_col]).to_numpy(dtype=np.float64)
        y_test = test_df[target_col].to_numpy()
        serialized_X_test = pickle.dumps(X_test)

        # 4. SCOPERTA WORKER E INIZIALIZZAZIONE STRUTTURE FAULT-TOLERANT
        available_workers = ServiceRegistry.get_available_workers(self.environment)
        if not available_workers:
            raise RuntimeError("Nessun worker disponibile nel Service Registry per l'inferenza.")

        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        print(f"[{self.orchestrator_name}] Worker pronti per l'inferenza: {num_workers} -> {worker_names}")

        # Calcolo dinamico granulare della dimensione del chunk di alberi
        CHUNK_SIZE = max(1, total_trees // (num_workers * 2))
        print(f"[{self.orchestrator_name}] CHUNK_SIZE di inferenza impostata a {CHUNK_SIZE} alberi per task.")

        # Popolamento della coda thread-safe dei sotto-task di inferenza
        task_queue = queue.Queue()
        tree_start = 0
        task_id_counter = 0
        
        while tree_start < total_trees:
            tree_end = min(tree_start + CHUNK_SIZE, total_trees)
            chunk_estimators = all_trees[tree_start:tree_end]
            serialized_chunk_trees = pickle.dumps(chunk_estimators)
            
            task_queue.put((task_id_counter, tree_start, tree_end, serialized_chunk_trees))
            task_id_counter += 1
            tree_start = tree_end

        # Strutture dati condivise protette da Lock per i thread consumatori
        predictions_chunks = []
        results_lock = threading.Lock()
        connessioni_attive = []
        connessioni_lock = threading.Lock()
        active_worker_names = list(worker_names)

        # 5. DEFINIZIONE DEL CONSUMATORE CONCORRENTE PER L'INFERENZA VIA RPC
        def inference_worker_consumer(w_name):
            w_info = available_workers[w_name]
            worker_conn = None
            try:
                print(f" [RPC INF -> {w_name}] Apertura connessione su {w_info['host']}:{w_info['port']}...")
                worker_conn = rpyc.connect(
                    w_info["host"], 
                    w_info["port"], 
                    config={
                        'allow_pickle': True,
                        'sync_request_timeout': 600,
                        'keepalive': True
                    }
                )
                with connessioni_lock:
                    connessioni_attive.append(worker_conn)
                
                while True:
                    try:
                        task_id, start_idx, end_idx, chunk_trees_bytes = task_queue.get(timeout=1)
                    except queue.Empty:
                        break

                    quota_alberi = end_idx - start_idx
                    print(f"[{self.orchestrator_name}-InfThread] Assegnazione Task {task_id} ({quota_alberi} alberi: {start_idx}-{end_idx}) a {w_name}")
                    
                    try:
                        # Invocazione remota sul metodo esposto dal BaseWorker
                        raw_response = worker_conn.root.predict_subset_forest(
                            chunk_trees_bytes, 
                            serialized_X_test
                        )
                        sub_predictions = pickle.loads(obtain(raw_response))
                        
                        with results_lock:
                            # Tracciamo start_idx per poter riordinare sequenzialmente i blocchi alla fine
                            predictions_chunks.append((start_idx, sub_predictions))
                            
                        print(f"   [RPC INF <- {w_name}] Task {task_id} completato con successo.")
                        task_queue.task_done()
                        
                    except Exception as e:
                        print(f"   [ERRORE RPC INFERENZA] Fallimento del worker {w_name} sul Task {task_id}: {e}")
                        
                        # FAILOVER: Inserimento immediato del task interrotto nuovamente in coda
                        task_queue.put((task_id, start_idx, end_idx, chunk_trees_bytes))
                        print(f"[{self.orchestrator_name}-InfThread] Task {task_id} riaccodato per il failover.")
                        
                        with results_lock:
                            if w_name in active_worker_names:
                                active_worker_names.remove(w_name)
                        break  # Interruzione del loop per questo canale RPC corrotto
                        
            except Exception as conn_err:
                print(f"   [ERRORE CONNESSIOINE INFERENZA] Impossibile raggiungere il worker {w_name}: {conn_err}")
                with results_lock:
                    if w_name in active_worker_names:
                        active_worker_names.remove(w_name)
            finally:
                if worker_conn:
                    try:
                        worker_conn.close()
                        with connessioni_lock:
                            if worker_conn in connessioni_attive:
                                connessioni_attive.remove(worker_conn)
                    except:
                        pass

        # 6. AVVIO MULTI-THREADING E SINCRONIZZAZIONE DEI CONSUMATORI
        rpc_start_time = time.perf_counter()
        threads = []
        for name in worker_names:
            t = threading.Thread(target=inference_worker_consumer, args=(name,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Rilevamento di un eventuale crash totale di tutti i nodi di calcolo
        if not task_queue.empty() and len(active_worker_names) == 0:
            print(f"   [{self.orchestrator_name}] Tutti i worker sono crashati durante lo step di inferenza!")
            raise RuntimeError("Sotto-sistema Fault Tolerance Inferenza interrotto: Nessun worker rimasto disponibile.")

        # Chiusura precauzionale di socket RPyC rimasti aperti
        with connessioni_lock:
            for conn in connessioni_attive:
                try: conn.close()
                except Exception: pass

        rpc_inference_time = time.perf_counter() - rpc_start_time

        # 7. ORDINAMENTO SEQUENZIALE E COMPOSIZIONE DELLA MATRICE DELLE PREDIZIONI
        print(f"[{self.orchestrator_name}] Collezionamento predizioni completato. Ricomposizione matrice in corso...")
        predictions_chunks.sort(key=lambda x: x[0])
        
        all_worker_predictions = []
        for _, sub_preds in predictions_chunks:
            all_worker_predictions.extend(sub_preds)

        predictions_matrix = np.array(all_worker_predictions)
        print(f"[{self.orchestrator_name}] Matrice complessiva delle predizioni rigenerata: {predictions_matrix.shape}")
        
        total_inference_time = time.perf_counter() - inference_start_time

        # 8. DELEGA AL METODO MODULARE PER IL CALCOLO E LA STAMPA DELLE METRICHE
        self._print_and_validate_metrics(
            predictions_matrix=predictions_matrix,
            y_test=y_test,
            tree_type=tree_type,
            testing_set_size=X_test.shape[0],
            job_id=job_id,
            total_inference_time=total_inference_time,
            rpc_inference_time=rpc_inference_time
        )

    def _print_and_validate_metrics(
        self, 
        predictions_matrix: np.ndarray, 
        y_test: np.ndarray, 
        tree_type: str, 
        testing_set_size: int,
        job_id: str,
        total_inference_time: float,
        rpc_inference_time: float
    ):
        """
        Metodo helper per il calcolo, la validazione statistica e la stampa 
        delle metriche di performance del modello globale.
        """
        print("\n" + "═" * 75)
        print(f"  VALUTAZIONE PRESTAZIONI MODELLO DISTRIBUITO FAULT-TOLERANT (JOB: {job_id[:8]})")
        print("═" * 75)
        print(f"  TEMPO TOTALE DI INFERENZA:              {total_inference_time:.4f} secondi")
        print("═" * 75 + "\n")
        print(f"  TEMPO INFERENZA DISTRIBUITA RPC:        {rpc_inference_time:.4f} secondi")

        if tree_type == "classifier":
            # Calcolo della maggioranza dei voti pesata (in questo caso pesi uniformi)
            uniform_weights = np.ones_like(predictions_matrix)
            final_predictions, _ = weighted_mode(predictions_matrix, uniform_weights, axis=0)
            final_predictions = final_predictions.ravel().astype(int)
            y_test = y_test.astype(int)
            
            # Calcolo delle metriche di classificazione standard
            accuracy = np.mean(final_predictions == y_test)
            precision = precision_score(y_test, final_predictions, zero_division=0)
            recall = recall_score(y_test, final_predictions, zero_division=0)
            f1 = f1_score(y_test, final_predictions, zero_division=0)
            cm = confusion_matrix(y_test, final_predictions)
            
            print(f"  Tipo di Modello:                        CLASSIFICATORE")
            print(f"  Testing Set size:                       {testing_set_size} campioni")
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
            # Caso Regressore: media aritmetica dei valori continui predetti dagli alberi
            final_predictions = np.mean(predictions_matrix, axis=0)
            mae = np.mean(np.abs(final_predictions - y_test))
            print(f"  Tipo di Modello:                        REGRESSORE")
            print(f"  Testing Set size:                       {testing_set_size} campioni")
            print(f"  MAE FINALE DISTRIBUITO:                 {mae:.4f}")

        print("═" * 75 + "\n")

if __name__ == "__main__":
    print("[BOOT] Avvio del nodo Orchestratore Centralizzato...")
    orchestrator = CentralizedOrchestrator()
    orchestrator.start()