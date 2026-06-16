import pickle
import os
import time
import rpyc
import traceback
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.shared.config import SystemConfig
from src.shared.factory import DatasetDAOFactory
from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.utilities.datasplitter import StratifiedDataSplitter
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
        self.test_df = None

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

        # --- ESTRAZIONE ---
        if dataset_type == "synthetic":
            loader = SyntheticDataLoader(n_samples=100000, random_seed=base_seed, target_column=target_col)
            df_clean = loader.load()
        else:
            if not dataset_path: 
                raise ValueError("dataset_path mancante.")
            loader = RawCSVDataLoader(data_url=dataset_path, sample_fraction=0.01, dataset_seed=base_seed)
            df_raw = loader.load()
            preprocessor = CICIDSPreprocessor(target_column=target_col)
            df_clean = preprocessor.process(df_raw)

        # --- SPLIT ---
        test_size = 0.1 if dataset_type == "synthetic" else 0.2
        splitter = StratifiedDataSplitter(target_column=target_col, test_size=test_size, random_state=base_seed)
        train_df, test_df = splitter.split(df_clean)

        # --- FEATURE SELECTION (Solo Real) ---
        if dataset_type == "real":
            fs = CICIDSFeatureSelector(target_column=target_col, correlation_threshold=0.05)
            train_df = fs.fit_transform(train_df)
            test_df = fs.transform(test_df)

        # --- SALVATAGGIO INTERAMENTE COORDINATO DAI DAO ---
        self.test_df = test_df
        
        # Stabiliamo il path di salvataggio in base all'infrastruttura
        if self.environment == "aws":
            self.train_data_path = f"s3://my-cluster-datasets-bucket/distributed_trains/shared_train_{self.current_job_id}.csv"
        else:
            self.train_data_path = f"./shared_train_{self.current_job_id}.csv"
            
        print(f"[{self.orchestrator_name}] Delega salvataggio a DatasetDAOFactory per l'ambiente {self.environment.upper()}...")
        
        try:
            # Otteniamo il DAO corretto in base all'ambiente letto dal file .env
            dao = DatasetDAOFactory.get_dao(self.environment)
            dao.save_dataset(path=self.train_data_path, df=train_df)
            print(f"[{self.orchestrator_name}] [OK] Dataset di addestramento pronto e archiviato in: {self.train_data_path}")
        except Exception as e:
            raise IOError(f"[{self.orchestrator_name}] Errore critico nel salvataggio del dataset tramite DAO: {e}")


    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int):
        if self.train_data_path is None:
            self._prepare_data(payload, seed)
        
        total_step_trees = target_alberi - start_alberi
        print(f"\n [{self.orchestrator_name}] Distribuzione carico: {total_step_trees} alberi da generare (seed: {seed})...")

        while True:
            available_workers = ServiceRegistry.get_available_workers(self.environment)
            if available_workers:
                print(f"[{self.orchestrator_name}] Worker rilevati: {list(available_workers.keys())}. Procedo...")
                break
            
            print(f"[{self.orchestrator_name}] Nessun worker disponibile. In Attesa...")
            time.sleep(10)

        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        print(f"[{self.orchestrator_name}] Worker disponibili: {num_workers} -> {worker_names}")

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
            conn = rpyc.connect(w_info["host"], w_info["port"], config={'allow_pickle': True})
            connessioni_attive.append(conn)  
            try: 
                remote_trees = conn.root.train_subset_forest(source_info=source_info, num_trees=n_trees, base_seed=w_seed, max_depth=max_depth)
                return remote_trees
            except Exception as e:
                print(f"   [ERRORE RPC] Connessione fallita con {w_name} ({w_info['host']}:{w_info['port']}): {e}")
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
                    if isinstance(result_raw, bytes):
                        result_trees = pickle.loads(result_raw)
                    else:
                        result_trees = result_raw
                    all_trained_trees.extend(result_trees)
                    print(f"   [RPC <- {w_name}] Ricevuti con successo {len(result_trees)} alberi.")
                except Exception as e:
                    print(f"   [ERRORE CRITICO] Il worker '{w_name}' ha fallito o si è disconnesso: {e}")
                    traceback.print_exc()  
                    raise e  

        print(f"[*] Pulizia risorse: chiusura di {len(connessioni_attive)} connessioni RPyC attive...")
        for conn in connessioni_attive:
            try:
                conn.close()
            except Exception:
                pass

        # --- RICOMPOSIZIONE E SALVATAGGIO ---
        if len(all_trained_trees) > 0:
            print(f"   [{self.orchestrator_name}] Ricomposizione foresta globale...")
            hp = payload.get("hyperparameters", {})
            tree_type = hp.get("tree_type", "classifier")
            
            try:
                if tree_type == "classifier":
                    global_model = RandomForestClassifier(n_estimators=len(all_trained_trees))
                    global_model.classes_ = np.array([0, 1]) 
                    global_model.n_classes_ = 2
                else:
                    global_model = RandomForestRegressor(n_estimators=len(all_trained_trees))
                
                global_model.estimators_ = all_trained_trees
                
                # Salvataggio isolato all'interno della cartella saved_models
                TARGET_DIR = "./saved_models"
                os.makedirs(TARGET_DIR, exist_ok=True)
                model_path = os.path.join(TARGET_DIR, f"model_{self.current_job_id}.pkl")
                
                with open(model_path, "wb") as f:
                    pickle.dump(global_model, f)
                
                print(f"   [{self.orchestrator_name}] Modello {tree_type} salvato in '{model_path}'.")
                return True
                
            except Exception as e:
                print(f"   [ERRORE AGGREGAZIONE] Impossibile creare il modello {tree_type}: {e}")
                traceback.print_exc()
                return False

        print(f"   [{self.orchestrator_name}] Nessun albero ricevuto. Fallimento.")
        return False
    
    def _execute_inference_step(self, payload: dict):
        """
        Esegue l'inferenza distribuita in modalità Centralizzata.
        Scarica/ottiene il testing set centralizzato e lo invia via RPC a tutti i worker
        insieme a sottoinsiemi della foresta per calcolare la predizione finale aggregata.
        """
        job_id = payload.get("job_id")
        hp = payload.get("hyperparameters", {})
        tree_type = hp.get("tree_type", "classifier")
        target_col = "Label"

        print(f"\n[{self.orchestrator_name}] === AVVIO INFERENZA DISTRIBUITA CENTRALIZZATA ===")

        # Caricamento puntato alla sottocartella saved_models
        model_path = os.path.join("./saved_models", f"model_{job_id}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modello globale non trovato in '{model_path}'. "
                f"Assicurati che il job di addestramento sia completato con successo."
            )

        print(f"[{self.orchestrator_name}] Caricamento della foresta da {model_path}...")
        with open(model_path, "rb") as f:
            global_model = pickle.load(f)
        
        all_trees = global_model.estimators_
        total_trees = len(all_trees)
        print(f"[{self.orchestrator_name}] Foresta caricata. Numero totale di alberi: {total_trees}")

        # 2. Controllo e preparazione del Testing Set centralizzato
        if self.test_df is None:
            # Fallback se il Master ha perso la RAM o è subentrato un failover post-training
            raise ValueError(
                f"[{self.orchestrator_name}] Errore: Il dataframe di test 'self.test_df' non è "
                f"presente in memoria. Impossibile procedere con l'inferenza centralizzata."
            )

        print(f"[{self.orchestrator_name}] Preparazione della matrice di test (Shape: {self.test_df.shape})...")
        X_test = self.test_df.drop(columns=[target_col]).to_numpy(dtype=np.float64)
        y_test = self.test_df[target_col].to_numpy()

        # Serializziamo X_test una sola volta per non appesantire i thread RPC
        serialized_X_test = pickle.dumps(X_test)

        # 3. Scoperta dei Worker disponibili per l'inferenza
        available_workers = ServiceRegistry.get_available_workers(self.environment)
        if not available_workers:
            raise RuntimeError("Nessun worker disponibile nel Service Registry per gestire la computazione dell'inferenza.")

        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        print(f"[{self.orchestrator_name}] Worker pronti per l'inferenza: {num_workers} -> {worker_names}")

        # Bilanciamento e partizionamento degli alberi tra i worker
        trees_per_worker = total_trees // num_workers
        remainder = total_trees % num_workers

        all_worker_predictions = []
        connessioni_attive = []

        # 4. Funzione di chiamata RPC parallela per l'inferenza
        def _rpc_inference_call(w_name, w_info, subset_trees_chunk):
            print(f" [RPC INFERENZA -> {w_name}] Invio di {len(subset_trees_chunk)} alberi...")
            conn = rpyc.connect(w_info["host"], w_info["port"], config={'allow_pickle': True})
            connessioni_attive.append(conn)
            
            # Richiamiamo il metodo del worker passando il chunk di alberi e il test set centralizzato
            serialized_chunk = pickle.dumps(subset_trees_chunk)
            raw_response = conn.root.predict_subset_forest(serialized_chunk, serialized_X_test)
            
            return pickle.loads(raw_response)

        # Invio parallelo delle computazioni tramite ThreadPool
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_worker = {}
            current_tree_idx = 0

            for i, name in enumerate(worker_names):
                allocated_trees = trees_per_worker + (1 if i < remainder else 0)
                if allocated_trees == 0:
                    continue

                # Estraiamo la fetta di alberi di competenza del worker i-esimo
                chunk = all_trees[current_tree_idx : current_tree_idx + allocated_trees]
                current_tree_idx += allocated_trees

                f = executor.submit(_rpc_inference_call, name, available_workers[name], chunk)
                future_to_worker[f] = name

            # Raccolta delle risposte parziali
            for future in as_completed(future_to_worker):
                w_name = future_to_worker[future]
                try:
                    sub_predictions = future.result()  # Lista di predizioni (vettore per ogni albero)
                    all_worker_predictions.extend(sub_predictions)
                    print(f"   [RPC INFERENZA <- {w_name}] Ricevute correttamente predizioni parziali.")
                except Exception as e:
                    print(f"   [ERRORE CRITICO INFERENZA] Il worker '{w_name}' ha fallito durante la predizione: {e}")
                    raise e

        # Pulizia delle connessioni aperte
        print(f"[*] Chiusura di {len(connessioni_attive)} connessioni RPC attive usate per l'inferenza...")
        for conn in connessioni_attive:
            try:
                conn.close()
            except Exception:
                pass

        # 5. Aggregazione Globale dei Risultati
        # Trasformiamo l'output combinato in una matrice numpy (Shape: num_alberi_totali, num_campioni_test)
        predictions_matrix = np.array(all_worker_predictions)
        print(f"[{self.orchestrator_name}] Matrice complessiva predizioni generata: {predictions_matrix.shape}")

        print("\n" + "═" * 75)
        print(f"  VALUTAZIONE FINALE PERFORMANCE MODELLO CENTRALIZZATO (JOB: {job_id[:8]})")
        print("═" * 75)

        if tree_type == "classifier":
            # Per i classificatori usiamo il voto di maggioranza (Moda lungo l'asse degli alberi)
            from scipy.stats import mode
            final_predictions, _ = mode(predictions_matrix, axis=0)
            final_predictions = final_predictions.ravel()
            
            # Calcolo Accuratezza
            accuracy = np.mean(final_predictions == y_test)
            print(f"  Tipo di Modello:                        CLASSIFICATORE")
            print(f"  Testing Set size:                       {X_test.shape[0]} campioni")
            print(f"  ACCURACY FINALE DISTRIBUITA:            {accuracy * 100:.2f} %")
        else:
            # Per i regressori calcoliamo la media delle predizioni di tutti gli alberi
            final_predictions = np.mean(predictions_matrix, axis=0)
            
            # Calcolo MAE (Mean Absolute Error)
            mae = np.mean(np.abs(final_predictions - y_test))
            print(f"  Tipo di Modello:                        REGRESSORE")
            print(f"  Testing Set size:                       {X_test.shape[0]} campioni")
            print(f"  MAE FINALE DISTRIBUITO:                 {mae:.4f}")

        print("═" * 75 + "\n")
                
if __name__ == "__main__":
    print("[BOOT] Avvio del nodo Orchestratore Centralizzato...")
    orchestrator = CentralizedOrchestrator()
    orchestrator.start()