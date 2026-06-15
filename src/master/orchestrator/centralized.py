import pickle
import os
import json
import time
import rpyc
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.utilities.datasplitter import StratifiedDataSplitter
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.featureselection import CICIDSFeatureSelector

class CentralizedOrchestrator(BaseOrchestrator):
    def __init__(self, environment: str = "local"):
        # Recuperiamo il Process ID per distinguere le repliche nei log
        pid = os.getpid()
        super().__init__(
            orchestrator_name=f"Orchestrator-Centralizzato-{pid}",
            queue_name="centralized_queue",
            environment=environment
        )
        self.current_job_id=None
        self.train_data_path=None
        self.test_df=None

    def _resolve_dataset_type(self, payload: dict) -> str:
        """
        Determina il tipo di dataset basandosi esclusivamente sull'informazione 
        esplicita inviata dal Client nel payload.
        """
        # Il client ora invia 'dataset_type' ("real" o "synthetic")
        dataset_type = payload.get("dataset_type")
        
        if dataset_type:
            return str(dataset_type).strip().lower()
            
        # Fallback di sicurezza: se per qualche motivo manca, default a 'real'
        return "real"
    
    def _prepare_data(self, payload: dict, base_seed: int):
        self.current_job_id = payload.get("job_id", "unknown_job") # Salva l'ID!
        dataset_path = payload.get("dataset_path")
        dataset_type = self._resolve_dataset_type(payload)
        target_col = "Label" # Standardizzato

        print(f"\n[{self.orchestrator_name}] Avvio ETL. Tipo: {dataset_type}")

        # --- ESTRAZIONE ---
        if dataset_type == "synthetic":
            loader = SyntheticDataLoader(n_samples=100000, random_seed=base_seed, target_column=target_col)
            df_clean = loader.load()
        else:
            if not dataset_path: raise ValueError("dataset_path mancante.")
            loader = RawCSVDataLoader(data_url=dataset_path, sample_fraction=0.01, dataset_seed=base_seed)
            df_raw = loader.load()
            preprocessor = CICIDSPreprocessor(target_column=target_col)
            df_clean = preprocessor.process(df_raw)

        # --- SPLIT (Logica unificata) ---
        test_size = 0.1 if dataset_type == "synthetic" else 0.2
        splitter = StratifiedDataSplitter(target_column=target_col, test_size=test_size, random_state=base_seed)
        train_df, test_df = splitter.split(df_clean)

        # --- FEATURE SELECTION (Solo Real) ---
        if dataset_type == "real":
            fs = CICIDSFeatureSelector(target_column=target_col, correlation_threshold=0.05)
            train_df = fs.fit_transform(train_df)
            test_df = fs.transform(test_df)

        # --- SALVATAGGIO ---
        self.test_df = test_df
        self.train_data_path = f"shared_train_{self.current_job_id}.csv"
        
        if self.environment == "aws":
            # ... logica S3 ...
            pass 
        else:
            train_df.to_csv(self.train_data_path, index=False)
            print(f"[{self.orchestrator_name}] Dati pronti: {self.train_data_path}")


    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int):
        
        if self.train_data_path is None:
            self._prepare_data(payload, seed)
        
        total_step_trees= target_alberi - start_alberi
        print(f"\n [{self.orchestrator_name}] Distribuzione carico: {total_step_trees} alberi da generare  (seed: {seed})...")

        while True:
            available_workers = ServiceRegistry.get_available_workers(self.environment)
            if available_workers:
                print(f"[{self.orchestrator_name}] Worker rilevati: {list(available_workers.keys())}. Procedo...")
                break # Esci dal ciclo e inizia il training
            
            print(f"[{self.orchestrator_name}] Nessun worker disponibile. In Attesa...")
            time.sleep(10) # Pausa di 10 secondi prima di scansionare di nuovo

        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        print(f"[{self.orchestrator_name}] Worker disponibili: {num_workers} -> {worker_names}")

        # Distribuzione del carico in modo bilanciato tra i worker disponibili
        trees_per_worker = total_step_trees // num_workers
        remainder = total_step_trees % num_workers

        source_info = self.train_data_path 
        
        hp = payload.get("hyperparameters", {})
        max_depth = hp.get("max_depth", None)

        all_trained_trees = []
        current_seed_offset = seed
        connessioni_attive = []

        #funzione di task per il thread pool: gestisce la singola connessione RPyC
        def _rpc_call(w_name,w_info,n_trees,w_seed):
            print(f" [RPC -> {w_name}] Invio richiesta per {n_trees} alberi su {w_info['host']}:{w_info['port']}...")
            conn = rpyc.connect(w_info["host"], w_info["port"], config= {'allow_pickle': True})
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

            # Raccolta asincrona dei risultati
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
               
                    print("\n" + "="*60)
                    print(f" DETTAGLIO ERRORE (TRACEBACK) PER IL WORKER: {w_name}")
                    print("="*60)
                    traceback.print_exc()  
                    print("="*60 + "\n")
                  
                    raise e  

        print(f"[*] Pulizia risorse: chiusura di {len(connessioni_attive)} connessioni RPyC attive...")
        for conn in connessioni_attive:
            try:
                conn.close()
            except Exception :
                pass
        if len(all_trained_trees) > 0:
            print(f"   [{self.orchestrator_name}] Step centralizzato completato.")
            return True
        
        return False
                

if __name__ == "__main__":
    env = "local"
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                env = json.load(f).get("environment", "local")
        except Exception:
            pass
            
    orchestrator = CentralizedOrchestrator(environment=env)
    orchestrator.start()