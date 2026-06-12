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
        self.current_jib_id=None
        self.train_data_path=None
        self.test_df=None

    def _resolve_dataset_type(self, payload: dict, hp: dict) -> str:
        """
        Determina se il dataset è reale o sintetico.

        Priorità:
        1. dataset_type nel payload/config;
        2. target_column negli hyperparameters;
        3. fallback su real.
        """
        dataset_type = payload.get("dataset_type")

        if dataset_type is not None:
            return str(dataset_type).strip().lower()

        target_column = hp.get("target_column")

        if target_column == "target":
            return "synthetic"

        if target_column == "Label":
            return "real"

        return "real"
    
    def _prepare_data(self, payload: dict, base_seed: int):
        print(f"\n[{self.orchestrator_name}] Avvio fase di estrazione e pre-processing (ETL)...")
        start_time = time.perf_counter()

        dataset_path = payload.get("dataset_path")
        hp = payload.get("hyperparameters", {})

        dataset_type = self._resolve_dataset_type(payload, hp)

        if dataset_type == "synthetic":
            target_col = hp.get("target_column", "target")
        elif dataset_type == "real":
            target_col = hp.get("target_column", "Label")
        else:
            raise ValueError(
                f"dataset_type non valido: {dataset_type}. Usa 'real' oppure 'synthetic'."
            )
        # ---------------------------------------------------------
        # ESTRAZIONE E PREPROCESSING
        # ---------------------------------------------------------
        if dataset_type == "synthetic":
            print(
                f"[{self.orchestrator_name}] -> Generazione Dataset Sintetico "
                f"con seed {base_seed}..."
            )

            loader = SyntheticDataLoader(
                n_samples=100000,
                random_seed=base_seed,
                target_column=target_col
            )

            df_clean = loader.load()

        elif dataset_type == "real":
            if not dataset_path:
                raise ValueError("dataset_path mancante per dataset reale.")

            print(f"[{self.orchestrator_name}] -> Estrazione Dataset Reale da: {dataset_path}")
            loader = RawCSVDataLoader(
                data_url=dataset_path,
                sample_fraction=0.01,
                dataset_seed=base_seed
            )

            df_raw = loader.load()

            print(f"[{self.orchestrator_name}] -> Esecuzione pulizia CICIDSPreprocessor...")

            preprocessor = CICIDSPreprocessor(
                target_column=target_col
            )

            df_clean = preprocessor.process(df_raw)

        etl_time = time.perf_counter() - start_time

        print(f"[{self.orchestrator_name}] ETL completato in "f"{etl_time:.2f}s. Dimensione: {df_clean.shape}")
        print(f"\nDistribuzione target '{target_col}':")
        print(df_clean[target_col].value_counts())

        print(f"\nDistribuzione target '{target_col}' percentuale:")
        print(df_clean[target_col].value_counts(normalize=True) * 100)
        # ---------------------------------------------------------
        # SPLIT TRAIN/TEST
        # ---------------------------------------------------------
        print(f"\n[{self.orchestrator_name}] Partizionamento stratificato in Train e Test...")

        if dataset_type == "real":
            test_size = 0.2
        else:
            test_size = 0.1

        splitter = StratifiedDataSplitter(
            target_column=target_col,
            test_size=test_size,
            random_state=base_seed
        )

        train_df, test_df = splitter.split(df_clean)

        # ---------------------------------------------------------
        # FEATURE SELECTION SOLO PER DATASET REALE
        # ---------------------------------------------------------
        if dataset_type == "real":
            print(f"\n[{self.orchestrator_name}] Feature Selection sul solo Train Set...")
            feature_selector = CICIDSFeatureSelector(
                target_column=target_col,
                correlation_threshold=0.05
            )

            train_df = feature_selector.fit_transform(train_df)
            test_df = feature_selector.transform(test_df)

            print(f" • Dimensione Train dopo Feature Selection: {train_df.shape}")
            print(f" • Dimensione Test dopo Feature Selection:  {test_df.shape}")

        self.test_df = test_df
        file_name = f"shared_train_{self.current_job_id}.csv"
        if self.environment == "aws":
            bucket_name = os.getenv("S3_TEMP_BUCKET", "il-mio-bucket-sdcc")
            self.train_data_path = f"s3://{bucket_name}/processed_jobs/{file_name}"

            print(f"[{self.orchestrator_name}] AWS Mode: Upload del Train Set su S3 in corso...")
            train_df.to_csv(self.train_data_path, index=False)
            print(f"[{self.orchestrator_name}] Dati caricati con successo su: {self.train_data_path}")

        else:
            self.train_data_path = file_name
            train_df.to_csv(self.train_data_path, index=False)
            print(f"[{self.orchestrator_name}] Local Mode: Train salvato in '{self.train_data_path}'.")

    

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int):
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

        #Estrazione iper param e dataset DA RIFARE
        # Cerca la chiave corretta che invia il client
        source_info = payload.get("dataset_path") or "default_dataset.csv"
        hp = payload.get("hyperparameters", {})
        max_depth = hp.get("max_depth", None)

        all_trained_trees = []
        current_seed_offset= seed

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