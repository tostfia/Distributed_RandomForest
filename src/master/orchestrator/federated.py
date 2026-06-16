import pickle
import os
import time
import rpyc
import traceback
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.shared.config import SystemConfig
from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator
from src.shared.binding.serviceregistry import ServiceRegistry


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

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int) -> bool:
        """
        Implementazione del coordinamento federato tramite chiamate RPC parallele (RPyC).
        I dati non viaggiano sulla rete: l'orchestratore comanda ai worker di addestrare
        i sotto-modelli sulle loro rispettive partizioni generate localmente.
        """
        self.current_job_id = payload.get("job_id", "unknown_job")
        total_step_trees = target_alberi - start_alberi
        round_num = (start_alberi // total_step_trees) + 1
        
        print(f"\n [{self.orchestrator_name}] === AVVIO ROUND FEDERATO RPC {round_num} ===")
        print(f" [{self.orchestrator_name}] Distribuzione carico: {total_step_trees} alberi federati (seed base: {seed})...")

        # --- SCOPERTA DEI WORKER ---
        while True:
            available_workers = ServiceRegistry.get_available_workers(self.environment)
            if available_workers:
                break
            
            print(f" [{self.orchestrator_name}] Nessun worker federato disponibile. In Attesa...")
            time.sleep(10)

        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        print(f" [{self.orchestrator_name}] Worker federati rilevati: {num_workers} -> {worker_names}")

        # Partizionamento del numero di alberi di questo step tra i worker attivi
        trees_per_worker = total_step_trees // num_workers
        remainder = total_step_trees % num_workers

        hp = payload.get("hyperparameters", {})
        max_depth = hp.get("max_depth", None)
        tree_type = hp.get("tree_type", "classifier")

        all_trained_trees = []
        current_seed_offset = seed
        connessioni_attive = []

        # --- FUNZIONE INTERNA DI CHIAMATA RPC ADATTATA AL FEDERATO ---
        def _federated_rpc_call(w_name, w_info, n_trees, w_seed, idx_worker):
            print(f" [RPC -> {w_name}] Invio comando training FEDERATO locale ({n_trees} alberi)...")
            conn = rpyc.connect(w_info["host"], w_info["port"], config={'allow_pickle': True})
            connessioni_attive.append(conn)  
            try: 
                # FIX: Impacchettiamo l'indice del worker nella stringa source_info (es: "NATIVE_PARTITIONED|1")
                # Rispettiamo al 100% la firma rigida di BaseWorker senza usare keyword arguments extra.
                federated_source_info = f"NATIVE_PARTITIONED|{idx_worker}"

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

                # Passiamo l'indice progressivo 'i' che verrà convertito in metadato di stringa
                f = executor.submit(_federated_rpc_call, name, info, allocated_trees, current_seed_offset, i)
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
                    print(f"   [RPC <- {w_name}] Ricevuti con successo {len(result_trees)} alberi federati.")
                except Exception as e:
                    print(f"   [ERRORE CRITICO FEDERATO] Il worker '{w_name}' ha fallito nel round: {e}")
                    traceback.print_exc()  
                    raise e  

        # --- PULIZIA CONNESSIONI RPC ---
        print(f"[*] Pulizia risorse: chiusura di {len(connessioni_attive)} connessioni RPyC attive...")
        for conn in connessioni_attive:
            try:
                conn.close()
            except Exception:
                pass

        # --- RICOMPOSIZIONE E AGGREGAZIONE DEL MODELLO GLOBALE ---
        if len(all_trained_trees) > 0:
            if target_alberi == hp.get("n_estimators", 100):
                print(f"   [{self.orchestrator_name}] Ricomposizione foresta globale federata...")
                try:
                    if tree_type == "classifier":
                        global_model = RandomForestClassifier(n_estimators=len(all_trained_trees))
                        global_model.classes_ = np.array([0, 1]) 
                        global_model.n_classes_ = 2
                    else:
                        global_model = RandomForestRegressor(n_estimators=len(all_trained_trees))
                    
                    global_model.estimators_ = all_trained_trees
                    
                    model_path = f"model_federated_{self.current_job_id}.pkl"
                    with open(model_path, "wb") as f:
                        pickle.dump(global_model, f)
                    
                    print(f"   [{self.orchestrator_name}] [OK] Modello finale federato salvato in '{model_path}'.")
                except Exception as e:
                    print(f"   [ERRORE AGGREGAZIONE FEDERATA] Impossibile creare il modello finale: {e}")
                    traceback.print_exc()
                    return False
            return True

        print(f"   [{self.orchestrator_name}] Nessun albero ricevuto dal round federato. Fallimento.")
        return False


if __name__ == "__main__":
    print("[BOOT] Avvio del nodo Orchestratore Federato (RPC Symmetric)...")
    orchestrator = FederatedOrchestrator()
    orchestrator.start()