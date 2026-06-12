from copyreg import pickle
import os
import json
import time
import rpyc
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator
from src.shared.binding.serviceregistry import ServiceRegistry

class CentralizedOrchestrator(BaseOrchestrator):
    def __init__(self, environment: str = "local"):
        # Recuperiamo il Process ID per distinguere le repliche nei log
        pid = os.getpid()
        super().__init__(
            orchestrator_name=f"Orchestrator-Centralizzato-{pid}",
            queue_name="centralized_queue",
            environment=environment
        )

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int):
        total_step_trees= target_alberi - start_alberi
        print(f"\n [{self.orchestrator_name}] Distribuzione carico: {total_step_trees} alberi da generare  (seed: {seed})...")

        #Scansione dei nodi worker attivi e disponibili (con heartbeat aggiornato)
        available_workers = ServiceRegistry.get_available_workers(self.environment)
        if not available_workers:
            # Invece di far fallire tutto subito, potresti mettere il messaggio 
            # di nuovo in coda con un delay (es. sleep) o loggare un avviso meno critico.
            print("[!] Attenzione: Nessun worker pronto. Aspetto...")
            time.sleep(10) 
            return False # Torna al loop principale senza far fallire il job
        
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
                    # ─── AGGIUNTA TRACEBACK ESATTO ──────────────────────────────────────
                    print("\n" + "="*60)
                    print(f" DETTAGLIO ERRORE (TRACEBACK) PER IL WORKER: {w_name}")
                    print("="*60)
                    traceback.print_exc()  # <--- Questa riga stampa l'intera catena di chiamate fino a RPyC
                    print("="*60 + "\n")
                    # ───────────────────────────────────────────────────────────────────
                    raise e  # Rilanciando l'errore attiviamo il meccanismo di failover nativo di BaseOrchestrator

        print(f"[*] Pulizia risorse: chiusura di {len(connessioni_attive)} connessioni RPyC attive...")
        for conn in connessioni_attive:
            try:
                conn.close()
            except Exception :
                pass
        if len(all_trained_trees) > 0:
            print(f"   [{self.orchestrator_name}] Step centralizzato completato.")
            return True # <--- MODIFICA: successo!
        
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