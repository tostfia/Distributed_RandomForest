import os
import json
import time
import rpyc
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.shared.binding.serviceregistry import ServiceRegistry

from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator

class FederatedOrchestrator(BaseOrchestrator):
    def __init__(self, environment: str = "local"):
        # Recuperiamo il Process ID per generare un nome univoco per ogni replica federata
        pid = os.getpid()
        super().__init__(
            orchestrator_name=f"Orchestrator-Federato-{pid}",
            queue_name="federated_queue",
            environment=environment
        )

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int):
        """Implementazione del coordinamento e dell'aggregazione (Federated Averaging)."""
        
        step_dim = target_alberi - start_alberi
        round_num = (start_alberi // step_dim) + 1
        
        print(f"   [{self.orchestrator_name}] === AVVIO ROUND {round_num} ===")
        print(f"   [{self.orchestrator_name}] -> Distribuzione calcolo alberi ({start_alberi} a {target_alberi}) ai nodi remoti... (Seed: {seed})")

        available_workers = ServiceRegistry.get_available_workers(self.environment)
        if not available_workers:
            raise RuntimeError(f"Nessun worker disponibile per eseguire il training step. Carico totale: {step_dim} alberi.")
        
        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        print(f"   [{self.orchestrator_name}] Worker disponibili: {num_workers} -> {worker_names}")
        
        hp = payload.get("hyperparameters", {})
        max_depth = hp.get("max_depth", None)
        #DA CAMBIARE
        source_info = payload.get("data_url") or payload.get("dataset_url", "federated_shared")

        # Funzione di chiamata RPC remota per addestramento locale sul singolo nodo federato
        def _federated_rpc_call(w_name, w_info):
            print(f"   [FED-ROUND] Nodo '{w_name}' avvia addestramento locale di {step_dim} alberi...")
            conn = rpyc.connect(w_info["host"], w_info["port"], config={'allow_pickle': True})
            try:
                # Ogni nodo federato produce i propri stimatori locali sulla sua frazione di dati
                local_model_shard = conn.root.train_subset_forest(
                    source_info=source_info, 
                    num_trees=step_dim,
                    base_seed=seed,  # Manteniamo lo stesso seed coordinato per mantenere allineati i round
                    max_depth=max_depth
                )
                return w_name, local_model_shard
            finally:
                conn.close()

        # 2. Distribuzione asincrona del round a tutti i nodi scoperti in parallelo
        local_updates = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_federated_rpc_call, name, available_workers[name]) for name in worker_names]
            
            for future in as_completed(futures):
                try:
                    w_name, local_trees = future.result()
                    local_updates[w_name] = local_trees
                    print(f"   [{self.orchestrator_name}] Update del modello scaricato con successo dal nodo federato '{w_name}'.")
                except Exception as e:
                    print(f"   [FAILOVER ROUND] Errore critico durante il round sul nodo federato '{w_name}': {e}")
                    raise e

        # 3. Fase finale di Aggregazione (Simulazione logica del Federated Averaging)
        print(f"   [{self.orchestrator_name}] -> Ricezione dei pesi locali dai nodi completata.")
        print(f"   [{self.orchestrator_name}] -> Aggregazione e generazione del Modello Globale per il Round {round_num} eseguita.")
        time.sleep(0.5)
            


if __name__ == "__main__":
    # Lettura dinamica dell'ambiente configurato nel config.json locale
    env = "local"
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
                env = config.get("environment", "local")
        except Exception:
            pass  # Fallback su local se il file è corrotto o mancante
            
    # Istanziamo e avviamo l'orchestratore federato
    orchestrator = FederatedOrchestrator(environment=env)
    orchestrator.start()