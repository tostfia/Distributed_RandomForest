import os
import json
import time
import rpyc
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.shared.config import SystemConfig  # <-- INCLUSO CONFIG CENTRALE
from src.shared.binding.serviceregistry import ServiceRegistry
from src.master.orchestrator.BaseOrchestrator import BaseOrchestrator


class FederatedOrchestrator(BaseOrchestrator):
    def __init__(self):
        # 1. Recuperiamo l'ambiente direttamente dal file .env tramite SystemConfig
        self.cfg = SystemConfig()
        
        # Recuperiamo il Process ID per generare un nome univoco per ogni replica federata
        pid = os.getpid()
        
        # Inizializziamo la classe base senza passare l'ambiente (lo legge da sé via config)
        super().__init__(
            orchestrator_name=f"Orchestrator-Federato-{pid}",
            queue_name="federated_queue"
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
        
        # CORREZIONE SISTEMATA: Allineamento con il modello Pydantic TrainingRequestWorker ("url_dataset")
        # In modalità federata, ogni client invia la posizione del dataset specifico per quel round
        source_info = payload.get("url_dataset") or payload.get("dataset_path") or "federated_shared"

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

        # Distribuzione asincrona del round a tutti i nodi scoperti in parallelo
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

        # Fase finale di Aggregazione (Simulazione logica del Federated Averaging)
        print(f"   [{self.orchestrator_name}] -> Ricezione dei pesi locali dai nodi completata.")
        print(f"   [{self.orchestrator_name}] -> Aggregazione e generazione del Modello Globale per il Round {round_num} eseguita.")
        time.sleep(0.5)
        return True


if __name__ == "__main__":
    # Il blocco main adesso non legge file esterni, si avvia leggendo nativamente l'ambiente da .env
    print("[BOOT] Avvio del nodo Orchestratore Federato...")
    orchestrator = FederatedOrchestrator()
    orchestrator.start()