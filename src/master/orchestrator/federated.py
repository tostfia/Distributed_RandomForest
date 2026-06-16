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
                    
                    # MODIFICA: Salvataggio isolato all'interno della cartella saved_models
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

        print(f"   [{self.orchestrator_name}] Nessun albero ricevuto dal round federato. Fallimento.")
        return False
    
    def _execute_inference_step(self, payload: dict):
        """
        Esegue l'inferenza distribuita/valutazione in modalità Federata.
        Invia a ciascun worker un sottoinsieme di alberi della foresta globale federata,
        lasciando che i worker computino le predizioni parziali in modo isolato sui loro
        rispettivi testing set privati salvati in RAM (nessun dato viaggia in rete).
        """
        job_id = payload.get("job_id")
        hp = payload.get("hyperparameters", {})
        tree_type = hp.get("tree_type", "classifier")

        print(f"\n[{self.orchestrator_name}] === AVVIO INFERENZA DISTRIBUITA FEDERATA ===")

        # MODIFICA: Caricamento puntato alla sottocartella saved_models
        model_path = os.path.join("./saved_models", f"model_federated_{job_id}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modello globale federato non trovato in '{model_path}'. "
                f"Assicurati che l'addestramento sia terminato."
            )

        print(f"[{self.orchestrator_name}] Caricamento della foresta federata da {model_path}...")
        with open(model_path, "rb") as f:
            global_model = pickle.load(f)
        
        all_trees = global_model.estimators_
        total_trees = len(all_trees)
        print(f"[{self.orchestrator_name}] Foresta federata caricata. Numero totale di alberi: {total_trees}")

        # 2. Scoperta dei Worker federati attivi
        available_workers = ServiceRegistry.get_available_workers(self.environment)
        if not available_workers:
            raise RuntimeError("Nessun worker federato disponibile nel Service Registry per l'inferenza.")

        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        print(f"[{self.orchestrator_name}] Worker federati pronti per la valutazione: {num_workers} -> {worker_names}")

        # Partizionamento equo degli alberi globali della foresta tra i nodi
        trees_per_worker = total_trees // num_workers
        remainder = total_trees % num_workers

        connessioni_attive = []
        metriche_nodi = []

        # 3. Funzione di chiamata RPC per l'inferenza asimmetrica federata
        def _rpc_federated_inference_call(w_name, w_info, subset_trees_chunk):
            print(f" [RPC FED-INFERENZA -> {w_name}] Invio di {len(subset_trees_chunk)} alberi globali...")
            conn = rpyc.connect(w_info["host"], w_info["port"], config={'allow_pickle': True})
            connessioni_attive.append(conn)

            # Invochiamo il metodo del worker passando il chunk di alberi MA lasciando serialized_X_test = None
            # Questo costringe il Worker a usare il proprio `self.X_test` locale trattenuto in RAM
            serialized_chunk = pickle.dumps(subset_trees_chunk)
            raw_response = conn.root.predict_subset_forest(serialized_chunk, None)
            sub_predictions = pickle.loads(raw_response) # Matrice di risposte (shape: num_alberi_chunk, num_campioni_locali)

            # Il worker deve implementare l'accesso esposto alle suas y_test per permetterci di calcolare le metriche di quel nodo
            if hasattr(conn.root, "get_local_y_test"):
                raw_y_test = conn.root.get_local_y_test()
            elif hasattr(conn.root, "y_test"):
                raw_y_test = conn.root.y_test
            else:
                raise AttributeError(f"Il worker {w_name} non espone il set delle label locali y_test necessarie alla validazione delle metriche.")
            
            y_test_locale = pickle.loads(raw_y_test) if isinstance(raw_y_test, bytes) else np.array(raw_y_test)

            # Aggregazione locale dei voti di questo specifico nodo (Voto di maggioranza o media delle risposte)
            matrix_pred_locale = np.array(sub_predictions)

            if tree_type == "classifier":
                from scipy.stats import mode
                final_local_pred, _ = mode(matrix_pred_locale, axis=0)
                final_local_pred = final_local_pred.ravel()
                acc_locale = np.mean(final_local_pred == y_test_locale)
                return w_name, "accuracy", acc_locale, len(y_test_locale)
            else:
                final_local_pred = np.mean(matrix_pred_locale, axis=0)
                mae_locale = np.mean(np.abs(final_local_pred - y_test_locale))
                return w_name, "mae", mae_locale, len(y_test_locale)

        # 4. Esecuzione in parallelo sui Worker via ThreadPoolExecutor
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

            # Raccolta e aggregazione delle metriche ricevute dai nodi isolati
            for future in as_completed(future_to_worker):
                w_name = future_to_worker[future]
                try:
                    name_w, m_type, valore, size_test = future.result()
                    metriche_nodi.append((valore, size_test))
                    
                    val_str = f"{valore * 100:.2f} %" if m_type == "accuracy" else f"{valore:.4f}"
                    print(f"   [RPC FED-INFERENZA <- {w_name}] Valutazione completata. Risultato Locale ({m_type.upper()}): {val_str} su {size_test} campioni.")
                except Exception as e:
                    print(f"   [ERRORE CRITICO FED-INFERENZA] Fallimento computazione sul worker '{w_name}': {e}")
                    raise e

        # Chiusura pulita dei canali RPC
        print(f"[*] Chiusura di {len(connessioni_attive)} canali RPC federati...")
        for conn in connessioni_attive:
            try:
                conn.close()
            except Exception:
                pass

        # 5. Calcolo della Metrica Globale Federata (Media Pesata in base al numero di campioni di ciascun nodo)
        total_samples = sum(item[1] for item in metriche_nodi)
        weighted_metric_sum = sum(item[0] * item[1] for item in metriche_nodi)
        global_federated_metric = weighted_metric_sum / total_samples if total_samples > 0 else 0

        print("\n" + "═" * 75)
        print(f"  VALUTAZIONE PERFORMANCE MODELLO FEDERATO GLOBALE (JOB: {job_id[:8]})")
        print("═" * 75)
        print(f"  Numero totale nodi federati valutati:  {len(metriche_nodi)}")
        print(f"  Dimensione aggregata testing set:      {total_samples} campioni totali")

        if tree_type == "classifier":
            print(f"  Tipo di Modello:                        CLASSIFICATORE")
            print(f"  ACCURACY MEDIA PESATA SUI NODI:         {global_federated_metric * 100:.2f} %")
        else:
            print(f"  Tipo di Modello:                        REGRESSORE")
            print(f"  MAE MEDIO PESATO SUI NODI:              {global_federated_metric:.4f}")

        print("═" * 75 + "\n")

    def exposed_get_local_y_test(self):
        """Rende disponibile all'Orchestratore il vettore y_test in formato serializzato."""
        if self.y_test is None:
            raise ValueError(f"[{self.orchestrator_name}] Errore: Nessun target vector locale y_test in RAM.")
        return pickle.dumps(self.y_test)


if __name__ == "__main__":
    print("[BOOT] Avvio del nodo Orchestratore Federato (RPC Symmetric)...")
    orchestrator = FederatedOrchestrator()
    orchestrator.start()