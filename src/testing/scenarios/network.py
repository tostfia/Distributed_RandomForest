import os
import subprocess
import time

from src.testing.scenarios.base import BaseTestScenario


class NetworkSimulationScenario(BaseTestScenario):
    """
    Scenario 3: Simulazione di ritardi di rete tramite tc netem.

    Lo scenario applica lui stesso il delay su 'lo' via tc, esegue il
    training RPC, poi ripristina le regole originali dell'interfaccia.

    NOTA IMPORTANTE SUI PERMESSI:
    tc richiede la capability Linux CAP_NET_ADMIN. Se lo scenario gira
    dentro un container Docker, questa capability va aggiunta esplicitamente
    al servizio nel docker-compose.yml:

        services:
          test-engine:
            cap_add:
              - NET_ADMIN

    Senza questa capability, nessuna combinazione di permessi utente/sudo
    dentro il container risolverà l'errore "Operation not permitted".
    """

    def __init__(self, config, orchestrator):
        super().__init__(config, orchestrator)
        self.tc_interface = os.environ.get("TC_INTERFACE", "eth0")

    # ------------------------------------------------------------------ #
    # tc helpers                                                          #
    # ------------------------------------------------------------------ #

    def _tc_available(self) -> bool:
        """
        Controlla che tc sia installato e i permessi siano sufficienti.
        Non usa sudo: tc deve avere la capability CAP_NET_ADMIN attaccata
        al binario stesso via 'setcap cap_net_admin+ep' (fatto in fase di
        build nel Dockerfile), e il container deve avere 'cap_add: NET_ADMIN'
        nel bounding set (docker-compose.yml). Con queste due condizioni,
        tc funziona indipendentemente dall'utente che lo esegue, senza sudo.
        """
        result = subprocess.run(
            ["tc", "qdisc", "show", "dev", self.tc_interface],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def _apply_tc_rules(self, latency_ms: int, loss_percentage: float) -> bool:
        """
        Applica delay (e opzionalmente packet loss) sull'interfaccia configurata.
        Ritorna True se il comando è andato a buon fine, False altrimenti
        (es. manca la capability CAP_NET_ADMIN nel container, o il binario tc
        non ha la file capability impostata via setcap).
        """
        print(
            f"\n[tc] Configurazione '{self.tc_interface}': +{latency_ms}ms delay"
            + (f", {loss_percentage:.1f}% loss" if loss_percentage > 0 else "")
        )

        # Rimuove eventuali regole residue per evitare "File exists"
        subprocess.run(
            ["tc", "qdisc", "del", "dev", self.tc_interface, "root"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        cmd = [
            "tc", "qdisc", "add", "dev", self.tc_interface,
            "root", "netem", "delay", f"{latency_ms}ms",
        ]
        if loss_percentage > 0:
            cmd += ["loss", f"{loss_percentage:.2f}%"]

        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            print(
                f"[tc WARNING] Impossibile applicare le regole tc: {result.stderr.strip()}\n"
                f"[tc WARNING] Causa probabile: manca 'cap_add: NET_ADMIN' nel docker-compose.yml "
                f"per questo servizio, oppure il binario tc non ha la file capability "
                f"(verifica nel Dockerfile: 'setcap cap_net_admin+ep /usr/sbin/tc')."
            )
            return False

        print(f"[tc OK] Regole applicate su '{self.tc_interface}'.")
        return True

    def _clear_tc_rules(self):
        """Rimuove tutte le regole dall'interfaccia e ripristina il comportamento normale."""
        print(f"[tc CLEANUP] Rimozione ritardi di rete da '{self.tc_interface}'...")
        subprocess.run(
            ["tc", "qdisc", "del", "dev", self.tc_interface, "root"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # ------------------------------------------------------------------ #
    # Misura della latenza percepita                                     #
    # ------------------------------------------------------------------ #

    def _measure_rpc_baseline(self) -> float:
        """
        Esegue un training minimo (1 albero) per misurare il tempo
        di una singola chiamata RPC nelle condizioni correnti di rete.
        Ritorna il tempo in secondi.
        """
        probe_payload = {
            "job_id": f"net_probe_{int(time.time() * 1000)}",
            "dataset_type": self.config.get("dataset_type", "synthetic"),
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": {
                "n_estimators": 1,
                "max_depth": 3,
                "tree_type": self.config["selected_task"],
            },
        }
        t0 = time.perf_counter()
        self.orchestrator._execute_training_step(
            probe_payload, start_alberi=0, target_alberi=1, seed=0
        )
        return time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # run                                                                  #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        net_cfg = self.config.get("network_simulation", {})

        latency_ms = int(net_cfg.get("latency_seconds", 0.0) * 1000)
        loss_percentage = float(net_cfg.get("packet_loss_rate", 0.0) * 100)
        delay_requested = latency_ms > 0 or loss_percentage > 0

        tc_applied = False
        try:
            if delay_requested:
                tc_applied = self._apply_tc_rules(latency_ms, loss_percentage)
                if not tc_applied:
                    print(
                        "[WARNING] tc non disponibile o senza permessi/capability. "
                        "Il test prosegue senza delay reale di rete "
                        "(i tempi misurati non rifletteranno la latenza configurata)."
                    )
            else:
                print("[INFO] latency_seconds=0 e packet_loss_rate=0: nessuna regola tc applicata.")

            # Misura probe RPC singolo (utile per confronto con/senza delay)
            probe_time = self._measure_rpc_baseline()
            print(f"[PROBE] Latenza RPC singola: {probe_time * 1000:.1f}ms")

            # Training reale
            task_type = self.config.get("selected_task", "classifier")
            payload = self._build_payload("network_test")
            n_trees = net_cfg.get("n_estimators_test", 10)

            t0 = time.perf_counter()
            trees_built = self.orchestrator._execute_training_step(
                payload, start_alberi=0, target_alberi=n_trees, seed=42
            )
            duration = time.perf_counter() - t0

            throughput = trees_built / duration if duration > 0 else 0
            accuracy_metrics = self._mock_metrics_and_infer(payload, task_type)

            # Lo stato deve distinguere esplicitamente il caso in cui il delay
            # era richiesto ma non è stato possibile applicarlo: altrimenti un
            # "SUCCESS" può nascondere un test che di fatto non ha simulato nulla.
            if delay_requested and not tc_applied:
                status = "SKIPPED_NO_TC_PERMISSIONS"
            elif trees_built == n_trees:
                status = "SUCCESS"
            else:
                status = "PARTIAL"

            return {
                "scenario_description": "Valutazione dell'impatto dei ritardi e della perdita di pacchetti sulle chiamate RPC.",
                "status": status,
                "applied_latency_ms": latency_ms if tc_applied else 0,
                "applied_loss_percent": loss_percentage if tc_applied else 0,
                "probe_rpc_baseline_ms": round(probe_time * 1000, 2),
                "duration_seconds": duration,
                "tc_rules_successfully_injected": tc_applied,
                "throughput_trees_per_second": throughput,
                "accuracy_metrics": accuracy_metrics
            }

        finally:
            # Il finally garantisce il ripristino anche in caso di eccezione
            if tc_applied:
                self._clear_tc_rules()

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _build_payload(self, tag: str) -> dict:
        net_cfg = self.config.get("network_simulation", {})
        return {
            "job_id": f"test_network_{tag}_{int(time.time() * 1000)}",
            "dataset_type": self.config.get("dataset_type", "synthetic"),
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": {
                "n_estimators": net_cfg.get("n_estimators_test", 10),
                "max_depth": 5,
                "tree_type": self.config["selected_task"],
            },
        }
    
    def _mock_metrics_and_infer(self, payload, task_type):
        # 1. Inizializziamo la variabile per permettere l'uso di nonlocal
        accuracy_metrics = {}
        
        # 2. Estraiamo il job_id dal payload perché ci serve per i percorsi dei file
        job_id = payload.get("job_id")

        def intercept_metrics_centralized(predictions_matrix, y_test, tree_type, **kwargs):
            nonlocal accuracy_metrics
            if tree_type == "classifier":
                # Ricostruzione votazione speculare al codice dell'orchestrator centralizzato
                from sklearn.utils.extmath import weighted_mode
                import numpy as np
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                uniform_weights = np.ones_like(predictions_matrix)
                final_predictions, _ = weighted_mode(predictions_matrix, uniform_weights, axis=0)
                final_predictions = final_predictions.ravel().astype(int)
                y_test = y_test.astype(int)
                
                accuracy_metrics = {
                    "accuracy": float(np.mean(final_predictions == y_test)),
                    "f1_score": float(f1_score(y_test, final_predictions, zero_division=0)),
                    "precision": float(precision_score(y_test, final_predictions, zero_division=0)),
                    "recall": float(recall_score(y_test, final_predictions, zero_division=0))
                }
            else:
                import numpy as np
                final_predictions = np.mean(predictions_matrix, axis=0)
                accuracy_metrics = {"mean_squared_error": float(np.mean((final_predictions - y_test) ** 2))}

        # 3. Rimosso "self" dai parametri, causa errore nel monkey patching su istanza
        def intercept_metrics_federated(y_pred, y_true, tree_type, **kwargs):
            nonlocal accuracy_metrics
            import numpy as np
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            if tree_type == "classifier":
                if np.issubdtype(y_pred.dtype, np.floating):
                    final_predictions = (y_pred >= 0.5).astype(int)
                else:
                    final_predictions = y_pred.astype(int)
                    
                y_true = y_true.astype(int)
                accuracy_metrics = {
                    "accuracy": float(np.mean(final_predictions == y_true)),
                    "f1_score": float(f1_score(y_true, final_predictions, zero_division=0)),
                    "precision": float(precision_score(y_true, final_predictions, zero_division=0)),
                    "recall": float(recall_score(y_true, final_predictions, zero_division=0))
                }
            else:
                accuracy_metrics = {"mean_squared_error": float(np.mean((y_pred.astype(float) - y_true.astype(float)) ** 2))}

        # Sostituzione temporanea (Monkey Patching sicuro per la durata del test)
        orig_centralized = getattr(self.orchestrator, "_print_and_validate_metrics", None)
        orig_federated = getattr(self.orchestrator, "_print_and_validate_metrics_federated", None)
        
        if orig_centralized:
            self.orchestrator._print_and_validate_metrics = intercept_metrics_centralized
        if orig_federated:
            self.orchestrator._print_and_validate_metrics_federated = intercept_metrics_federated

        modello_creato = os.path.join("./saved_models", f"cen_model_{job_id}.pkl")
        if not os.path.exists(modello_creato): # Prova anche la variante federata se applicabile
            modello_creato = os.path.join("./saved_models", f"fed_model_{job_id}.pkl")  
            
        modello_atteso_da_inferenza = os.path.join("./saved_models", f"model_{job_id}.pkl") 
        creato_link_temporaneo = False
        
        if os.path.exists(modello_creato) and not os.path.exists(modello_atteso_da_inferenza):
            try:
                # Creiamo un alias temporaneo così _execute_inference_step trova il file
                os.link(modello_creato, modello_atteso_da_inferenza)
                creato_link_temporaneo = True
            except Exception as link_err:
                print(f"[WARN PERF TEST] Impossibile creare link temporaneo per il modello: {link_err}")
                
        try:
            # Eseguiamo l'inferenza nativa dell'orchestratore
            self.orchestrator._execute_inference_step(payload)
        except Exception as e:
            print(f"[ERROR PERF TEST] Errore durante l'esecuzione dell'inferenza distribuita: {e}")
        finally:
            # Ripristino immediato dei metodi originali
            if orig_centralized: self.orchestrator._print_and_validate_metrics = orig_centralized
            if orig_federated: self.orchestrator._print_and_validate_metrics_federated = orig_federated
            if creato_link_temporaneo and os.path.exists(modello_atteso_da_inferenza):
                try:
                    os.remove(modello_atteso_da_inferenza)
                except:
                    pass

        # Fallback descrittivo in caso di fallimento dell'inferenza
        if not accuracy_metrics:
            print("[WARN PERF TEST] Impossibile estrarre metriche reali dall'inferenza. Verificare i log dei Worker.")
            if task_type == "classifier":
                accuracy_metrics = {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}
            else:
                accuracy_metrics = {"mean_squared_error": 0.0}

        # 4. Ritorniamo SOLO il dizionario delle metriche, come si aspetta _run_locally
        return accuracy_metrics