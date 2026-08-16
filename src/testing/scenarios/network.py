import os
import subprocess
import time

from src.testing.scenarios.base import BaseTestScenario


class NetworkSimulationScenario(BaseTestScenario):
    """
    Scenario 3: Simulazione di ritardi di rete tramite tc netem.

    Lo scenario applica lui stesso il delay su un'interfaccia via tc, esegue il
    training RPC, poi ripristina le regole originali dell'interfaccia.

    NOTA IMPORTANTE SUI PERMESSI:
    tc richiede la capability Linux CAP_NET_ADMIN.

    - In Docker (RUNNING_IN_DOCKER=true): la capability va data al container
      via 'cap_add: NET_ADMIN' nel docker-compose.yml, e/o al binario tc via
      'setcap cap_net_admin+ep' nel Dockerfile. In questo caso i comandi tc
      vengono eseguiti senza sudo.

    - In locale/bare metal (RUNNING_IN_DOCKER=false): il binario di sistema
      di solito non ha la file capability, quindi i comandi tc vengono
      eseguiti con 'sudo -n' (non interattivo). Serve una regola NOPASSWD
      in /etc/sudoers per tc, oppure lanciare l'intero engine con sudo.
      Senza questo, lo scenario prosegue comunque ma senza applicare un
      delay di rete reale (status: SKIPPED_NO_TC_PERMISSIONS).
    """

    def __init__(self, config, orchestrator):
        super().__init__(config, orchestrator)
        
        
        # In Docker la capability CAP_NET_ADMIN viene data al container/binario
        # (docker-compose 'cap_add' + 'setcap' sul binario tc), quindi tc funziona
        # senza sudo. In esecuzione locale (bare metal) questa capability di solito
        # non è presente sul binario, quindi serve sudo per modificare le regole
        # di rete del kernel.
        self.running_in_docker = os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true"

        if self.running_in_docker:
            print("[INFO] Esecuzione in Docker: tc senza sudo (capability CAP_NET_ADMIN).")
            default_interface = "eth0"
        else:
            print("[INFO] Esecuzione locale: tc con sudo (richiesta capability CAP_NET_ADMIN).")
            # Il traffico RPC passa su localhost, non sull'interfaccia fisica/virtuale:
            # 'lo' è il default corretto per riflettere davvero il delay sulle chiamate RPC.
            default_interface = "lo"

        self.tc_interface = os.environ.get("TC_INTERFACE", default_interface)

    def _tc_cmd(self, *args) -> list:
        """Costruisce il comando tc, anteponendo sudo se non siamo in Docker."""
        base = ["tc", *args]
        return base if self.running_in_docker else ["sudo", "-n", *base]

    # ------------------------------------------------------------------ #
    # tc helpers                                                          #
    # ------------------------------------------------------------------ #

    def _tc_available(self) -> bool:
        """
        Controlla che tc sia installato e i permessi siano sufficienti.

        In Docker: tc deve avere la capability CAP_NET_ADMIN attaccata al
        binario stesso via 'setcap cap_net_admin+ep' (fatto in fase di build
        nel Dockerfile), e il container deve avere 'cap_add: NET_ADMIN' nel
        bounding set (docker-compose.yml).

        In locale (bare metal): usiamo 'sudo -n' (non-interattivo). Se non è
        configurato un NOPASSWD per tc in /etc/sudoers, il comando fallisce
        silenziosamente e lo scenario procede senza applicare il delay reale.
        """
        result = subprocess.run(
            self._tc_cmd("qdisc", "show", "dev", self.tc_interface),
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
            self._tc_cmd("qdisc", "del", "dev", self.tc_interface, "root"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        cmd = self._tc_cmd(
            "qdisc", "add", "dev", self.tc_interface,
            "root", "netem", "delay", f"{latency_ms}ms",
        )
        if loss_percentage > 0:
            cmd += ["loss", f"{loss_percentage:.2f}%"]

        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            if self.running_in_docker:
                causa = (
                    "manca 'cap_add: NET_ADMIN' nel docker-compose.yml per questo servizio, "
                    "oppure il binario tc non ha la file capability "
                    "(verifica nel Dockerfile: 'setcap cap_net_admin+ep /usr/sbin/tc')."
                )
            else:
                causa = (
                    "esecuzione locale senza NOPASSWD sudo per tc. Aggiungi una regola in "
                    "/etc/sudoers (es. 'tuo_utente ALL=(ALL) NOPASSWD: /usr/sbin/tc'), "
                    "oppure esegui l'intero engine con 'sudo python -m ...'."
                )
            print(
                f"[tc WARNING] Impossibile applicare le regole tc: {result.stderr.strip()}\n"
                f"[tc WARNING] Causa probabile: {causa}"
            )
            return False

        print(f"[tc OK] Regole applicate su '{self.tc_interface}'.")
        return True

    def _clear_tc_rules(self):
        """Rimuove tutte le regole dall'interfaccia e ripristina il comportamento normale."""
        print(f"[tc CLEANUP] Rimozione ritardi di rete da '{self.tc_interface}'...")
        subprocess.run(
            self._tc_cmd("qdisc", "del", "dev", self.tc_interface, "root"),
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
        self._mark_job_finished(probe_payload["job_id"], alberi_addestrati=1)
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
            if task_type == "classifier":
                target_trees = self.config.get("hyperparameters_class", {}).get("n_estimators", 30)
            else:
                target_trees = self.config.get("hyperparameters_regre", {}).get("n_estimators", 100)

            t0 = time.perf_counter()
            trees_built = self.orchestrator._execute_training_step(
                payload, start_alberi=0, target_alberi=target_trees, seed = 123
            )
            duration = time.perf_counter() - t0
            self._mark_job_finished(payload["job_id"], alberi_addestrati=trees_built)

            throughput = trees_built / duration if duration > 0 else 0
            accuracy_metrics = self._run_inference_and_get_metrics(payload, task_type)

            # Lo stato deve distinguere esplicitamente il caso in cui il delay
            # era richiesto ma non è stato possibile applicarlo: altrimenti un
            # "SUCCESS" può nascondere un test che di fatto non ha simulato nulla.
            if delay_requested and not tc_applied:
                status = "SKIPPED_NO_TC_PERMISSIONS"
            elif trees_built == target_trees:
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
        if self.config.get("selected_task") == "classifier":
            hp = self.config.get("hyperparameters_class", {})
        else:
            hp = self.config.get("hyperparameters_regre", {})
        return {
            "job_id": f"test_network_{tag}_{int(time.time() * 1000)}",
            "dataset_type": self.config.get("dataset_type", "synthetic"),
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": hp,
        }
    
    def _run_inference_and_get_metrics(self, payload, task_type):
        """
        Esegue l'inferenza nativa dell'orchestratore e legge le metriche reali
        dal suo valore di ritorno (sia centralized.py che federated.py restituiscono
        {"metrics": {...}, "testing_set_size": ..., ...} da _execute_inference_step).
        Il modello è già salvato dal training precedente esattamente al path atteso
        da _resolve_model_path (./saved_models/model_{job_id}.pkl in entrambe le
        modalità): non serve nessun link/alias temporaneo.
        """
        accuracy_metrics = {}
        try:
            result = self.orchestrator._execute_inference_step(payload) or {}
            accuracy_metrics = dict(result.get("metrics", {}))
            accuracy_metrics["testing_set_size"] = result.get("testing_set_size", 0)
        except Exception as e:
            print(f"[ERROR PERF TEST] Errore durante l'esecuzione dell'inferenza distribuita: {e}")

        # Fallback descrittivo in caso di fallimento dell'inferenza
        if not accuracy_metrics:
            print("[WARN PERF TEST] Impossibile estrarre metriche reali dall'inferenza. Verificare i log dei Worker.")
            if task_type == "classifier":
                accuracy_metrics = {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}
            else:
                accuracy_metrics = {"mean_squared_error": 0.0}

        return accuracy_metrics