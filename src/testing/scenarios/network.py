import os
import subprocess
import time

from src.testing.scenarios.base import BaseTestScenario



class NetworkSimulationScenario(BaseTestScenario):
    """
    Scenario 3: Simulazione di ritardi di rete tramite tc netem.

    Supporta due modalità, configurabili via test_config.json:

    -  questo scenario applica lui stesso
      il delay su 'lo' via tc, esegue il training RPC, poi ripristina.
      Si usa quando lanci i test direttamente senza run_local.sh."""

    
    def __init__(self, config, orchestrator):
        super().__init__(config, orchestrator)
        self.tc_interface = os.environ.get("TC_INTERFACE", "lo")
       

    # ------------------------------------------------------------------ #
    # tc helpers                                                           #
    # ------------------------------------------------------------------ #

    def _tc_available(self) -> bool:
        """Controlla che tc sia installato e sudo funzioni senza password."""
        result = subprocess.run(
            [ "-n", "tc", "qdisc", "show", "dev", self.tc_interface],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def _apply_tc_rules(self, latency_ms: int, loss_percentage: float) -> bool:
        """
        Applica delay (e opzionalmente packet loss) su 'lo'.
        Ritorna True se il comando è andato a buon fine, False altrimenti
        (es. mancano i permessi sudo).
        """
        print(
            f"\n[tc] Configurazione 'lo': +{latency_ms}ms delay"
            + (f", {loss_percentage:.1f}% loss" if loss_percentage > 0 else "")
        )

        # Rimuove eventuali regole residue per evitare "File exists"
        subprocess.run(
            [ "tc", "qdisc", "del", "dev", "lo", "root"],
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
                f"[tc WARNING] Impossibile applicare le regole "
                f"(sudo senza password configurato?): {result.stderr.strip()}"
            )
            return False

        print(f"[tc OK] Regole applicate su 'lo'.")
        return True

    def _clear_tc_rules(self):
        """Rimuove tutte le regole da 'lo' e ripristina il comportamento normale."""
        print("[tc CLEANUP] Rimozione ritardi di rete da 'lo'...")
        subprocess.run(
            [ "tc", "qdisc", "del", "dev", "lo", "root"],
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
            "job_id": f"net_probe_{int(time.time())}",
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

        # Lo scenario applica tc da solo, misura, poi ripristina.
        
        tc_applied = False
        try:
            if latency_ms > 0 or loss_percentage > 0:
                tc_applied = self._apply_tc_rules(latency_ms, loss_percentage)
                if not tc_applied:
                    print(
                        "[WARNING] tc non disponibile o senza permessi. "
                        "Il test prosegue senza delay reale di rete "
                        "(i tempi misurati non rifletteranno la latenza configurata)."
                    )
            else:
                print("[INFO] latency_seconds=0 e packet_loss_rate=0: nessuna regola tc applicata.")

            # Misura probe RPC singolo (utile per confronto con/senza delay)
            probe_time = self._measure_rpc_baseline()
            print(f"[PROBE] Latenza RPC singola: {probe_time*1000:.1f}ms")

            # Training reale
            payload = self._build_payload("network_test")
            n_trees = net_cfg.get("n_estimators_test", 10)

            t0 = time.perf_counter()
            trees_built = self.orchestrator._execute_training_step(
                payload, start_alberi=0, target_alberi=n_trees, seed=42
            )
            duration = time.perf_counter() - t0

            return {
                "scenario_description": "Valutazione dell'impatto dei ritardi e della perdita di pacchetti sulle chiamate RPC.",
                "status": "SUCCESS" if trees_built == n_trees else "PARTIAL", 
                "applied_latency_ms": latency_ms if tc_applied else 0, 
                "applied_loss_percent": loss_percentage if tc_applied else 0, 
                "probe_rpc_baseline_ms": round(probe_time * 1000, 2), 
                "duration_seconds": duration, 
                "tc_rules_successfully_injected": tc_applied 
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
            "job_id": f"test_network_{tag}_{int(time.time())}",
            "dataset_type": self.config.get("dataset_type", "synthetic"),
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": {
                "n_estimators": net_cfg.get("n_estimators_test", 10),
                "max_depth": 5,
                "tree_type": self.config["selected_task"],
            },
        }