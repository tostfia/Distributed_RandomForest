import os
import statistics
import subprocess
import time

from src.testing.scenarios.base import BaseTestScenario
from src.testing.scenarios import aws_ecs_utils


class NetworkSimulationScenario(BaseTestScenario):
    """
    Scenario 3: Simulazione/misurazione di ritardi di rete.

    - In locale/Docker: inietta artificialmente un delay via 'tc netem' su
      un'interfaccia (altrimenti la latenza RPC su loopback/bridge Docker
      sarebbe pressoché nulla, quindi non ci sarebbe nulla da misurare).

    - Su AWS/Fargate: NON inietta nulla. Verificato che l'account AWS
      Academy Learner Lab usato in questo progetto non ha accesso ad AWS
      Fault Injection Simulator (aws:fis:ListExperimentTemplates ->
      AccessDeniedException per l'identità 'voclabs/...'), che sarebbe
      stato il modo AWS-native per iniettare latenza/loss reali sui task
      ECS senza CAP_NET_ADMIN. 'tc' diretto non è utilizzabile su Fargate
      per lo stesso motivo (CAP_NET_ADMIN non disponibile nel task).
      Lo scenario diventa quindi una MISURA della latenza RPC reale tra i
      task (leader<->worker, stessa VPC, ENI separate) invece di una
      simulazione: più probe consecutivi per avere min/media/max, poi lo
      stesso training/inferenza reale degli altri scenari. Il risultato
      NON è direttamente comparabile al numero "1.5s" impostato in
      test_config.json per il caso locale (quello è un delay artificiale
      scelto da voi, questo è un tempo osservato); vanno presentati nella
      relazione come due esperimenti diversi, non come stesso esperimento
      su due ambienti.

    NOTA IMPORTANTE SUI PERMESSI (solo rilevante per il ramo locale/Docker):
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

    # Numero di probe RPC consecutivi usati SOLO nel ramo AWS per stimare
    # min/media/max della latenza reale (vedi _measure_rpc_baseline_stats_aws).
    AWS_PROBE_COUNT = 5

    def __init__(self, config, orchestrator):
        super().__init__(config, orchestrator)

        self.aws_env = aws_ecs_utils.is_aws_environment(orchestrator)

        # In Docker la capability CAP_NET_ADMIN viene data al container/binario
        # (docker-compose 'cap_add' + 'setcap' sul binario tc), quindi tc funziona
        # senza sudo. In esecuzione locale (bare metal) questa capability di solito
        # non è presente sul binario, quindi serve sudo per modificare le regole
        # di rete del kernel. Su AWS questo attributo non viene mai usato (vedi
        # run()), lasciato solo per compatibilità con gli helper del ramo locale.
        self.running_in_docker = os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true"

        if self.aws_env:
            # Su AWS il ramo tc non viene mai usato (vedi run(), che fa
            # short-circuit su _run_aws_measurement_only prima di toccare
            # self.tc_interface): il messaggio "Docker" qui sarebbe
            # fuorviante, anche se RUNNING_IN_DOCKER=true è corretto (i
            # worker girano comunque come task Fargate containerizzati).
            # default_interface resta comunque definita (valore inerte) per
            # sicurezza, nel caso questo attributo venga letto altrove in futuro.
            print("[INFO] Esecuzione su AWS/ECS: tc non applicabile (vedi run()), CAP_NET_ADMIN non disponibile su Fargate.")
            default_interface = "eth0"
        elif self.running_in_docker:
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
    # tc helpers (solo locale/Docker)                                    #
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
        Esegue un training minimo (1 albero) per ottenere un tempo di
        riferimento nelle condizioni correnti di rete.

        ATTENZIONE: nonostante il nome storico, questo NON isola la sola
        latenza RPC — _execute_training_step esegue anche l'intero ETL
        (parsing CSV, split, preprocessing) prima di contattare il worker,
        quindi il valore misurato è dominato dal tempo di preparazione dati
        (tipicamente 130-260s), non dalla rete. Per isolare la vera latenza
        RPC servirebbe un metodo "ping" leggero esposto dal worker (es.
        BaseWorker.exposed_ping()) che bypassi del tutto l'ETL — non ancora
        implementato. Ritorna il tempo totale in secondi.
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
        # Costruito a mano, separatamente da _build_payload(): senza questo
        # ricadeva sui DEFAULT SILENZIOSI di federated.py ("iid"/"proportional")
        # invece della strategia realmente dichiarata nel manifesto — osservato
        # in un run reale: i probe di questo metodo dichiaravano
        # tree_allocation='proportional' mentre il job principale dello stesso
        # scenario dichiarava correttamente 'equal' (stessa sessione, stesso
        # manifesto, due valori diversi).
        if os.environ.get("SYS_MODE", "centralized") == "federated":
            probe_payload = self._augment_payload_with_partitioning(probe_payload)
        t0 = time.perf_counter()
        self._reuse_dataset_if_available(probe_payload, seed=0)
        self.orchestrator._execute_training_step(
            probe_payload, start_alberi=0, target_alberi=1, seed=0
        )
        self._mark_job_finished(probe_payload["job_id"], alberi_addestrati=1)
        return time.perf_counter() - t0

    def _measure_rpc_baseline_stats_aws(self, num_probes: int) -> dict:
        """
        Ripete _measure_rpc_baseline() più volte per stimare min/media/max
        della latenza reale osservata tra orchestratore e worker su AWS.

        Ha lo stesso limite documentato in _measure_rpc_baseline: ogni probe
        include comunque l'ETL (short-circuit dopo il primo grazie a
        _reuse_dataset_if_available, quindi dal secondo probe in poi il
        tempo è più rappresentativo del solo RPC+training di 1 albero).
        Per questo il PRIMO probe viene scartato dalle statistiche: include
        il costo ETL "a freddo" e falserebbe min/media verso l'alto.
        """
        samples_ms = []
        for i in range(num_probes):
            t = self._measure_rpc_baseline() * 1000
            print(f"[NETWORK AWS] Probe {i + 1}/{num_probes}: {t:.1f}ms"
                  + (" (scartato dalle statistiche: include ETL a freddo)" if i == 0 else ""))
            samples_ms.append(t)

        stats_samples = samples_ms[1:] if len(samples_ms) > 1 else samples_ms
        return {
            "samples_ms": [round(s, 2) for s in samples_ms],
            "cold_first_probe_excluded_from_stats": len(samples_ms) > 1,
            "min_ms": round(min(stats_samples), 2),
            "max_ms": round(max(stats_samples), 2),
            "avg_ms": round(statistics.mean(stats_samples), 2),
            "median_ms": round(statistics.median(stats_samples), 2),
        }

    # ------------------------------------------------------------------ #
    # run                                                                  #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        if self.aws_env:
            return self._run_aws_measurement_only()

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

            # Ping puro (nessun ETL): con tc_applied=True riflette anche il
            # delay artificiale iniettato sull'interfaccia, a differenza della
            # misura sotto (probe_time) che lo annega nel tempo di ETL.
            pure_ping_stats = self.orchestrator._measure_rpc_ping_stats(num_probes=5)
            if pure_ping_stats["avg_ms"] is not None:
                print(f"[PING] Latenza RPC PURA (nessun ETL): "
                      f"min={pure_ping_stats['min_ms']}ms avg={pure_ping_stats['avg_ms']}ms "
                      f"median={pure_ping_stats['median_ms']}ms max={pure_ping_stats['max_ms']}ms")

            # Misura di riferimento (include ETL, NON è la sola latenza RPC —
            # vedi docstring di _measure_rpc_baseline)
            probe_time = self._measure_rpc_baseline()
            print(f"[PROBE] Tempo totale job di probe (1 albero, ETL incluso): {probe_time * 1000:.1f}ms")

            # Training reale
            task_type = self.config.get("selected_task", "classifier")
            payload = self._build_payload("network_test")
            # Numero di alberi dal manifesto della baseline (vedi
            # BaseTestScenario._resolve_hyperparameters): stessa fonte del payload,
            # quindi non si puo' piu' chiedere N alberi dichiarandone M ai worker.
            target_trees = self._resolve_target_trees()

            t0 = time.perf_counter()
            self._reuse_dataset_if_available(payload, seed=123)
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
                "execution_mode": "local",
                "applied_latency_ms": latency_ms if tc_applied else 0,
                "applied_loss_percent": loss_percentage if tc_applied else 0,
                "pure_rpc_ping_latency_ms": pure_ping_stats,
                "probe_job_total_time_ms": round(probe_time * 1000, 2),
                "duration_seconds": round(duration, 2),
                "tc_rules_successfully_injected": tc_applied,
                "throughput_trees_per_second": round(throughput, 2),
                "accuracy_metrics": accuracy_metrics
            }

        finally:
            # Il finally garantisce il ripristino anche in caso di eccezione
            if tc_applied:
                self._clear_tc_rules()

    def _run_aws_measurement_only(self) -> dict:
        """
        Ramo AWS: nessuna iniezione di delay/loss (CAP_NET_ADMIN non
        disponibile su Fargate, e AWS Fault Injection Simulator non
        accessibile con le credenziali del Learner Lab usato per questo
        progetto: 'aws fis list-experiment-templates' ritorna
        AccessDeniedException per l'identità 'voclabs/...'). Misura invece
        la latenza RPC reale tra i task su più probe, poi esegue comunque
        il training/inferenza reale.
        """
        print("\n--- [SCENARIO 3] Simulazione di Rete: NON applicabile su AWS/ECS (misura reale) ---")
        print("[INFO] CAP_NET_ADMIN non disponibile su Fargate; AWS Fault Injection Simulator "
              "non accessibile con le credenziali AWS Academy di questo progetto (verificato: "
              "'fis:ListExperimentTemplates' -> AccessDeniedException). Nessun delay/loss viene "
              "iniettato: questo scenario misura invece la latenza RPC REALE tra i task "
              "(leader<->worker, stessa VPC) su più probe consecutivi.")

        # Ping puro (nessun ETL, solo round-trip RPyC): isola la vera latenza
        # di rete tra orchestratore e worker, a differenza della misura sotto
        # (probe_stats) che include comunque l'ETL a partire dal secondo probe.
        pure_ping_stats = self.orchestrator._measure_rpc_ping_stats(self.AWS_PROBE_COUNT)
        if pure_ping_stats["avg_ms"] is not None:
            print(f"[NETWORK AWS] Latenza RPC PURA (ping, nessun ETL): "
                  f"min={pure_ping_stats['min_ms']}ms avg={pure_ping_stats['avg_ms']}ms "
                  f"median={pure_ping_stats['median_ms']}ms max={pure_ping_stats['max_ms']}ms")

        probe_stats = self._measure_rpc_baseline_stats_aws(self.AWS_PROBE_COUNT)
        print(f"[NETWORK AWS] Latenza RPC osservata (job di probe, ETL incluso dopo il primo): "
              f"min={probe_stats['min_ms']}ms avg={probe_stats['avg_ms']}ms "
              f"median={probe_stats['median_ms']}ms max={probe_stats['max_ms']}ms")

        task_type = self.config.get("selected_task", "classifier")
        payload = self._build_payload("network_test_aws")
        # Numero di alberi dal manifesto della baseline (vedi
        # BaseTestScenario._resolve_hyperparameters): stessa fonte del payload,
        # quindi non si puo' piu' chiedere N alberi dichiarandone M ai worker.
        target_trees = self._resolve_target_trees()

        t0 = time.perf_counter()
        self._reuse_dataset_if_available(payload, seed=123)
        trees_built = self.orchestrator._execute_training_step(
            payload, start_alberi=0, target_alberi=target_trees, seed=123
        )
        duration = time.perf_counter() - t0
        self._mark_job_finished(payload["job_id"], alberi_addestrati=trees_built)

        throughput = trees_built / duration if duration > 0 else 0
        accuracy_metrics = self._run_inference_and_get_metrics(payload, task_type)

        status = "SUCCESS_REAL_NETWORK_NO_INJECTION" if trees_built == target_trees else "PARTIAL"

        return {
            "scenario_description": (
                "Su AWS non viene iniettato alcun delay/loss artificiale (CAP_NET_ADMIN non "
                "disponibile su Fargate, AWS FIS non accessibile con le credenziali Academy usate). "
                "Lo scenario misura invece la latenza RPC reale tra i task distribuiti sulla VPC "
                "e ne riporta min/media/mediana/max su più probe, poi esegue il training/inferenza "
                "reale. Non comparabile 1:1 col delay artificiale di 1.5s usato nel test locale: "
                "sono due esperimenti diversi (uno simula condizioni avverse, l'altro misura le "
                "condizioni reali dell'infrastruttura)."
            ),
            "status": status,
            "execution_mode": "aws",
            "network_injection_applied": False,
            "network_injection_unavailable_reason": (
                "CAP_NET_ADMIN non disponibile su Fargate; AWS FIS non accessibile "
                "(fis:ListExperimentTemplates -> AccessDeniedException per le credenziali "
                "AWS Academy Learner Lab di questo progetto)."
            ),
            "pure_rpc_ping_latency_ms": pure_ping_stats,
            # Rinominato da 'measured_real_rpc_latency_ms': il nome vecchio
            # suggeriva latenza di rete pura, ma probe_stats include l'ETL
            # (vedi log subito sopra: "job di probe, ETL incluso dopo il
            # primo"). Nome distinto da 'probe_job_total_time_ms' del ramo
            # locale (che è uno scalare) perché qui è un dict di statistiche
            # su più probe (min/avg/median/max), non lo stesso tipo di dato.
            "probe_job_total_time_stats_ms": probe_stats,
            "duration_seconds": round(duration, 2),
            "throughput_trees_per_second": round(throughput, 2),
            "accuracy_metrics": accuracy_metrics,
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _build_payload(self, tag: str) -> dict:
        net_cfg = self.config.get("network_simulation", {})
        # Vedi BaseTestScenario._resolve_hyperparameters: fonte unica condivisa
        # con la baseline locale. NOTA: il payload di probe costruito in
        # _measure_rpc_baseline resta volutamente separato e con 1 solo albero,
        # perché serve a misurare la latenza, non a produrre un modello
        # confrontabile.
        hp = self._resolve_hyperparameters()
        payload = {
            "job_id": f"test_network_{tag}_{int(time.time() * 1000)}",
            "dataset_type": self.config.get("dataset_type", "synthetic"),
            "dataset_path": self.config["dataset_path"],
            "hyperparameters": hp,
        }
        if os.environ.get("SYS_MODE", "centralized") == "federated":
            payload = self._augment_payload_with_partitioning(payload)
        return payload

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