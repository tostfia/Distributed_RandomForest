from src.shared.binding.serviceregistry import ServiceRegistry
from src.testing.scenarios.base import BaseTestScenario
import os
import time
import random

class ScalabilityScenario(BaseTestScenario):
    """Copre lo Scenario 2: Analisi della Scalabilità e del Throughput al variare dei Worker."""

    def run(self) -> dict:
        print("\n--- [SCENARIO 2] Test di Scalabilità e Throughput ---")

        scal_cfg = self.config.get("scalability_test", {})
        results = {}
        task_type = self.config.get("selected_task", "classifier")
        env = self.orchestrator.environment
        execution_mode = env
        all_active_workers = ServiceRegistry.get_available_workers(env)
        total_available = len(all_active_workers)
        workers_to_test = [w for w in scal_cfg.get("worker_counts_to_test", []) if w <= total_available]
        raw_metrics = {}

        if not workers_to_test:
            print(f"[SCALABILITY {execution_mode.upper()}] [WARN] Nessuna configurazione testabile: "
                  f"worker_counts_to_test={scal_cfg.get('worker_counts_to_test', [])} ma solo "
                  f"{total_available} worker realmente disponibili/registrati "
                  f"(ServiceRegistry.get_available_workers). Su AWS verifica che il desired-count "
                  f"del/dei worker-service copra il valore massimo che vuoi testare.")
            return {
                "scenario_description": (
                    "Strong scaling test NON eseguito: nessuna configurazione di worker_counts_to_test "
                    "risultava <= al numero di worker realmente disponibili."
                ),
                "status": "SKIPPED_NO_TESTABLE_WORKER_COUNT",
                "execution_mode": execution_mode,
                "total_available_workers": total_available,
                "requested_worker_counts": scal_cfg.get("worker_counts_to_test", []),
            }

        # Carico fisso per ogni configurazione di worker, preso dal manifesto
        # della baseline: così lo strong scaling misura lo STESSO lavoro che la
        # baseline ha cronometrato in sequenziale, e lo speedup rispetto a
        # T_seq è confrontabile.
        target_trees = self._resolve_target_trees()

        rng = random.Random(123)
        worker_ids = list(all_active_workers.keys())
        for worker_count in workers_to_test:
            print(f"[SCALABILITY {execution_mode.upper()}] Test con {worker_count} Worker attivi (Mock ServiceRegistry)...")
            sampled_ids = rng.sample(worker_ids, worker_count)
            sampled_workers = {k: all_active_workers[k] for k in sampled_ids}
            original_get_workers = ServiceRegistry.get_available_workers
            ServiceRegistry.get_available_workers = lambda environment: sampled_workers
            payload = self._build_payload(worker_count, target_trees)
            try:
                # TIMING ADDESTRAMENTO
                start_train = time.perf_counter()
                self._reuse_dataset_if_available(payload, seed=123)
                num_trees = self.orchestrator._execute_training_step(payload, start_alberi=0, target_alberi=target_trees, seed=123)
                train_duration = time.perf_counter() - start_train
                self._mark_job_finished(payload["job_id"], alberi_addestrati=num_trees)

                # TIMING INFERENZA
                start_infer = time.perf_counter()
                accuracy_metrics = self._run_inference_and_get_metrics(payload, task_type)
                infer_duration = time.perf_counter() - start_infer

                # Calcolo throughput immediati
                train_throughput = num_trees / train_duration if train_duration > 0 else 0
                num_samples = accuracy_metrics.get("testing_set_size", 0)
                infer_throughput = num_samples / infer_duration if infer_duration > 0 else 0

                # Tempo della sola costruzione degli alberi (vedi
                # CentralizedOrchestrator.last_dispatch_seconds). È la misura
                # corretta per lo strong scaling: 'train_duration' include
                # anche ETL, aggregazione e stima OOB, che sono costi
                # PRATICAMENTE COSTANTI rispetto al numero di worker e quindi
                # comprimono artificialmente lo speedup — con un overhead fisso
                # di pochi secondi, raddoppiare i worker non può mai avvicinarsi
                # al 2x anche se la parte parallela scala perfettamente
                # (è la legge di Amdahl applicata suo malgrado alla misura).
                #
                # 'last_dispatch_seconds' esiste solo su CentralizedOrchestrator
                # in questo momento: FederatedOrchestrator non lo strumenta
                # ancora. PRIMA questo ramo ricadeva silenziosamente su
                # train_duration quando l'attributo mancava — che rende
                # 'train_only_duration' identico a 'train_duration' e fa
                # sembrare una misura valida quella che è solo un placeholder,
                # falsando speedup/efficienza in modo non distinguibile a
                # valle. Ora lo distinguiamo esplicitamente.
                dispatch_seconds_raw = getattr(self.orchestrator, "last_dispatch_seconds", None)
                train_only_instrumented = dispatch_seconds_raw is not None
                train_only = dispatch_seconds_raw if train_only_instrumented else train_duration
                if not train_only_instrumented:
                    print(f"[SCALABILITY {execution_mode.upper()}] [WARN] 'last_dispatch_seconds' non "
                          f"disponibile su questo orchestratore ({worker_count} worker): "
                          f"'training_only_seconds'/speedup 'soli alberi' NON sono misure valide per "
                          f"questa configurazione, sono un placeholder = duration_seconds totale.")
                train_only_throughput = num_trees / train_only if train_only > 0 else 0

                raw_metrics[worker_count] = {
                    "train_duration": train_duration,
                    "train_only_duration": train_only,
                    "train_only_instrumented": train_only_instrumented,
                    "train_throughput": train_throughput,
                    "train_only_throughput": train_only_throughput,
                    "etl_seconds": getattr(self.orchestrator, "last_etl_seconds", 0.0),
                    "aggregation_seconds": getattr(self.orchestrator, "last_aggregation_seconds", 0.0),
                    "oob_seconds": getattr(self.orchestrator, "last_oob_seconds", 0.0),
                    "infer_duration": infer_duration,
                    "infer_throughput": infer_throughput,
                    "num_trees": num_trees,
                    "num_samples": num_samples,
                    "accuracy": accuracy_metrics
                }
            finally:
                ServiceRegistry.get_available_workers = original_get_workers
                # Questo scenario chiama _execute_training_step/_execute_inference_step
                # DIRETTAMENTE, bypassando _process_job (dove normalmente avviene la
                # pulizia post-job): senza questa chiamata esplicita, _trees_cache
                # dell'orchestratore (lista di oggetti DecisionTree VIVI in memoria,
                # non compressi) si accumula per ogni job_id di questo scenario senza
                # mai essere liberata, fino a saturare la memoria del container dopo
                # poche configurazioni di worker. Non tocca modello/metriche/dataset
                # già salvati su S3 (path separati), solo stato transitorio/di resume
                # ormai inutile a job concluso.
                try:
                    self.orchestrator._clean_checkpoint(payload["job_id"])
                except Exception as e_clean:
                    print(f"[SCALABILITY {execution_mode.upper()}] [WARN] Pulizia checkpoint per "
                          f"'{payload['job_id']}' fallita (non bloccante): {e_clean}")
        # ─── FASE 2: CALCOLO SPEEDUP E STAMPA IN MODO ELEGANTE ───
        baseline_w = min(workers_to_test)
        base_train_time = raw_metrics[baseline_w]["train_duration"]
        base_train_only_time = raw_metrics[baseline_w]["train_only_duration"]
        base_infer_time = raw_metrics[baseline_w]["infer_duration"]

        print("\n" + "="*80)
        print(f"   REPORT DI SCALABILITÀ COMPLETO (Baseline di riferimento: {baseline_w} Worker)")
        print("="*80)

        for worker_count in workers_to_test:
            m = raw_metrics[worker_count]

            # Formula dello Speedup: Tempo con 1 Worker (o baseline) / Tempo con N Worker
            train_speedup = base_train_time / m["train_duration"] if m["train_duration"] > 0 else 1.0
            # Speedup della sola parte parallelizzabile: è quello che dice
            # quanto bene scala l'architettura, senza che l'overhead costante
            # (ETL, aggregazione, OOB) lo appiattisca.
            train_only_speedup = (base_train_only_time / m["train_only_duration"]
                                  if m["train_only_duration"] > 0 else 1.0)
            # Efficienza parallela = speedup / numero di worker: 1.0 significa
            # scaling ideale, valori bassi indicano che l'overhead per worker
            # sta mangiando il guadagno.
            train_only_efficiency = train_only_speedup / worker_count if worker_count > 0 else 0.0
            infer_speedup = base_infer_time / m["infer_duration"] if m["infer_duration"] > 0 else 1.0

            # Stampa a schermo strutturata
            print(f"\n[Configurazione: {worker_count} Worker]")
            print(f"    ADDESTRAMENTO ({m['num_trees']} alberi complessivi):")
            print(f"     • Durata totale:        {m['train_duration']:.2f} s "
                  f"(di cui ETL {m['etl_seconds']:.2f} s, aggregazione {m['aggregation_seconds']:.2f} s, "
                  f"OOB {m['oob_seconds']:.2f} s)")
            print(f"     • Durata soli alberi:   {m['train_only_duration']:.2f} s"
                  + ("" if m["train_only_instrumented"] else "  [PLACEHOLDER: non strumentato, = durata totale]"))
            print(f"     • Throughput:           {m['train_throughput']:.2f} alberi/s "
                  f"({m['train_only_throughput']:.2f} alberi/s sui soli alberi)")
            print(f"     • Speedup totale:       {train_speedup:.2f}x")
            print(f"     • Speedup soli alberi:  {train_only_speedup:.2f}x  "
                  f"(efficienza parallela {train_only_efficiency:.2f})"
                  + ("" if m["train_only_instrumented"] else "  [NON VALIDO: vedi placeholder sopra]"))

            print(f"   INFERENZA :")
            print(f"     • Durata:     {m['infer_duration']:.2f} secondi")
            print(f"     • Speedup:    {infer_speedup:.2f}x")
            if task_type == "classifier":
                print(f"     • Metric:     Accuracy = {m['accuracy'].get('accuracy', 0.0)*100:.2f}%")
            else:
                print(f"     • Metric:     MSE = {m['accuracy'].get('mse', 0.0):.4f}")

            # Salvataggio nel dizionario di output finale richiesto dall'orchestratore
            results[f"workers_{worker_count}"] = {
                "training": {
                    "duration_seconds": round(m["train_duration"], 2),
                    "training_only_seconds": round(m["train_only_duration"], 2),
                    "training_only_instrumented": m["train_only_instrumented"],
                    "etl_seconds": round(m["etl_seconds"], 2),
                    "aggregation_seconds": round(m["aggregation_seconds"], 2),
                    "oob_estimation_seconds": round(m["oob_seconds"], 2),
                    "throughput_trees_per_s": round(m['train_throughput'], 2),
                    "throughput_trees_per_s_training_only": round(m['train_only_throughput'], 2),
                    "speedup": round(train_speedup, 2),
                    "speedup_training_only": round(train_only_speedup, 2),
                    "parallel_efficiency_training_only": round(train_only_efficiency, 3)
                },
                "inference": {
                    "duration_seconds": round(m["infer_duration"], 2),
                    "throughput_samples_per_s": round(m['infer_throughput'], 2),
                    "speedup": round(infer_speedup, 2)
                },
                "accuracy_metrics": m["accuracy"]
            }

        print("\n" + "="*80)

        # In federato ogni worker valida il proprio shard: il testing_set_size
        # totale CRESCE con il numero di worker testati (non è lo stesso
        # carico rivalutato più volte, come in centralized). Lo speedup
        # dell'inferenza confronta quindi tempi su MOLI DI LAVORO diverse tra
        # una configurazione e l'altra: non è strong scaling, è un artefatto.
        # Il throughput (samples/s) resta invece confrontabile.
        federated_mode = os.environ.get("TRAINING_MODE", "centralized") == "federated"
        inference_speedup_note = (
            "In modalità federata ogni worker valida il proprio shard: il testing_set_size totale "
            "cresce con il numero di worker (vedi 'num_samples' per configurazione). "
            "'inference.speedup' qui NON è comparabile tra configurazioni (confronta moli di lavoro "
            "diverse, non lo stesso carico su più worker): usare 'throughput_samples_per_s' per il "
            "confronto reale."
        ) if federated_mode else None

        return {
            "scenario_description": (
                f"Strong scaling test completato per Addestramento ed Inferenza in ambiente {execution_mode}. "
                f"Carico fisso di {target_trees} alberi per ciascuna configurazione di worker testata."
            ),
            "execution_mode": execution_mode,
            "scaling_type": "strong",
            "baseline_worker_count": baseline_w,
            "total_available_workers": total_available,
            "inference_speedup_caveat": inference_speedup_note,
            "metrics_per_scale": results
        }

    def _build_payload(self, worker_count, target_trees):
        # Vedi BaseTestScenario._resolve_hyperparameters: fonte unica condivisa
        # con la baseline locale.
        hp = self._resolve_hyperparameters()
        # _resolve_hyperparameters() legge SOLO la sotto-chiave 'hyperparameters'
        # del manifesto. n_samples/n_features/noise vivono a livello radice,
        # sibling di 'hyperparameters': senza questa unione esplicita non
        # arrivano mai nel payload di test.
        dataset_shape = self._resolve_dataset_shape()
        hp = {**hp, **{k: v for k, v in dataset_shape.items() if v is not None}}
        if hp.get("n_estimators") != target_trees:
            hp = dict(hp)
            hp["n_estimators"] = target_trees
        payload = {
            "job_id": f"test_scal_{worker_count}_{int(time.time())}",
            "dataset_type": self.config.get("dataset_type", "csv"),
            "dataset_path": self.config.get("dataset_path", ""),
            "hyperparameters": hp,
        }
        if os.environ.get("TRAINING_MODE", "centralized") == "federated":
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
            import traceback
            traceback.print_exc()
            print(f"[ERROR PERF TEST] Errore durante l'esecuzione dell'inferenza distribuita: {e}")

        # Fallback descrittivo in caso di fallimento dell'inferenza
        if not accuracy_metrics:
            print("[WARN PERF TEST] Impossibile estrarre metriche reali dall'inferenza. Verificare i log dei Worker.")
            if task_type == "classifier":
                accuracy_metrics = {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}
            else:
                accuracy_metrics = {"mean_squared_error": 0.0}

        return accuracy_metrics