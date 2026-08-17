from abc import ABC, abstractmethod
import os
import shutil

class BaseTestScenario(ABC):
    """Classe base astratta per tutti gli scenari di test."""

    # Cache A LIVELLO DI CLASSE (condivisa tra TUTTE le istanze di scenario
    # create nello stesso processo dell'engine, es. quando si sceglie 'all').
    # Chiave: firma dei parametri che determinano il CONTENUTO del dataset
    # preprocessato (path, tipo, tipo di albero, seed). Valore: (train_path,
    # test_path) del primo scenario che li ha prodotti.
    #
    # NOTA: questo NON tocca in alcun modo il codice dell'orchestratore. Sfrutta
    # solo il fatto che l'orchestratore già cerca da sé, prima di rifare l'ETL,
    # un file 'shared_train_{job_id}.csv'/'shared_test_{job_id}.csv' — se quel
    # file esiste già quando lo scenario chiama _execute_training_step, il suo
    # normale SHORT-CIRCUIT ETL interno lo trova e salta il preprocessing da
    # solo. Qui ci limitiamo a "precopiare" un dataset già pronto (prodotto da
    # uno scenario precedente con parametri IDENTICI) nel path che
    # l'orchestratore si aspetterà per il prossimo job_id.
    _dataset_cache: dict = {}

    def __init__(self, config: dict, orchestrator):
        self.config = config
        self.orchestrator = orchestrator

    @abstractmethod
    def run(self) -> dict:
        """Esegue lo scenario e restituisce un dizionario con i risultati/metriche."""
        pass

    def _dataset_signature(self, payload: dict, seed: int) -> tuple:
        hp = payload.get("hyperparameters", {})
        return (
            payload.get("dataset_path", ""),
            payload.get("dataset_type", ""),
            hp.get("tree_type", "classifier"),
            seed,
        )

    def _reuse_dataset_if_available(self, payload: dict, seed: int = 123):
        """
        Da chiamare SEMPRE subito prima di orchestrator._execute_training_step().

        Se uno scenario precedente in questa sessione ha già preparato un
        dataset con parametri di ETL identici (stesso dataset_path, tipo,
        tipo di albero e seed), copia quei file già pronti nel path che
        l'orchestratore si aspetta per il job_id corrente. L'orchestratore
        troverà così il file già presente e salterà da solo l'intera fase di
        preprocessing (che può costare 130-260s), grazie al suo meccanismo di
        SHORT-CIRCUIT ETL già esistente — nessuna modifica al suo codice.

        Se invece è la prima volta che si vede questa combinazione di
        parametri, registra semplicemente il path che AVRÀ questo job_id una
        volta completata la sua ETL, così il PROSSIMO scenario con parametri
        identici potrà riusarlo.

        Applicabile solo in ambiente 'local' (storage su filesystem locale
        via bind mount): su AWS il DAO usa S3 e questa scorciatoia lato-test
        non si applica (l'ETL verrebbe comunque rifatta ad ogni scenario).
        """
        environment = getattr(self.orchestrator, "environment", "local")
        if environment != "local":
            return

        job_id = payload.get("job_id")
        if not job_id:
            return

        signature = self._dataset_signature(payload, seed)
        expected_train = f"./.local_storage/shared_train_{job_id}.csv"
        expected_test = f"./.local_storage/shared_test_{job_id}.csv"

        cached = BaseTestScenario._dataset_cache.get(signature)
        if cached:
            ref_train, ref_test = cached
            if os.path.exists(ref_train) and os.path.exists(ref_test):
                try:
                    os.makedirs("./.local_storage", exist_ok=True)
                    shutil.copyfile(ref_train, expected_train)
                    shutil.copyfile(ref_test, expected_test)
                    print(f"[TEST CACHE] Dataset riusato da uno scenario precedente con "
                          f"parametri ETL identici (seed={seed}): ETL saltata per questo job.")
                except Exception as e:
                    print(f"[TEST CACHE WARN] Copia del dataset in cache fallita, "
                          f"procedo con ETL normale: {e}")
                return

        # Prima apparizione di questa combinazione di parametri: registriamo
        # SOLO il path futuro. Il file non esiste ancora (l'ETL vera e propria
        # avverrà dentro _execute_training_step subito dopo questa chiamata),
        # ma da qui in avanti qualunque altro scenario con la stessa firma lo
        # troverà già pronto sul disco al momento in cui gli servirà.
        BaseTestScenario._dataset_cache[signature] = (expected_train, expected_test)

    def _mark_job_finished(self, job_id: str, alberi_addestrati: int = 0):
        """
        Da chiamare DOPO ogni invocazione diretta di _execute_training_step o
        _execute_inference_step (che, a differenza di _process_job, non finalizzano
        mai lo stato del job). Senza questa chiamata il job resta per sempre in
        "PROCESSING": una sessione di test futura lo troverebbe via
        _perform_active_recovery() e lo "riprenderebbe" da capo con iperparametri
        di default, sprecando lavoro dei worker su un job estraneo e ritardando
        il test reale. Fallisce in modo silenzioso (solo un warning) per non far
        crashare lo scenario se lo state_manager non è disponibile o l'update fallisce.
        """
        state_manager = getattr(self.orchestrator, "state_manager", None)
        if not state_manager:
            return
        try:
            state_manager.update_request_status(
                job_id=job_id,
                status="COMPLETED",
                orchestrator_id=getattr(self.orchestrator, "orchestrator_name", "TEST-SCENARIO"),
                retries=0,
                base_random_state=0,
                alberi_addestrati=alberi_addestrati,
            )
        except Exception as e:
            print(f"[TEST WARN] Impossibile finalizzare lo stato del job {job_id[:8]}: {e}")