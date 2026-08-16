from abc import ABC, abstractmethod

class BaseTestScenario(ABC):
    """Classe base astratta per tutti gli scenari di test."""
    def __init__(self, config: dict, orchestrator):
        self.config = config
        self.orchestrator = orchestrator

    @abstractmethod
    def run(self) -> dict:
        """Esegue lo scenario e restituisce un dizionario con i risultati/metriche."""
        pass

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