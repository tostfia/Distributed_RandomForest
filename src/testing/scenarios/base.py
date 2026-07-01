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