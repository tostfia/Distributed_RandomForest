from abc import ABC, abstractmethod
import pandas as pd


class DatasetLoader(ABC):
    """
    Interfaccia comune per le strategie di caricamento/generazione dati.

    Ogni loader deve occuparsi solo della fase di estrazione o generazione
    del dataset e deve restituire un pandas DataFrame.

    """
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass