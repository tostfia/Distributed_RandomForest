import numpy as np
import pandas as pd
from src.dataset.dataset_dao_factory import DatasetDAOFactory
from src.worker.BaseWorker import BaseWorker


class CentralizedWorker(BaseWorker):
    """Worker per la gestione dell'addestramento in modalità centralizzata.
    
    Carica il dataset tramite il DAO specifico per l'ambiente e separa
    le feature dalla colonna target convertendole in matrici NumPy.
    """

    def __init__(
        self,
        worker_name: str,
        queue_name: str,
        tree_class_reference: type,
        target_column: str,
        max_samples: float = None,
        bootstrap: bool = True,
        tree_type: str = "classifier"
    ):
        # Passiamo i parametri alla classe base (l'ambiente viene letto in automatico dal .env)
        super().__init__(
            worker_name=worker_name,
            queue_name=queue_name,
            tree_class_reference=tree_class_reference,
            max_samples=max_samples,
            bootstrap=bootstrap,
        )
        self.target_column = target_column
        self.tree_type = tree_type
        
        self.dao = DatasetDAOFactory.get_dao()
        
        print(
            f"[CentralizedWorker] Inizializzato in ambiente: {self.environment.upper()} "
            f"con DAO: {type(self.dao).__name__}"
        )

        self._cached_source = None
        self._cached_X = None
        self._cached_y = None
    
    def is_regression(self) -> bool:
        return self.tree_type == "regressor"

    def _load_data(self, source_info: str) -> tuple[np.ndarray, np.ndarray]:
        
        """Carica il dataset centralizzato delegando al DAO e lo trasforma in matrici NumPy.
        Args:
            source_info (str): URL S3 o path locale passato dinamicamente dall'Orchestratore.
        """
        
        if self._cached_source == source_info and self._cached_X is not None and self._cached_y is not None:
            print("[CentralizedWorker] Utilizzo dei dati già caricati in cache.")
            return self._cached_X, self._cached_y
        print(f"[CentralizedWorker] Richiesta di caricamento dati tramite DAO da: {source_info}")
        df: pd.DataFrame = self.dao.load_dataset(source_info)
        
        if self.target_column not in df.columns:
            raise ValueError(
                f"Colonna target '{self.target_column}' non trovata nel dataset."
            )
        
        y_df = df[self.target_column]
        X_df = df.drop(columns=[self.target_column])

        # Conversione esplicita in matrici NumPy stabili per Scikit-Learn
        X = X_df.to_numpy(dtype=np.float64)
        if self.tree_type == "regressor":
            y = y_df.to_numpy(dtype=np.float64)
        else:
            y = y_df.to_numpy(dtype=np.int64)
        
        self._cached_source = source_info
        self._cached_X = X
        self._cached_y = y
        print(
            f"[CentralizedWorker] Dati caricati con successo: "
            f"X shape = {X.shape}, y shape = {y.shape}"
        )
        
        return self._cached_X, self._cached_y

    def _get_tree_class(self) -> type:
        """Restituisce il riferimento alla classe dell'albero (es. DecisionTreeClassifier)."""
        return self.tree_class_reference