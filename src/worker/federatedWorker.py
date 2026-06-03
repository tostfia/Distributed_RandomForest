import os
import numpy as np  # <--- CORRETTO: Importato numpy (prima causava NameError)
import pandas as pd
from src.worker.BaseWorker import BaseWorker


class FederatedWorker(BaseWorker):
    """Worker per la gestione dell'addestramento in modalità federata.
    
    Accede a un dataset locale protetto memorizzato sul nodo stesso,
    estrae le feature e la colonna target e le converte in matrici NumPy.
    """

    _LOCAL_DATASET_PATH = os.environ.get("LOCAL_DATASET_PATH", "/data/private_data.csv")

    def __init__(
        self,
        worker_name: str,
        queue_name: str,
        environment: str,
        url_dataset: str,
        tree_class_reference: type,
        target_column: str,
        max_samples: float = 1.0,
        bootstrap: bool = False,
    ):
        super().__init__(
            worker_name=worker_name,
            queue_name=queue_name,
            environment=environment,
            url_dataset=url_dataset,
            tree_class_reference=tree_class_reference,
            max_samples=max_samples,
            bootstrap=bootstrap,
        )
        self.target_column = target_column
        
        print(
            f"[FederatedWorker] Inizializzato — dataset locale: {self._LOCAL_DATASET_PATH}"
        )

    def _load_data(self, source_info: str) -> tuple[np.ndarray, np.ndarray]:
        """Carica il dataset locale protetto in formato CSV.

        Args:
            source_info (str): Informazioni sulla sorgente (ereditato da BaseWorker, 
                               non usato direttamente nel federato).

        Returns:
            tuple[np.ndarray, np.ndarray]: Matrice delle feature (X, float64) 
                                           e vettore dei target (y, int64).
        """
        print(f"[Federated] Accesso al database locale protetto: {self._LOCAL_DATASET_PATH}")
        df: pd.DataFrame = pd.read_csv(self._LOCAL_DATASET_PATH)

        if self.target_column not in df.columns:
            raise ValueError(
                f"Colonna target '{self.target_column}' non trovata nel dataset locale."
            )

        y_df = df[self.target_column]
        X_df = df.drop(columns=[self.target_column])

        # Conversione in matrici NumPy stabili per Scikit-Learn
        X = X_df.to_numpy(dtype=np.float64)
        y = y_df.to_numpy(dtype=np.int64)
        
        print(f"[Federated] Dati caricati: X shape = {X.shape}, y shape = {y.shape}")
        return X, y

    def _get_tree_class(self) -> type:
        """Restituisce il riferimento alla classe dell'albero (es. DecisionTreeClassifier)."""
        return self.tree_class_reference