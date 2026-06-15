import os
import numpy as np
import pandas as pd

from src.worker.BaseWorker import BaseWorker


class FederatedWorker(BaseWorker):
    """
    Worker per la gestione dell'addestramento in modalità federata.
    
    Accede a un dataset locale protetto memorizzato sul nodo stesso,
    estrae le feature e la colonna target e le converte in matrici NumPy.
    """

    def __init__(
        self,
        worker_name: str,
        queue_name: str,
        tree_class_reference: type,
        target_column: str,
        max_samples: float = 1.0,
        bootstrap: bool = False,
    ):
        # Passiamo i parametri alla classe base (l'ambiente viene gestito internamente dal .env)
        super().__init__(
            worker_name=worker_name,
            queue_name=queue_name,
            tree_class_reference=tree_class_reference,
            max_samples=max_samples,
            bootstrap=bootstrap,
        )
        self.target_column = target_column
        
        # Recuperiamo il path del dataset locale dal file .env tramite config o usiamo il fallback di sicurezza
        # Assicurati di poter mappare o estrarre LOCAL_DATASET_PATH se vuoi sovrascriverlo da .env
        self.local_dataset_path = os.getenv("LOCAL_DATASET_PATH", "/data/private_data.csv")
        
        print(
            f"[FederatedWorker] Inizializzato in ambiente: {self.environment.upper()} — "
            f"Dataset locale protetto: {self.local_dataset_path}"
        )

    def is_regression(self) -> bool:
        # Nel federato puoi decidere se impostarlo dinamicamente o passarlo come parametro.
        # Per impostazione predefinita basata sulle classi di classificazione:
        return "regressor" in self.tree_class_reference.__name__.lower()

    def _load_data(self, source_info: str) -> tuple[np.ndarray, np.ndarray]:
        """Carica il dataset locale protetto in formato CSV.

        Args:
            source_info (str): Informazioni ereditate da BaseWorker. Nel federato 
                               viene ignorato poiché la sorgente è strettamente locale.
        """
        print(f"[FederatedWorker] Accesso al database locale privato: {self.local_dataset_path}")
        
        if not os.path.exists(self.local_dataset_path):
            raise FileNotFoundError(
                f"Errore critico: Il dataset locale protetto '{self.local_dataset_path}' "
                f"non esiste su questo nodo federato."
            )
            
        df: pd.DataFrame = pd.read_csv(self.local_dataset_path)

        if self.target_column not in df.columns:
            raise ValueError(
                f"Colonna target '{self.target_column}' non trovata nel dataset locale."
            )

        y_df = df[self.target_column]
        X_df = df.drop(columns=[self.target_column])

        # Conversione in matrici NumPy stabili per Scikit-Learn
        X = X_df.to_numpy(dtype=np.float64)
        
        if self.is_regression():
            y = y_df.to_numpy(dtype=np.float64)
        else:
            y = y_df.to_numpy(dtype=np.int64)
        
        print(f"[FederatedWorker] Dati caricati localmente con successo: X shape = {X.shape}, y shape = {y.shape}")
        return X, y

    def _get_tree_class(self) -> type:
        """Restituisce il riferimento alla classe dell'albero (es. DecisionTreeClassifier)."""
        return self.tree_class_reference