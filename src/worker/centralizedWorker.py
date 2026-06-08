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
        environment: str,
        url_dataset: str,
        tree_class_reference: type,
        target_column: str,
        max_samples: float = None,
        bootstrap: bool = True,
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
        self.dao = DatasetDAOFactory.get_dao(self.environment)
        
        print(
            f"[CentralizedWorker] Inizializzato in ambiente: {self.environment} "
            f"con DAO: {type(self.dao).__name__}"
        )

    def _load_data(self, source_info: str) -> tuple[np.ndarray, np.ndarray]:
        """Carica il dataset centralizzato e lo trasforma in matrici NumPy compatibili.

        Args:
            source_info (str): Informazioni o URL sulla sorgente dati da passare al DAO.

        Returns:
            tuple[np.ndarray, np.ndarray]: Matrice delle feature (X, float64) 
                                           e vettore dei target (y, int64).
        """
        df: pd.DataFrame = self.dao.load_dataset(source_info)
        
        if self.target_column not in df.columns:
            raise ValueError(
                f"Colonna target '{self.target_column}' non trovata nel dataset."
            )
        
        y_df = df[self.target_column]
        X_df = df.drop(columns=[self.target_column])

        # Conversione esplicita in matrici NumPy stabili per Scikit-Learn
        X = X_df.to_numpy(dtype=np.float64)
        y = y_df.to_numpy(dtype=np.int64)
        
        print(
            f"[Centralized] Dati convertiti in matrice NumPy: "
            f"X shape = {X.shape}, y shape = {y.shape}"
        )
        return X, y

    def _get_tree_class(self) -> type:
        """Restituisce il riferimento alla classe dell'albero (es. DecisionTreeClassifier)."""
        return self.tree_class_reference
    
    ##Nell'addestramento --> il worker dovrà confermare l'addestramento andato a buon fine su Dynamo Db