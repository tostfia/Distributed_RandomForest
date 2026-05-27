import os

from src.worker.BaseWorker import BaseWorker
import pandas as pd
class FederatedWorker(BaseWorker):

    _LOCAL_DATASET_PATH = os.environ.get("LOCAL_DATASET_PATH", "/data/private_data.csv")

    def __init__(
            self,
            worker_name: str,
            queue_name: str,
            environment: str,
            url_dataset: str,
            tree_class_reference,
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
                 

    def _load_data(self, source_info):
        print(f"[Federated] Accesso al database locale protetto: {self._LOCAL_DATASET_PATH}")
        df: pd.DataFrame = pd.read_csv(self._LOCAL_DATASET_PATH)
 
        if self.target_column not in df.columns:
            raise ValueError(
                f"Colonna target '{self.target_column}' non trovata nel dataset locale."
            )
 
        y_df = df[self.target_column]
        X_df = df.drop(columns=[self.target_column])
 
        X = X_df.to_numpy(dtype=np.float64)
        y = y_df.to_numpy(dtype=np.float64)
        print(f"[Federated] Dati caricati: X shape = {X.shape}, y shape = {y.shape}")
        return X, y

    def _get_tree_class(self):
        return self.tree_class_reference