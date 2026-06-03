from src.dataset.dataset_dao_factory import DatasetDAOFactory
from src.worker.BaseWorker import BaseWorker
import pandas as pd
import numpy as np


class CentralizedWorker(BaseWorker):

    def __init__(self,
        worker_name: str
        queue_name: str,
        environment: str,
        url_dataset: str,
        tree_class_reference,
        target_column: str,
        max_samples=None,
        bootstrap: bool = True,):
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

    def  _load_data(self, source_info):
        df: pd.DataFrame = self.dao.load_dataset(source_info)
        if self.target_column not in df.columns:
            raise ValueError(f"Colonna target '{self.target_column}' non trovata nel dataset.")
        
        y_df = df[self.target_column]
        X_df = df.drop(columns=[self.target_column])

        X = X_df.to_numpy(dtype = np.float64)
        y = y_df.to_numpy(dtype = np.int64)
        print(f"[Centralized]  Dati comvertiti in matrice NumPy: X shape = {X.shape}, y shape = {y.shape}")
        return X,y

    def _get_tree_class(self):

        return self.tree_class_reference

    