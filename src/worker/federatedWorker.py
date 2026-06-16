import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from src.worker.BaseWorker import BaseWorker


class FederatedWorker(BaseWorker):
    """
    Worker per la gestione dell'addestramento in modalità federata.
    Se invocato in modalità sintetica, genera localmente in memoria il proprio frammento
    di dati (ispirandosi a scikit-learn). Altrimenti accede a un dataset locale privato sul nodo.
    """

    def __init__(
        self,
        worker_name: str,
        queue_name: str,
        tree_class_reference: type,
        target_column: str = "Label",
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
        
        # Recuperiamo l'eventuale path per i dati reali dal .env o fallback di sicurezza
        self.local_dataset_path = os.getenv("LOCAL_DATASET_PATH", "/data/private_data.csv")
        
        # Estraiamo un indice numerico univoco dal nome del worker (es. "Worker-0" -> 0) 
        # come fallback iniziale per differenziare i dataset sintetici locali.
        self.worker_index = 0
        for char in worker_name.split("-"):
            if char.isdigit():
                self.worker_index = int(char)
                break

        print(
            f"[FederatedWorker] Inizializzato in ambiente: {self.environment.upper()} — "
            f"Worker Index di fallback: {self.worker_index} — Target: {self.target_column}"
        )

    def is_regression(self) -> bool:
        return self.tree_type == "regressor"

    def _load_data(self, source_info: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Carica i dati per l'addestramento locale.
        Se source_info inizia con 'NATIVE_PARTITIONED', genera in autonomia un dataset sintetico disgiunto.
        """
        
        # --- OPZIONE A: GENERAZIONE SINTETICA LOCALE AUTONOMA ---
        if source_info.startswith("NATIVE_PARTITIONED"):
            
            # Estraiamo l'indice del worker inviato via RPC (es. "NATIVE_PARTITIONED|1" -> idx = 1)
            idx_da_stringa = self.worker_index  # Usiamo il fallback del costruttore se non c'è la pipe
            if "|" in source_info:
                try:
                    idx_da_stringa = int(source_info.split("|")[1])
                except ValueError:
                    pass

            # Generiamo il seed deterministico per questo specifico frammento di nodo
            seed_locale = 42 + idx_da_stringa
            print(f"[FederatedWorker] Rilevato addestramento sintetico locale. Generazione in RAM (Index: {idx_da_stringa}, Seed: {seed_locale})...")
            
            n_samples = 25000  # Dimensione del frammento privato di questo nodo
            n_features = 20
            
            if self.is_regression():
                X, y = make_regression(
                    n_samples=n_samples, n_features=n_features, noise=0.1, random_state=seed_locale
                )
                y = y.astype(np.float64)
            else:
                X, y = make_classification(
                    n_samples=n_samples, n_features=n_features, n_informative=15, n_classes=2, random_state=seed_locale
                )
                y = y.astype(np.int64)
                
            print(f"[FederatedWorker] [OK] Dataset sintetico isolato pronto: X shape = {X.shape}, y shape = {y.shape}")
            return X, y

        # --- OPZIONE B: CARICAMENTO DA FILE REALE SUL DISCO EBS DELLA MACCHINA EC2 ---
        print(f"[FederatedWorker] Accesso al file system locale per dati reali privati: {self.local_dataset_path}")
        if not os.path.exists(self.local_dataset_path):
            raise FileNotFoundError(
                f"Errore critico: Il dataset locale protetto '{self.local_dataset_path}' non esiste."
            )
            
        df = pd.read_csv(self.local_dataset_path)
        if self.target_column not in df.columns:
            raise ValueError(f"Colonna target '{self.target_column}' non trovata nel dataset locale.")

        y_df = df[self.target_column]
        X_df = df.drop(columns=[self.target_column])

        X = X_df.to_numpy(dtype=np.float64)
        y = y_df.to_numpy(dtype=np.float64 if self.is_regression() else np.int64)
        
        print(f"[FederatedWorker] Dati reali caricati dal disco: X shape = {X.shape}, y shape = {y.shape}")
        return X, y

    def _get_tree_class(self) -> type:
        return self.tree_class_reference