import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from src.worker.BaseWorker import BaseWorker


class FederatedWorker(BaseWorker):
    """
    Worker per la gestione dell'addestramento in modalità federata.
    Se invocato in modalità sintetica, genera localmente in memoria il proprio frammento
    di dati. Altrimenti accede a un dataset locale privato sul nodo.
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
        super().__init__(
            worker_name=worker_name,
            queue_name=queue_name,
            tree_class_reference=tree_class_reference,
            max_samples=max_samples,
            bootstrap=bootstrap,
        )
        self.target_column = target_column
        self.tree_type = tree_type
        self.local_dataset_path = os.getenv("LOCAL_DATASET_PATH", "/data/private_data.csv")
        
        # Attributi per preservare il testing set locale per l'inferenza federata
        self.X_test = None
        self.y_test = None

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
        Carica i dati per l'addestramento locale e conserva una quota di split
        per la successiva validazione/inferenza distribuita.
        """
        X_raw, y_raw = None, None
        
        # --- OPZIONE A: GENERAZIONE SINTETICA LOCALE AUTONOMA ---
        if source_info.startswith("NATIVE_PARTITIONED"):
            idx_da_stringa = self.worker_index
            if "|" in source_info:
                try:
                    idx_da_stringa = int(source_info.split("|")[1])
                except ValueError:
                    pass

            seed_locale = 42 + idx_da_stringa
            print(f"[FederatedWorker] Generazione sintetica in RAM (Index: {idx_da_stringa}, Seed: {seed_locale})...")
            
            n_samples = 25000  
            n_features = 20
            
            if self.is_regression():
                X_raw, y_raw = make_regression(
                    n_samples=n_samples, n_features=n_features, noise=0.1, random_state=seed_locale
                )
                y_raw = y_raw.astype(np.float64)
            else:
                X_raw, y_raw = make_classification(
                    n_samples=n_samples, n_features=n_features, n_informative=15, n_classes=2, random_state=seed_locale
                )
                y_raw = y_raw.astype(np.int64)

        # --- OPZIONE B: CARICAMENTO DA FILE REALE SUL DISCO ---
        else:
            print(f"[FederatedWorker] Accesso al file system per dati reali: {self.local_dataset_path}")
            if not os.path.exists(self.local_dataset_path):
                raise FileNotFoundError(f"Errore: Il dataset locale '{self.local_dataset_path}' non esiste.")
                
            df = pd.read_csv(self.local_dataset_path)
            if self.target_column not in df.columns:
                raise ValueError(f"Colonna target '{self.target_column}' non trovata.")

            y_df = df[self.target_column]
            X_df = df.drop(columns=[self.target_column])

            X_raw = X_df.to_numpy(dtype=np.float64)
            y_raw = y_df.to_numpy(dtype=np.float64 if self.is_regression() else np.int64)

        # --- DETACHMENT DEL TESTING SET (20%) ---
        # Splittiamo i dati in modo che il Worker memorizzi internamente il suo test set privato
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=0.20, random_state=42, stratify=None if self.is_regression() else y_raw
        )
        
        # Salviamo nello stato dell'oggetto per l'interfaccia di inferenza
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"[FederatedWorker] [SPLIT OK] Dati di Addestramento dedicati: {X_train.shape}")
        print(f"[FederatedWorker] [SPLIT OK] Dati di Validazione trattenuti in RAM: {self.X_test.shape}")
        
        # Restituiamo solo il train set al motore d'addestramento dell'albero della classe base
        return X_train, y_train

    def _get_tree_class(self) -> type:
        return self.tree_class_reference