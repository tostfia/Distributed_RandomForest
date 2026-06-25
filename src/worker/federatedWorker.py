import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

from src.worker.BaseWorker import BaseWorker
from src.shared.factory import DatasetDAOFactory 


class FederatedWorker(BaseWorker):
    """Worker per la gestione dell'addestramento in modalità federata.

    Interpreta le stringhe sintetiche oppure scarica lo shard reale assegnato
    dall'Orchestratore salvandolo nella cache del disco rigido locale (EBS su AWS).
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
        
        # Gestione asimmetrica della cache locale in base all'ambiente
        if self.environment == "aws":
            self.local_cache_dir = f"/tmp/{worker_name}_cache"
        else:
            self.local_cache_dir = f"./{worker_name}_cache"
            
        os.makedirs(self.local_cache_dir, exist_ok=True)
        
        # Attributi per preservare il testing set locale per l'inferenza federata
        self.X_test = None
        self.y_test = None
        self.local_sample_count = 0

        self.worker_index = 0
        for char in worker_name.split("-"):
            if char.isdigit():
                self.worker_index = int(char)
                break

        print(
            f"[FederatedWorker] Inizializzato in ambiente: {self.environment.upper()} — "
            f"Directory Cache locale: {self.local_cache_dir} — Target: {self.target_column}"
        )

    def is_regression(self) -> bool:
        return self.tree_type == "regressor"
   
    def _load_data(self, dataset_tag: str):
        """Implementazione obbligatoria per la classe base."""
        
        dao = DatasetDAOFactory.get_dao(self.environment)
        
        # Carica dai percorsi standard definiti dallo splitter
        train_path = os.path.join(self.local_cache_dir, "train_shard.csv")
        test_path = os.path.join(self.local_cache_dir, "test_shard.csv")
        
        train_df = dao.load_dataset(train_path)
        self.X_train = train_df.drop(columns=[self.target_column])
        self.y_train = train_df[self.target_column]
        
        test_df = dao.load_dataset(test_path)
        self.X_test = test_df.drop(columns=[self.target_column])
        self.y_test = test_df[self.target_column]
        return True

    def exposed_load_local_shard(self):
        """Metodo RPC per forzare il caricamento."""
        return self._load_data("real")
   
    # ─── PUNTO DI INSERIMENTO: METODO RPC DI TRAINING ───
    
    def exposed_train_local_federated_forest(self, dataset_tag: str, num_trees: int, base_seed: int, max_depth: int = None) -> bytes:
        """Metodo esposto tramite RPC richiesto dall'Orchestratore per avviare l'addestramento.
        
        Chiama il caricamento dinamico dei dati e restituisce gli alberi locali in formato binario.
        """
        try:
            # 1. CHIAMATA A LOAD DATA: Popola X_train, y_train, X_test, y_test in RAM prima del fit
            self._load_data(dataset_tag)
            
            print(f"[{self.worker_name}] Inizio addestramento locale di {num_trees} alberi...")
            local_estimators = []
            
            # 2. Loop di addestramento degli alberi (Logica Bagging tipica di Random Forest)
            for i in range(num_trees):
                current_seed = base_seed + i
                
                # Istanziamo il singolo stimatore (es: DecisionTreeClassifier o DecisionTreeRegressor)
                # ereditato tramite la reference passata nel costruttore base
                tree = self.tree_class_reference(
                    max_depth=max_depth,
                    random_state=current_seed
                )
                
                # Gestione del Bootstrap (campionamento locale con ripetizione)
                if self.bootstrap:
                    n_samples = len(self.X_train)
                    # Genera indici casuali stabili usando il seed dell'albero corrente
                    boot_indices = np.random.RandomState(current_seed).choice(
                        n_samples, size=n_samples, replace=True
                    )
                    X_sampled = self.X_train.iloc[boot_indices]
                    y_sampled = self.y_train.iloc[boot_indices]
                else:
                    X_sampled = self.X_train
                    y_sampled = self.y_train
                
                # Fit dell'albero sullo shard locale (o sui dati sintetici)
                tree.fit(X_sampled, y_sampled)
                local_estimators.append(tree)
                
            print(f"[{self.worker_name}] [OK] Addestrati con successo {len(local_estimators)} alberi.")
            
            # 3. Serializzazione e ritorno dei pesi via socket RPC
            return pickle.dumps(local_estimators)
            
        except Exception as e:
            print(f"[{self.worker_name}] [ERRORE CRITICO TRAINING] Impossibile completare il task locale: {e}")
            raise e

    


    # ─── METODI DI VALIDAZIONE FEDERATA ───

    def exposed_predict_subset_forest(self, serialized_trees: bytes, serialized_X_test: bytes = None) -> bytes:
        """Metodo RPC per effettuare inferenza locale sul proprio test set blindato in RAM."""
        print(f"[{self.worker_name}] [FL-EVAL] Ricevuto modello globale aggregato per validazione distribuita...")
        
        if serialized_X_test is not None:
            X_to_predict = pickle.loads(serialized_X_test)
        else:
            X_to_predict = self.X_test
            
        if X_to_predict is None:
            raise ValueError(f"[{self.worker_name}] Errore: Nessun dataset di test caricato in RAM.")
            
        unpacked_model = pickle.loads(serialized_trees)
        
        if isinstance(unpacked_model, list):
            if self.is_regression():
                rf = RandomForestRegressor(n_estimators=len(unpacked_model))
            else:
                rf = RandomForestClassifier(n_estimators=len(unpacked_model))
                rf.classes_ = np.array([0, 1])
                rf.n_classes_ = 2
                
            rf.estimators_ = unpacked_model
            rf.n_features_in_ = X_to_predict.shape[1]
            rf.n_outputs_ = 1
            y_pred = rf.predict(X_to_predict)
        else:
            y_pred = unpacked_model.predict(X_to_predict)
            
        return pickle.dumps({
            "y_pred": y_pred,
            "y_true": self.y_test,
            "n_samples": self.local_sample_count
        })

   
    def _get_tree_class(self) -> type:
        """Restituisce il riferimento alla classe dell'albero (es. DecisionTreeClassifier)."""
        return self.tree_class_reference


    # ─── METODI DI METADATI PER L'ORCHESTRATORE ───

    def exposed_get_local_y_test(self) -> bytes:
        if self.y_test is None:
            raise ValueError(f"[{self.worker_name}] Errore: Nessun target vector locale y_test in RAM.")
        return pickle.dumps(self.y_test)
    
    def exposed_get_local_sample_count(self) -> int:
        return self.local_sample_count