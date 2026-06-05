import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

class StratifiedDataSplitter:
    """
    Gestisce la logica di partizionamento del dataset.
    """
    def __init__(
        self, 
        target_column: str = "Label", 
        test_size: float = 0.20, 
        random_state: int = 123
    ):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Esegue la suddivisione in Train e Test set mantenendo inalterata 
        la distribuzione statistica della variabile target (Stratificazione).
        
        Ritorna:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        if df.empty:
            raise ValueError("Il DataFrame in input per lo split è vuoto.")
            
        if self.target_column not in df.columns:
            raise KeyError(
                f"Impossibile effettuare lo split: la colonna target '{self.target_column}' "
                f"non è presente nel DataFrame. Colonne disponibili: {list(df.columns)}"
            )

        print(f"\n[Splitter] Esecuzione split stratificato (Test: {self.test_size*100}%, Seed: {self.random_state})...")
        
        train_df, test_df = train_test_split(
            df, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=df[self.target_column]
        )
        
        print(f"   -> Train set isolato: {train_df.shape[0]} istanze")
        print(f"   -> Test set isolato:  {test_df.shape[0]} istanze")
        
        return train_df.copy(), test_df.copy()