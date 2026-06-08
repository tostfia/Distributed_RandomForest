import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple


class StratifiedDataSplitter:
    """
    Gestisce la logica di partizionamento stratificato del dataset.

    Replica la logica del notebook basata su StratifiedShuffleSplit:
    viene eseguito un singolo split train/test mantenendo la distribuzione
    della variabile target.
    """

    def __init__(
        self,
        target_column: str = "Label",
        test_size: float = 0.1,
        random_state: int = 123
    ):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Esegue la suddivisione in train e test set mantenendo inalterata
        la distribuzione statistica della variabile target.

        Ritorna:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        if df.empty:
            raise ValueError("Il DataFrame in input per lo split è vuoto.")

        if self.target_column not in df.columns:
            raise KeyError(
                f"Impossibile effettuare lo split: la colonna target "
                f"'{self.target_column}' non è presente nel DataFrame. "
                f"Colonne disponibili: {list(df.columns)}"
            )

        print(
            f"\n[Splitter] Esecuzione split stratificato "
            f"(Test: {self.test_size * 100}%, Seed: {self.random_state})..."
        )

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state
        )

        for train_index, test_index in splitter.split(df, df[self.target_column]):
            train_df = df.iloc[train_index].copy()
            test_df = df.iloc[test_index].copy()

        print(f"   -> Train set isolato: {train_df.shape[0]} istanze")
        print(f"   -> Test set isolato:  {test_df.shape[0]} istanze")

        return train_df, test_df