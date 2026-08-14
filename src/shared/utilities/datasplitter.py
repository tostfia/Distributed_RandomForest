import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
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

        # StratifiedShuffleSplit richiede almeno 2 esempi per classe (uno per
        # train, uno per test). Con dataset fortemente sbilanciati e/o dopo un
        # campionamento aggressivo (es. sample_fraction basso), può capitare che
        # una classe rara collassi a 0/1 esempio: la isoliamo esplicitamente
        # invece di far crashare sklearn con un errore poco leggibile, e la
        # mandiamo per intero nel train set (non ha senso valutarla su un test
        # set quando esiste una sola osservazione storica).
        class_counts = df[self.target_column].value_counts()
        rare_classes = class_counts[class_counts < 2].index.tolist()

        if rare_classes:
            rare_counts = class_counts.loc[rare_classes].to_dict()
            print(
                f"   [ATTENZIONE] {len(rare_classes)} classe/i con meno di 2 esempi "
                f"non possono partecipare allo split stratificato: {rare_counts}. "
                f"Finiranno interamente nel train set."
            )
            rare_mask = df[self.target_column].isin(rare_classes)
            rare_df = df[rare_mask]
            df_stratifiable = df[~rare_mask]
        else:
            rare_df = df.iloc[0:0]
            df_stratifiable = df

        if df_stratifiable[self.target_column].nunique() < 2:
            raise ValueError(
                "Impossibile eseguire lo split stratificato: dopo aver escluso le "
                "classi con meno di 2 esempi, ne resta al massimo 1 classe "
                "stratificabile. Aumenta 'sample_fraction' o verifica il dataset sorgente."
            )

        sss =  StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_df, test_df = None, None
        for train_index, test_index in sss.split(df_stratifiable, df_stratifiable[self.target_column]):
            train_df = df_stratifiable.iloc[train_index]
            test_df = df_stratifiable.iloc[test_index]

        if not rare_df.empty:
            train_df = pd.concat([train_df, rare_df])

        print(f"   -> Train set isolato: {train_df.shape[0]} istanze")
        print(f"   -> Test set isolato:  {test_df.shape[0]} istanze")
        
        return train_df.copy(), test_df.copy()