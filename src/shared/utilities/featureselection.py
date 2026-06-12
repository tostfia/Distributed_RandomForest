from typing import List
import pandas as pd


class CICIDSFeatureSelector:
    """
    Feature selector per CIC-IDS2018.

    Calcola le feature da eliminare solo sul training set,
    poi applica la stessa trasformazione a train e test.
    """

    def __init__(
        self,
        target_column: str = "Label",
        correlation_threshold: float = 0.05
    ):
        self.target_column = target_column
        self.correlation_threshold = correlation_threshold
        self.columns_to_drop_: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> "CICIDSFeatureSelector":
        if self.target_column not in train_df.columns:
            raise KeyError(f"Target '{self.target_column}' non trovato nel train set.")

        columns_to_drop = []

        # 1. Feature costanti calcolate solo sul train
        variances = train_df.var(numeric_only=True)
        constant_features = variances[variances == 0].index.tolist()

        constant_features = [
            col for col in constant_features
            if col != self.target_column
        ]

        columns_to_drop.extend(constant_features)

        # 2. Feature con bassa correlazione col target calcolate solo sul train
        corr_with_label = (
            train_df
            .corr(numeric_only=True)[self.target_column]
            .drop(self.target_column)
            .abs()
        )

        low_corr_features = corr_with_label[
            corr_with_label < self.correlation_threshold
        ].index.tolist()

        columns_to_drop.extend(low_corr_features)

        # Rimuove duplicati mantenendo ordine
        self.columns_to_drop_ = list(dict.fromkeys(columns_to_drop))

        print(f" • Feature selezionate per la rimozione: {len(self.columns_to_drop_)}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns_to_drop_ is None:
            raise RuntimeError("Devi chiamare fit() prima di transform().")

        return df.drop(columns=self.columns_to_drop_, errors="ignore").copy()

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df)