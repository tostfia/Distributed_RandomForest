from typing import Dict, List
import pandas as pd
import numpy as np


class CICIDSFeatureSelector:
    """
    Feature selector per CIC-IDS2018.
    """

    def __init__(
        self,
        target_column: str = "Label",
        correlation_threshold: float = 0.05
    ):
        self.target_column = target_column
        self.correlation_threshold = correlation_threshold
        self.columns_to_drop_: List[str] = []
        self.feature_summary_: Dict[str, List[str]] = {}

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

        # STAMPA COMPATIBILE COLAB - STEP 1
        print("=" * 60)
        print("   RIMOZIONE FEATURE COSTANTI (QUASI-CONSTANT FEATURES)")
        print("=" * 60)
        print(f"Colonne eliminate ({len(constant_features)}): {constant_features}")
        print("=" * 60)

        # 2. Feature con bassa correlazione col target calcolate solo sul train
        corr_with_label = (
            train_df
            .corr(numeric_only=True)[self.target_column]
            .drop(self.target_column, errors="ignore")
            .abs()
        )

        low_corr_features = corr_with_label[corr_with_label < self.correlation_threshold].index.tolist()

        columns_to_drop.extend(low_corr_features)

        print("=" * 60)
        print("   FILTRAGGIO FEATURE MEDIANTE SOGLIA DI CORRELAZIONE")
        print("=" * 60)
        print(f"Numero feature eliminate: {len(low_corr_features)}")
        print("=" * 60)

        # Rimuove duplicati mantenendo ordine
        self.columns_to_drop_ = list(dict.fromkeys(columns_to_drop))

        tutte_le_feature = [col for col in train_df.columns if col != self.target_column]
        feature_salvate = [col for col in tutte_le_feature if col not in self.columns_to_drop_]

        # "eliminate" resta l'unione di entrambi i criteri (retrocompatibile con
        # chi legge solo quella chiave), ma teniamo separate le due categorie:
        # sono concettualmente diverse. La rimozione per varianza zero è
        # ineccepibile (una feature costante non porta MAI informazione, per
        # nessun modello). Il filtro per bassa correlazione LINEARE, invece, può
        # scartare feature predittive solo in interazione/non linearmente — un
        # Random Forest le sfrutterebbe comunque. Separarle rende l'effetto del
        # secondo filtro misurabile e ispezionabile a parte.
        self.feature_summary_ = {
            "eliminate": self.columns_to_drop_,
            "eliminate_varianza_zero": constant_features,
            "eliminate_bassa_correlazione": low_corr_features,
            "salvate": feature_salvate
        }

        print(f"\n [FeatureSelector] Totale feature univoche contrassegnate per la rimozione: {len(self.columns_to_drop_)}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns_to_drop_ is None:
            raise RuntimeError("Devi chiamare fit() prima di transform().")

        df_transformed = df.drop(columns=self.columns_to_drop_, errors="ignore").copy()
        
        # Estrazione reale delle feature residue direttamente dal dataset aggiornato
        features_attuali = [col for col in df_transformed.columns if col != self.target_column]
        print(f" • Features predittive totali rimaste ({len(features_attuali)}): {features_attuali}")
        print(f" • Dimensione attuale del blocco (X + y): {df_transformed.shape}\n")
        
        return df_transformed

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df)