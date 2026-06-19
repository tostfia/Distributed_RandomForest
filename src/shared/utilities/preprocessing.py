from typing import List, Optional
import numpy as np
import pandas as pd

class CICIDSPreprocessor:
    """
    Pipeline di Preprocessing specifica per il dataset di network traffic CIC-IDS2018.
    Configurata per emulare specularmente al millimetro la logica e i conteggi di Colab.
    """

    def __init__(
        self,
        target_column: str = "Label",
        drop_metadata_columns: bool = True,
        drop_invalid_rows: bool = True,
        metadata_keywords: Optional[List[str]] = None,
    ):
        self.target_column = target_column
        self.drop_metadata_columns = drop_metadata_columns
        self.drop_invalid_rows = drop_invalid_rows
        
        self.metadata_keywords = metadata_keywords or [
            "timestamp",
            "flow id",
            "ip",
            "port",
            "mac",
        ]

    def binarize_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FASE 1 DI COLAB: Rimozione righe spurie e binarizzazione del target (Benign=0, rest=1).
        Eseguire sul dataset intero prima dello split per evitare crash sulle classi rare.
        """
        print("Pre-binarizzazione Target CIC-IDS2018...")
        df = df.copy()
        
        # 1. Rimozione righe di intestazione spuria
        df = df[~df[self.target_column].isin(['Label', ' Label', 'Label '])].copy()

        # 2. Codifica del Target
        df[self.target_column] = np.where(df[self.target_column] == 'Benign', 0, 1).astype(np.int8)
        
        # Report statistico immediato sul dato totale
        total_records = len(df)
        if total_records > 0:
            count_benign = int((df[self.target_column] == 0).sum())
            count_attack = int((df[self.target_column] == 1).sum())
            pct_benign = (count_benign / total_records) * 100
            pct_attack = (count_attack / total_records) * 100

            print("\n=======================================================")
            print("  SOVRASCRITTURA EFFETTUATA PER CLASSIFICAZIONE BINARIA (PRE-SPLIT)")
            print("=======================================================")
            print(f"  Classe Codificata [0] -> 0 (Benign)   :   {count_benign:,} record ({pct_benign:.2f}%)".replace(',', '.'))
            print(f"  Classe Codificata [1] -> 1 (Attacco)  :   {count_attack:,} record ({pct_attack:.2f}%)".replace(',', '.'))
            print("=======================================================\n")
            
        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FASE 3 E 4 DI COLAB: Rimozione metadati e sanificazione NaN/inf.
        Da eseguire in modo indipendente sulle singole fette (Train e Test) dopo lo split.
        """
        df = df.copy()
        initial_shape = df.shape

        # 1. Rimozione Metadati non generalizzabili (Data Leakage)
        if self.drop_metadata_columns:
            df = self._drop_metadata_columns(df)

        # 2. Cast numerico e Sanificazione finale per i Worker
        df = self._convert_feature_columns_to_numeric(df)

        if self.drop_invalid_rows:
            df = self._drop_invalid_rows(df)

        print(f" • Pulizia completata. Shape: {initial_shape} -> {df.shape}")
        return df

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        return df

    def _drop_metadata_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        columns_to_drop = [
            col for col in df.columns
            if col != self.target_column
            and any(k in str(col).lower() for k in self.metadata_keywords)
        ]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors="ignore")
            print(f"   - Colonne metadata rimosse ({len(columns_to_drop)})")
        return df

    def _convert_feature_columns_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        feature_columns = df.columns.difference([self.target_column])
        df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
        return df

    def _drop_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rows_before = df.shape[0]
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna().reset_index(drop=True)
        removed = rows_before - df.shape[0]
        if removed > 0:
            print(f"   - Righe rimosse per NaN/inf: {removed}")
        return df