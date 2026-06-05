from typing import List, Optional

import numpy as np
import pandas as pd


class CICIDSPreprocessor:
    """
    Pipeline di Preprocessing specifica per il dataset di network traffic CIC-IDS2018.

    Responsabilità Architetturale:
    - Standardizzare nomi colonne e target.
    - Rimuovere header spuri e prevenire il Data Leakage (metadati).
    - Convertire tipi numerici e gestire i NaN/inf.
    - Binarizzare il target (0 = Benign, 1 = Attack).
    - Eseguire Feature Selection statica (Varianza = 0 e Correlazione < 0.05).
    """

    def __init__(
        self,
        target_column: str = "Label",
        drop_metadata_columns: bool = True,
        drop_invalid_rows: bool = True,
        correlation_threshold: float = 0.05,
        metadata_keywords: Optional[List[str]] = None,
    ):
        self.target_column = target_column
        self.drop_metadata_columns = drop_metadata_columns
        self.drop_invalid_rows = drop_invalid_rows
        self.correlation_threshold = correlation_threshold
        
        # Parole chiave legate all'infrastruttura di rete
        self.metadata_keywords = metadata_keywords or [
            "timestamp",
            "flow id",
            "ip",
            "port",
            "mac",
        ]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Esegue l'intera pipeline di trasformazione sequenziale.
        """
        print("Avvio preprocessing CIC-IDS2018...")
        initial_shape = df.shape

        # 1. Pulizia Strutturale
        df = self._standardize_columns(df)
        df = self._standardize_target_column(df)
        df = self._remove_repeated_header_rows(df)

        # 2. Codifica del Target
        df = self._encode_binary_labels(df)

        # 3. Prevenzione Data Leakage
        if self.drop_metadata_columns:
            df = self._drop_metadata_columns(df)

        # 4. Cast numerico e Sanificazione
        df = self._convert_feature_columns_to_numeric(df)

        if self.drop_invalid_rows:
            df = self._drop_invalid_rows(df)

        # 5. Ottimizzazione e Dimensionality Reduction
        df = self._remove_constant_features(df)
        df = self._remove_low_correlation_features(df)

        print("\n[OK] Preprocessing completato.")
        print(f" • Shape iniziale:    {initial_shape}")
        print(f" • Shape finale:      {df.shape}")
        print(" • Target codificato: 0 = Benign, 1 = Attack")

        return df

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        return df

    def _standardize_target_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        lower_to_original = {str(col).lower(): col for col in df.columns}

        if self.target_column in df.columns:
            return df
        if self.target_column.lower() in lower_to_original:
            original_name = lower_to_original[self.target_column.lower()]
            return df.rename(columns={original_name: self.target_column})
        if "label" in lower_to_original:
            original_name = lower_to_original["label"]
            return df.rename(columns={original_name: self.target_column})

        raise ValueError(f"Target '{self.target_column}' o 'label' non trovato.")

    def _remove_repeated_header_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        label_as_str = df[self.target_column].astype(str).str.strip()
        df = df[label_as_str.str.lower() != self.target_column.lower()].copy()
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
            print(f" • Colonne metadata rimosse ({len(columns_to_drop)})")
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
            print(f" • Righe rimosse per NaN/inf: {removed}")
        return df

    def _encode_binary_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        raw_labels = df[self.target_column]

        # Se già numeriche {0,1}
        numeric_labels = pd.to_numeric(raw_labels, errors="coerce")
        non_null_numeric = numeric_labels.dropna()
        if not non_null_numeric.empty:
            unique_values = set(non_null_numeric.unique())
            if unique_values.issubset({0, 1}):
                df[self.target_column] = numeric_labels.astype(np.int8)
                return df

        # Altrimenti testo (Benign=0, Resto=1)
        labels_as_str = raw_labels.astype(str).str.strip().str.lower()
        df[self.target_column] = np.where(labels_as_str == "benign", 0, 1).astype(np.int8)
        return df

    def _remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina le colonne con varianza zero (completamente inutili predittivamente)."""
        df = df.copy()
        varianze = df.var(numeric_only=True)
        colonne_costanti = varianze[varianze == 0].index
        
        if len(colonne_costanti) > 0:
            df = df.drop(columns=colonne_costanti, errors="ignore")
            print(f" • Feature costanti (varianza=0) rimosse: {len(colonne_costanti)}")
            
        return df

    def _remove_low_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina le feature la cui correlazione di Pearson assoluta col target è sotto la soglia."""
        df = df.copy()
        
        # Protezione: si assicura che il target sia effettivamente numerico prima di calcolare
        if self.target_column in df.columns and pd.api.types.is_numeric_dtype(df[self.target_column]):
            # Calcola correlazione e isola la riga del target (escludendo il target stesso = 1.0)
            corr_with_label = df.corr(numeric_only=True)[self.target_column].drop(self.target_column).abs()
            
            # Filtra le feature sotto la soglia specificata nel costruttore
            low_corr_features = corr_with_label[corr_with_label < self.correlation_threshold].index.tolist()
            
            if low_corr_features:
                df = df.drop(columns=low_corr_features, errors="ignore")
                print(f" • Feature rimosse per bassa correlazione (<{self.correlation_threshold}): {len(low_corr_features)}")
                
        return df