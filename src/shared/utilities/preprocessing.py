from typing import List, Optional
import numpy as np
import pandas as pd

class CICIDSPreprocessor:
    """
    Pipeline di Preprocessing specifica per il dataset di network traffic CIC-IDS2018.
    Configurata per emulare specularmente al millimetro la logica e i conteggi di Colab.
    """

    # Colonne di metadata NOTE per lo schema CIC-IDS2018/CICFlowMeter-V3 (78-84
    # feature, a seconda della versione/giorno del CSV) -- elenco esplicito,
    # verificato manualmente contro le feature realmente presenti nei CSV
    # usati in questo lavoro (nessuna feature comportamentale legittima,
    # es. Init Fwd Win Byts / Down/Up Ratio, contiene per caso una delle
    # parole chiave sotto come sottostringa). Serve da riferimento esplicito
    # per il controllo di sicurezza in _drop_metadata_columns: il filtro per
    # parola chiave resta il meccanismo di rimozione effettivo (più robusto a
    # differenze di naming tra versioni del CSV), ma qualunque colonna che
    # matcha una parola chiave SENZA comparire in questo elenco produce un
    # avviso esplicito invece di essere scartata silenziosamente -- così un
    # eventuale falso positivo (nome nuovo o inatteso che contiene "port",
    # "ip", ecc. per coincidenza) non passa inosservato in run futuri.
    KNOWN_METADATA_COLUMNS = frozenset({
        "flow id", "src ip", "source ip", "dst ip", "destination ip",
        "src port", "source port", "dst port", "destination port",
        "timestamp",
    })

    def __init__(
        self,
        target_column: str = "Label",
        drop_metadata_columns: bool = True,
        drop_invalid_rows: bool = True,
        add_engineered_features: bool = True,
        metadata_keywords: Optional[List[str]] = None,
    ):
        self.target_column = target_column
        self.drop_metadata_columns = drop_metadata_columns
        self.drop_invalid_rows = drop_invalid_rows
        self.add_engineered_features = add_engineered_features
        
        self.metadata_keywords = metadata_keywords or [
            "timestamp",
            "flow id",
            "ip",
            "port",
            "mac",
        ]

    def binarize_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Binarizzazione del target (Benign=0, rest=1). La rimozione delle
        righe di intestazione spuria avviene a monte, in
        RawCSVDataLoader (vedi nota sotto).
        Eseguire sul dataset intero prima dello split per evitare crash sulle classi rare.
        """
        print("Pre-binarizzazione Target CIC-IDS2018...")
        df = df.copy()

        # NOTA: la rimozione delle righe di intestazione spuria (righe con
        # Label=='Label', tipiche dei CSV CIC-IDS2018 con header ripetuto a
        # metà file) NON viene rifatta qui. È già gestita a monte da
        # RawCSVDataLoader._read_single_csv, in modo più robusto (usa
        # .str.strip() sul valore, quindi cattura qualunque variante di
        # spaziatura) di un controllo fisso su un elenco di stringhe
        # letterali. Rifarla qui con un meccanismo più debole era solo
        # codice ridondante, non un secondo strato di sicurezza reale.

        # Codifica del Target
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
        Rimozione metadati, feature ingegnerizzate e sanificazione NaN/inf.
        Da eseguire in modo indipendente sulle singole fette (Train e Test) dopo lo split.
        """
        df = df.copy()
        initial_shape = df.shape

        # 1. Rimozione Metadati non generalizzabili (Data Leakage)
        if self.drop_metadata_columns:
            df = self._drop_metadata_columns(df)

        # 2. Cast numerico
        df = self._convert_feature_columns_to_numeric(df)

        # 3. Feature ingegnerizzate -- DOPO il cast numerico (servono valori
        # float, non le stringhe con cui il caricamento in streaming legge
        # inizialmente le colonne) e PRIMA della sanificazione NaN/Inf, così
        # eventuali NaN/Inf che queste nuove feature possono introdurre (es.
        # divisione per un denominatore a zero) vengono gestiti dallo stesso
        # passo che già gestisce gli altri NaN/Inf del dataset, in modo
        # coerente e visibile nello stesso log.
        if self.add_engineered_features:
            df = self._add_engineered_features(df)

        # 4. Sanificazione finale per i Worker
        if self.drop_invalid_rows:
            df = self._drop_invalid_rows(df)

        print(f" • Pulizia completata. Shape: {initial_shape} -> {df.shape}")
        return df

    def _drop_metadata_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        columns_to_drop = [
            col for col in df.columns
            if col != self.target_column
            and any(k in str(col).lower() for k in self.metadata_keywords)
        ]

        # Verifica esplicita contro l'elenco noto (vedi KNOWN_METADATA_COLUMNS):
        # qualunque colonna catturata dal filtro per parola chiave ma NON
        # presente nell'elenco atteso genera un avviso visibile, invece di
        # essere scartata silenziosamente -- protezione contro falsi positivi
        # (es. una futura feature "Airport_Score" scartata solo perché
        # contiene "port") che altrimenti passerebbero inosservati.
        unexpected = [
            col for col in columns_to_drop
            if str(col).strip().lower() not in self.KNOWN_METADATA_COLUMNS
        ]
        if unexpected:
            print(f"   [ATTENZIONE] {len(unexpected)} colonna/e rimossa/e come metadata ma "
                  f"NON presente/i nell'elenco noto (KNOWN_METADATA_COLUMNS): {unexpected}. "
                  f"Verificare manualmente che non sia un falso positivo del filtro per "
                  f"parola chiave prima di procedere.")

        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors="ignore")
            print(f"   - Colonne metadata rimosse ({len(columns_to_drop)}): {columns_to_drop}")
        return df

    def _convert_feature_columns_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Unico punto della pipeline in cui avviene la conversione a numerico
        (pd.to_numeric, errors="coerce"). PRIMA veniva fatta anche dentro
        RawCSVDataLoader, con un'esclusione leggermente diversa
        (["Label", "_capture_day"] invece del solo target_column) --
        doppione rimosso: la tipizzazione è ora una responsabilità unica di
        questo preprocessor, chiamato sempre a valle del loader.
        """
        df = df.copy()
        feature_columns = df.columns.difference([self.target_column])
        df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
        return df

    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Due feature derivate, calcolabili dalle colonne CICFlowMeter già
        presenti (nessun bisogno di riprocessare pcap grezzi), pensate per
        catturare informazione comportamentale non già presente nelle
        colonne singole prese separatamente:

          - Flow/Fwd IAT CV (coefficiente di variazione, std/mean): misura
            quanto è irregolare il ritmo di un flusso INDIPENDENTEMENTE da
            quanto è veloce -- un flusso lento e regolare e uno veloce e
            regolare hanno CV simile pur avendo media/deviazione standard
            assolute molto diverse. Statistica nota in letteratura sul
            traffic classification come indicatore di "burstiness".

          - SYN_ACK_Ratio: rapporto tra conteggio di pacchetti SYN e ACK nel
            flusso. Un rapporto anomalo (troppi SYN rispetto agli ACK di
            risposta) è il segnale da manuale per identificare uno SYN
            flood/half-open scan -- un pattern che i conteggi assoluti presi
            singolarmente non isolano altrettanto direttamente. Il +1 al
            denominatore evita la divisione per zero senza dover scartare
            righe (a differenza delle due feature CV sopra, che possono
            produrre NaN se il denominatore è zero -- gestito dalla
            sanificazione NaN/Inf subito dopo questo passo).

        Le colonne sorgente sono verificate esplicitamente prima del calcolo:
        se lo schema del CSV dovesse cambiare (es. una versione diversa di
        CICFlowMeter con nomi di colonna diversi), questo metodo solleva un
        errore chiaro invece di fallire silenziosamente o produrre colonne
        vuote.
        """
        required_columns = {
            "Flow IAT Std", "Flow IAT Mean", "Fwd IAT Std", "Fwd IAT Mean",
            "SYN Flag Cnt", "ACK Flag Cnt",
        }
        missing = required_columns - set(df.columns)
        if missing:
            raise KeyError(
                f"Colonne necessarie per le feature ingegnerizzate assenti nel DataFrame: "
                f"{sorted(missing)}. Verificare lo schema del CSV sorgente (potrebbe essere "
                f"cambiato rispetto a quello atteso da CICFlowMeter-V3) prima di procedere."
            )

        df = df.copy()

        df["Flow IAT CV"] = df["Flow IAT Std"] / df["Flow IAT Mean"].replace(0, np.nan)
        df["Fwd IAT CV"] = df["Fwd IAT Std"] / df["Fwd IAT Mean"].replace(0, np.nan)
        df["SYN_ACK_Ratio"] = df["SYN Flag Cnt"] / (df["ACK Flag Cnt"] + 1)

        n_nan_iat_cv = int(df["Flow IAT CV"].isna().sum() + df["Fwd IAT CV"].isna().sum())
        print(f"   - Feature ingegnerizzate aggiunte (3): 'Flow IAT CV', 'Fwd IAT CV', "
              f"'SYN_ACK_Ratio'")
        if n_nan_iat_cv > 0:
            print(f"     [NOTA] {n_nan_iat_cv} valori NaN introdotti dalle due feature CV "
                  f"(denominatore Mean pari a zero -- tipicamente flussi con un solo "
                  f"pacchetto). Verranno rimossi dalla sanificazione NaN/Inf subito dopo: "
                  f"il conteggio 'Righe rimosse per NaN/inf' qui sotto include anche questo "
                  f"contributo, non solo NaN/Inf già presenti nel dataset originale.")

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