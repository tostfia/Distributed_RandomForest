from typing import Dict, List, Optional
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

try:
    from sklearn.ensemble._forest import _generate_sample_indices
except ImportError:
    _generate_sample_indices = None


class CICIDSFeatureSelector:
    """
    Feature selector per CIC-IDS2018.

    Criterio unico: OOB permutation importance, fedele a Breiman (2001),
    Section 10 ("Exploring the random forest mechanism"). Si addestra una
    foresta preliminare, e per ciascun albero si permuta una feature alla
    volta SOLO sui suoi campioni out-of-bag, misurando l'aumento percentuale
    del tasso di errore rispetto al baseline OOB (con tutte le feature
    intatte) — esattamente la metrica di Figure 4-6 del paper. Cattura anche
    importanza non-lineare e di interazione, perché usa il modello stesso
    (e i suoi OOB, non un validation set esterno) come strumento di misura —
    a differenza di un filtro per correlazione lineare, che scarterebbe
    feature predittive solo per interazione o non linearmente.
    """

    def __init__(
        self,
        target_column: str = "Label",
        importance_threshold: float = 0.0,
        rf_n_estimators: int = 100,
        rf_max_features="sqrt",
        rf_max_depth: Optional[int] = None,
        rf_random_state: int = 123,
        n_jobs: Optional[int] = None,
    ):
        self.target_column = target_column

        # Parametri della foresta PRELIMINARE usata solo per stimare
        # l'importanza — indipendente dagli iperparametri della foresta
        # finale (che a questo punto della pipeline non sono ancora stati
        # scelti: il tuning avviene DOPO la feature selection). n_estimators
        # più basso di quello usato da Breiman (1000, Section 10) per
        # contenere i tempi su CPU, ma comunque sufficiente per stime OOB
        # stabili; max_features='sqrt' segue la convenzione standard per
        # classificazione (vicina a int(log2(M)+1) usato da Breiman).
        self.importance_threshold = importance_threshold
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_features = rf_max_features
        self.rf_max_depth = rf_max_depth
        self.rf_random_state = rf_random_state

        # Default: tutti i core MENO UNO, non -1 (= tutti i core). Lascia
        # deliberatamente margine di CPU/RAM al sistema operativo e alle
        # altre applicazioni durante il fit e la permutation importance,
        # invece di saturare la macchina. Passa n_jobs=-1 esplicitamente se
        # preferisci usare tutti i core (più veloce, meno margine libero).
        if n_jobs is None:
            cpu_count = os.cpu_count() or 2
            n_jobs = max(1, cpu_count - 1)
        self.n_jobs = n_jobs

        self.columns_to_drop_: List[str] = []
        self.feature_summary_: Dict[str, List[str]] = {}
        # Series (indice = nome feature) con l'aumento percentuale medio OOB
        # del tasso di errore, ordinata crescente. Utile per ispezione/plot.
        self.importance_scores_: Optional[pd.Series] = None

    def fit(self, train_df: pd.DataFrame) -> "CICIDSFeatureSelector":
        if self.target_column not in train_df.columns:
            raise KeyError(f"Target '{self.target_column}' non trovato nel train set.")
        if _generate_sample_indices is None:
            raise ImportError(
                "sklearn.ensemble._forest._generate_sample_indices non disponibile in "
                "questa versione di scikit-learn: impossibile ricostruire gli indici "
                "in-bag/OOB per singolo albero, necessari per la permutation importance "
                "fedele a Breiman. Aggiorna/cambia versione di scikit-learn."
            )

        columns_to_drop = []

        # 1. Feature costanti calcolate solo sul train.
        # Una feature a varianza zero non porta MAI informazione, per nessun
        # modello — vale la pena tenerla come primo filtro, indipendente
        # dalla permutation importance (che su una feature costante darebbe
        # comunque importanza ~0, ma qui evitiamo pure di allenare/permutare
        # su qualcosa di palesemente inutile).
        variances = train_df.var(numeric_only=True)
        constant_features = variances[variances == 0].index.tolist()
        constant_features = [col for col in constant_features if col != self.target_column]
        columns_to_drop.extend(constant_features)

        print("=" * 60)
        print("   RIMOZIONE FEATURE COSTANTI (QUASI-CONSTANT FEATURES)")
        print("=" * 60)
        print(f"Colonne eliminate ({len(constant_features)}): {constant_features}")
        print("=" * 60)

        # 2. OOB permutation importance sulle feature rimanenti
        low_importance_features, extra_summary = self._select_by_oob_permutation_importance(
            train_df, exclude_columns=constant_features
        )
        columns_to_drop.extend(low_importance_features)

        # Rimuove duplicati mantenendo ordine
        self.columns_to_drop_ = list(dict.fromkeys(columns_to_drop))

        tutte_le_feature = [col for col in train_df.columns if col != self.target_column]
        feature_salvate = [col for col in tutte_le_feature if col not in self.columns_to_drop_]

        self.feature_summary_ = {
            "eliminate": self.columns_to_drop_,
            "eliminate_varianza_zero": constant_features,
            "salvate": feature_salvate,
            "selection_method": "permutation_importance",
            **extra_summary,
        }

        print(f"\n [FeatureSelector] Totale feature univoche contrassegnate per la rimozione: {len(self.columns_to_drop_)}")

        return self

    # ------------------------------------------------------------------
    # OOB permutation importance, fedele a Breiman (2001) Sec. 10
    # ------------------------------------------------------------------
    def _select_by_oob_permutation_importance(self, train_df: pd.DataFrame, exclude_columns: List[str]):
        feature_cols = [
            c for c in train_df.columns
            if c != self.target_column and c not in exclude_columns
        ]
        X = train_df[feature_cols].to_numpy()
        y = train_df[self.target_column].to_numpy()
        n_samples, n_features = X.shape

        print("=" * 60)
        print("   FILTRAGGIO FEATURE MEDIANTE OOB PERMUTATION IMPORTANCE")
        print("   (Breiman 2001, Sec. 10 — 'percent increase in misclassification")
        print("    rate as compared to the out-of-bag rate')")
        print("=" * 60)
        print(f" • Foresta preliminare: n_estimators={self.rf_n_estimators}, "
              f"max_features={self.rf_max_features}, max_depth={self.rf_max_depth}, "
              f"n_jobs={self.n_jobs}")
        print(f" • Dimensione train: {n_samples:,} righe x {n_features} feature "
              f"(dopo rimozione costanti). Su dataset di questa scala il fit e "
              f"soprattutto il ciclo di permutazione richiedono qualche minuto — "
              f"vedi il progresso qui sotto.".replace(",", "."))

        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_features=self.rf_max_features,
            max_depth=self.rf_max_depth,
            bootstrap=True,
            n_jobs=self.n_jobs,
            random_state=self.rf_random_state,
            verbose=1,
        )
        rf.fit(X, y)

        # Il ciclo sugli alberi è parallelizzato (joblib, già una dipendenza
        # di scikit-learn): ogni albero fa ~n_features chiamate predict()
        # indipendenti dagli altri alberi — con n_estimators=200 e ~60-70
        # feature erano 12.000-14.000 predict() in un ciclo Python seriale,
        # il vero collo di bottiglia (il solo fit della foresta è rapido).
        #
        # prefer="threads" invece del default a processi (loky): i thread
        # condividono la memoria del processo principale, quindi X/y e gli
        # alberi non vengono duplicati per worker — a differenza dei processi,
        # dove ogni worker è un interprete Python separato con il proprio
        # overhead di memoria. Più parco in RAM, leggermente meno veloce per
        # via del GIL (che scikit-learn rilascia comunque per gran parte
        # delle operazioni C interne di predict). self.n_jobs lascia già un
        # core libero per il sistema operativo (vedi __init__).
        #
        # La randomicità della permutazione usa tree.random_state (non un rng
        # condiviso): risultato deterministico e riproducibile
        # indipendentemente dall'ordine di esecuzione dei worker.
        #
        # verbose=10: joblib stampa "Done N out of M" via via che gli alberi
        # completano — indicatore di avanzamento su un ciclo che, a questa
        # scala di dataset, può durare diversi minuti.
        from joblib import Parallel, delayed

        print(f" • Avvio ciclo di permutazione: {self.rf_n_estimators} alberi x "
              f"{n_features} feature ≈ {self.rf_n_estimators * n_features:,} "
              f"predict() totali (parallelizzati su {self.n_jobs} thread)...".replace(",", "."))

        all_indices = np.arange(n_samples)

        def _process_tree(tree):
            in_bag = _generate_sample_indices(tree.random_state, n_samples, n_samples)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[in_bag] = False
            oob_idx = all_indices[oob_mask]

            if len(oob_idx) == 0:
                return None, "empty_oob"

            X_oob = X[oob_idx]
            y_oob = y[oob_idx]

            baseline_pred = tree.predict(X_oob)
            baseline_err = float(np.mean(baseline_pred != y_oob))

            if baseline_err == 0.0:
                return None, "zero_error"

            tree_rng = np.random.default_rng(tree.random_state)
            pct_increase = np.empty(n_features)
            for f_idx in range(n_features):
                perm_order = tree_rng.permutation(len(oob_idx))
                X_perm = X_oob.copy()
                X_perm[:, f_idx] = X_perm[perm_order, f_idx]

                perm_pred = tree.predict(X_perm)
                perm_err = float(np.mean(perm_pred != y_oob))
                pct_increase[f_idx] = (perm_err - baseline_err) / baseline_err * 100.0

            return pct_increase, None

        tree_results = Parallel(n_jobs=self.n_jobs, prefer="threads", verbose=10)(
            delayed(_process_tree)(tree) for tree in rf.estimators_
        )

        pct_increase_sum = np.zeros(n_features)
        pct_increase_count = np.zeros(n_features)
        trees_skipped_empty_oob = 0
        trees_skipped_zero_error = 0

        for pct_increase, skip_reason in tree_results:
            if skip_reason == "empty_oob":
                trees_skipped_empty_oob += 1
                continue
            if skip_reason == "zero_error":
                trees_skipped_zero_error += 1
                continue
            pct_increase_sum += pct_increase
            pct_increase_count += 1

        if trees_skipped_empty_oob or trees_skipped_zero_error:
            print(f" • [NOTA] Alberi esclusi dalla stima: {trees_skipped_empty_oob} con OOB vuoto, "
                  f"{trees_skipped_zero_error} con errore OOB baseline pari a zero.")

        mean_pct_increase = np.divide(
            pct_increase_sum, pct_increase_count,
            out=np.zeros_like(pct_increase_sum),
            where=pct_increase_count > 0,
        )

        importance_series = pd.Series(mean_pct_increase, index=feature_cols).sort_values()
        self.importance_scores_ = importance_series

        low_importance_features = importance_series[
            importance_series <= self.importance_threshold
        ].index.tolist()

        print(f" • Soglia di importanza: {self.importance_threshold} (percent increase OOB)")
        print(f" • Feature con importanza <= soglia: {len(low_importance_features)}")
        print(" • Le 10 feature meno importanti (permutandole, l'errore OOB non peggiora "
              "o addirittura migliora):")
        for feat, val in importance_series.head(10).items():
            print(f"     {feat:<40} {val:+7.2f}%")
        print(" • Le 10 feature più importanti:")
        for feat, val in importance_series.tail(10).items():
            print(f"     {feat:<40} {val:+7.2f}%")
        print("=" * 60)

        return low_importance_features, {
            "eliminate_bassa_importanza": low_importance_features,
            "importance_scores": importance_series.round(4).to_dict(),
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns_to_drop_ is None:
            raise RuntimeError("Devi chiamare fit() prima di transform().")

        df_transformed = df.drop(columns=self.columns_to_drop_, errors="ignore").copy()

        features_attuali = [col for col in df_transformed.columns if col != self.target_column]
        print(f" • Features predittive totali rimaste ({len(features_attuali)}): {features_attuali}")
        print(f" • Dimensione attuale del blocco (X + y): {df_transformed.shape}\n")

        return df_transformed

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df)