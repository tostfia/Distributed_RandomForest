import os
import json
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import confusion_matrix, classification_report

from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
from src.shared.utilities.datasplitter import StratifiedDataSplitter
from src.shared.utilities.featureselection import CICIDSFeatureSelector


def run_baseline():
    print("=====================================================")
    print("      AVVIO BASELINE NON DISTRIBUITA (LOCAL)         ")
    print("=====================================================\n")

    # 1. Lettura configurazione dal config.json
    config_path = "config.json"
    config = {}

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Estrazione parametri generali
    dataset_type = config.get("dataset_type", "real")
    dataset_url = config.get("dataset_path", "./dataset_completo")
    dataset_seed = config.get("dataset_seed", 123)

    # Estrazione iperparametri
    hp = config.get("hyperparameters", {})
    n_estimators = hp.get("n_estimators", 100)
    max_depth = hp.get("max_depth", 10)
    max_samples = hp.get("max_samples", 0.2)
    class_weight = hp.get("class_weight", "balanced")

    # ---------------------------------------------------------
    # FASE 1: ETL
    # ---------------------------------------------------------
    print(">>> FASE 1: Estrazione e Pulizia Dati (ETL)")
    etl_start_time = time.perf_counter()

    if dataset_type == "real":
        print(f" • Tipo Dataset: Reale (Campionamento 1%, Seed: {dataset_seed})")

        loader = RawCSVDataLoader(
            data_url=dataset_url,
            sample_fraction=0.01,
            dataset_seed=dataset_seed
        )

        df_raw = loader.load()

        preprocessor = CICIDSPreprocessor()
        df_clean = preprocessor.process(df_raw)

    else:
        print(" • Tipo Dataset: Sintetico (Stress Test Task 2)")

        loader = SyntheticDataLoader(
            n_samples=100000,
            random_seed=dataset_seed
        )

        df_clean = loader.load()

    etl_time = time.perf_counter() - etl_start_time

    print(f"\n[OK] ETL completato in {etl_time:.4f} secondi.")
    print(f" • Dimensioni Dataset finale: {df_clean.shape}")

    print("\nDistribuzione Label dataset finale:")
    print(df_clean["Label"].value_counts())
    print("\nDistribuzione Label percentuale:")
    print(df_clean["Label"].value_counts(normalize=True) * 100)

    # ---------------------------------------------------------
    # FASE 2: SPLIT TRAIN/TEST
    # ---------------------------------------------------------
    if dataset_type == "real":
        # Nel notebook reale: test_size = 0.2
        test_size = 0.2
    else:
        # Nel notebook sintetico: test_size = 0.1
        test_size = 0.1

    splitter = StratifiedDataSplitter(
        target_column="Label",
        test_size=test_size,
        random_state=dataset_seed
    )

    train_df, test_df = splitter.split(df_clean)

    #Feature selection per dataset reale
    
    if dataset_type == "real":
        print("\n>>> Feature Selection sul solo Train Set")

        feature_selector = CICIDSFeatureSelector(
            target_column="Label",
            correlation_threshold=0.05
        )

        train_df = feature_selector.fit_transform(train_df)
        test_df = feature_selector.transform(test_df)

        print(f" • Dimensione Train dopo Feature Selection: {train_df.shape}")
        print(f" • Dimensione Test dopo Feature Selection:  {test_df.shape}")


    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]

    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"]

    # ---------------------------------------------------------
    # FASE 3: MODELLO E CROSS-VALIDATION
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("  INIZIALIZZAZIONE WORKFLOW: CONFIGURAZIONE METRICHE E VERIFICA MATRICI")
    print("=" * 70)
    print(f"  Features estratte per il training (colonne): {X_train.shape[1]}")
    print(f"  Volume totale campioni di Train (righe):     {X_train.shape[0]:,}".replace(",", "."))
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_samples=max_samples,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=dataset_seed
    )

    if dataset_type == "real":
        # Notebook reale:
        # kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_folds = 5
        cv_strategy = KFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=dataset_seed
        )
        scoring_metrics = ["accuracy", "precision", "recall", "f1"]

    else:
        # Notebook sintetico:
        # cross_validate(rf, X_tr, y_tr, cv=10, scoring=[...])
        cv_folds = 10
        cv_strategy = cv_folds
        scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    print(f"\n[1/4] Calcolo della Cross-Validation a {cv_folds} Fold in corso...")

    start_cv = time.perf_counter()

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv_strategy,
        scoring=scoring_metrics,
        return_train_score=False
    )

    tempo_totale_cv = time.perf_counter() - start_cv
    tempo_medio_fold = tempo_totale_cv / cv_folds

    print("Calcolo Cross-Validation completato con successo.")

    # ---------------------------------------------------------
    # FASE 4: FIT FINALE
    # ---------------------------------------------------------
    print("\n[2/4] Fitting finale sull'intero dataset di Train per estrazione T_seq...")

    start_train_finale = time.perf_counter()
    model.fit(X_train, y_train)
    t_seq = time.perf_counter() - start_train_finale

    # ---------------------------------------------------------
    # FASE 5: INFERENZA SU TEST SET
    # ---------------------------------------------------------
    print("\n[3/4] Benchmarking dei tempi di inferenza su massa dati (Test Set)...")

    start_inferenza = time.perf_counter()
    y_pred = model.predict(X_test)
    tempo_inferenza_totale = time.perf_counter() - start_inferenza

    print("Addestramento finale e test di latenza completati.")

    print("\n[TEST SET] Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    print("\n[TEST SET] Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # ---------------------------------------------------------
    # FASE 6: REPORT
    # ---------------------------------------------------------
    print("\n" + "═" * 75)
    print("                 REPORT ESTESO DI VALIDAZIONE E BENCHMARK")
    print("═" * 75)

    print("\n1. ANALISI DETTAGLIATA ITERAZIONE PER ITERAZIONE (PER FOLD)")
    print("-" * 75)

    for i in range(cv_folds):
        msg = (
            f"  [Fold {i + 1}/{cv_folds}] -> "
            f"Acc: {cv_results['test_accuracy'][i] * 100:.2f}% | "
            f"Prec: {cv_results['test_precision'][i] * 100:.2f}% | "
            f"Rec: {cv_results['test_recall'][i] * 100:.2f}% | "
            f"F1: {cv_results['test_f1'][i] * 100:.2f}% | "
        )

        if "test_roc_auc" in cv_results:
            msg += f"ROC-AUC: {cv_results['test_roc_auc'][i] * 100:.2f}% | "

        msg += f"Tempo Fit: {cv_results['fit_time'][i]:.3f}s"

        print(msg)

    print("\n2. METRICHE PREVENTIVE AGGREGATE (MEDIE VIRTUALI +/- DEVIAZIONE STANDARD)")
    print("-" * 75)

    print(
        f"  ACCURATEZZA GLOBALE:      "
        f"{cv_results['test_accuracy'].mean() * 100:.2f} % "
        f"(+/- {cv_results['test_accuracy'].std() * 100:.2f}%)"
    )

    print(
        f"  PRECISION MEDIA:          "
        f"{cv_results['test_precision'].mean() * 100:.2f} % "
        f"(+/- {cv_results['test_precision'].std() * 100:.2f}%)"
    )

    print(
        f"  RECALL MEDIA (Sens.):     "
        f"{cv_results['test_recall'].mean() * 100:.2f} % "
        f"(+/- {cv_results['test_recall'].std() * 100:.2f}%)"
    )

    print(
        f"  F1-SCORE MEDIO:           "
        f"{cv_results['test_f1'].mean() * 100:.2f} % "
        f"(+/- {cv_results['test_f1'].std() * 100:.2f}%)"
    )

    if "test_roc_auc" in cv_results:
        print(
            f"  ROC-AUC MEDIO:            "
            f"{cv_results['test_roc_auc'].mean() * 100:.2f} % "
            f"(+/- {cv_results['test_roc_auc'].std() * 100:.2f}%)"
        )

    print("\n3. DIAGNOSTICA TEMPORALE E PROFILAZIONE HARDWARE")
    print("-" * 75)

    print(f"  Tempo di Preprocessing (ETL):         {etl_time:.4f} secondi")
    print(f"  Tempo Totale della Cross-Validation:  {tempo_totale_cv:.4f} secondi")
    print(f"  Latenza Media di un Singolo Fold:     {tempo_medio_fold:.4f} secondi")
    print(f"  ADDESTRAMENTO FINALE (T_seq):         {t_seq:.4f} secondi  <-- BASELINE CLOUD")
    print(f"  Tempo di Inferenza su Test Set:       {tempo_inferenza_totale:.4f} secondi")

    if len(X_test) > 0:
        print(f"  Tempo Inferenza per Campione:         {tempo_inferenza_totale / len(X_test):.8f} secondi")
        print(f"  Throughput:                           {len(X_test) / tempo_inferenza_totale:.2f} samples/sec")

    print("═" * 75)


if __name__ == "__main__":
    run_baseline()