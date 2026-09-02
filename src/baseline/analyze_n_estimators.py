"""
Modulo UNICO per la giustificazione empirica di n_estimators, sia per la
classificazione (dataset reale CICIDS) sia per la regressione (dataset
sintetico) -- sostituisce analyze_classification_n_estimators.py e
consolida in un solo posto una logica prima sparsa su tre livelli diversi
di run_baseline.py (Optuna categorico dentro il tuning, un secondo Optuna
GridSampler di raffinamento senza warm_start, una scelta interattiva con
soglia di tolleranza) più questo stesso script per la sola classificazione.

RUOLO NELLA PIPELINE -- SOLA DIAGNOSTICA, MAI NEL HOT PATH:
    Questo script NON tocca tuning né feature selection. Li dà per fatti:
      - classificazione: legge gli iperparametri (incluso max_features, che
        determina la feature selection) da 'outputs_baseline/config_real.json',
        prodotto da run_baseline.py -- optuna_oob_hyperparameter_search;
      - regressione: legge gli iperparametri da
        'outputs_baseline/config_synthetic_regressor.json' -- di default
        sklearn (nessun tuning: il sintetico serve solo da stress-test di
        scalabilità, non a massimizzare l'R² -- vedi REGRESSOR_DEFAULT_HP in
        run_baseline.py), tranne n_estimators, che è ESATTAMENTE il valore
        che QUESTO script calcola: run_baseline.py lo rilegge da qui al giro
        successivo.
    Fa crescere SOLO n_estimators via warm_start, a parità di tutto il
    resto, e produce grafico + tabella. Il criterio di scelta resta la
    lettura VISIVA della curva (vedi sotto): questo script non scrive né
    sovrascrive alcun manifesto (l'aggiornamento di n_estimators nel
    manifesto, dopo aver letto il grafico, è manuale).

METODOLOGIA -- allineata all'esempio ufficiale scikit-learn:
    "OOB Errors for Random Forests"
    https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
    (The scikit-learn developers, BSD-3-Clause)

    Una sola foresta cresce incrementalmente con warm_start=True (si
    aggiungono alberi via via), non tante foreste indipendenti create da
    zero per ogni n_estimators: con fit indipendenti lo schema di seeding
    interno di scikit-learn assegna semi diversi ai singoli alberi a
    seconda del numero totale di stimatori richiesto, quindi la curva
    confronterebbe foreste leggermente diverse punto per punto, non la
    stessa foresta osservata a stadi di crescita diversi.

    UNA SOLA CURVA per task (non un confronto tra più configurazioni come
    nell'esempio ufficiale, che affianca diversi max_features): qui gli
    altri iperparametri sono già stati scelti altrove (tuning OOB per la
    classificazione, configurazione dichiarata a priori per la
    regressione) -- l'obiettivo di questo script è raffinare/giustificare
    solo n_estimators a valle di quella scelta, non riaprire lo spazio
    degli altri iperparametri.

    LETTURA: il criterio primario per scegliere n_estimators resta la
    lettura VISIVA del grafico (dove le curve corrette si appiattiscono).
    Nessun algoritmo di knee-detection automatico (rimosso in una versione
    precedente di questo script per la classificazione: il Kneedle
    semplificato segnala sistematicamente un ginocchio troppo basso su
    curve che saturano rapidamente). Come riferimento numerico supplementare
    resta il punto in cui la copertura OOB raggiunge il 100% (limite
    INFERIORE assoluto, non un target).

CORREZIONE DEL BIAS OOB A n_estimators BASSO -- necessaria per ENTRAMBI i
task, ma con un meccanismo di rilevamento diverso:

    CLASSIFICAZIONE: quando una riga non è mai out-of-bag per nessun
    albero, scikit-learn (verificato empiricamente su 1.6.1) riempie
    oob_decision_function_[riga] con [0, 0, ...] invece di NaN. Il segnale
    "riga mai OOB" è qui INEQUIVOCABILE: una riga davvero valutata ha
    sempre somma della decision function > 0 (almeno un voto), una riga
    mai valutata ha somma ESATTAMENTE zero. Basta quindi:
        valid_mask = oob_decision_function_.sum(axis=1) != 0

    REGRESSIONE: una predizione di regressione vera può valere esattamente
    0.0, indistinguibile dal riempimento di default -- il trucco della
    somma non è applicabile. Qui la copertura va ricostruita direttamente
    dagli INDICI DI BOOTSTRAP di ciascun albero (API privata di
    scikit-learn, sklearn.ensemble._forest, la stessa usata internamente
    da RandomForestRegressor per calcolare oob_prediction_): una riga è
    "valida" se esiste ALMENO un albero per cui quella riga non è stata
    campionata nel bootstrap (cioè è out-of-bag per quell'albero). Questo
    NON ricalcola oob_prediction_ da capo (costoso, ridondante): usa solo
    la ricostruzione degli indici per sapere QUALI righe di
    forest.oob_prediction_ (già calcolato da scikit-learn) sono affidabili
    e quali sono il riempimento di default.

    Entrambe le curve, naive e corretta, sono mostrate nel grafico -- la
    lettura visiva va fatta sulle curve corrette.

Uso:
    python -m src.baseline.analyze_n_estimators --task classifier
    python -m src.baseline.analyze_n_estimators --task regressor
    python -m src.baseline.analyze_n_estimators --task both   (default)
"""
import os
import json
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap
from sklearn.metrics import f1_score, accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.datasplitter import StratifiedDataSplitter
from src.shared.utilities.featureselection import CICIDSFeatureSelector
from src.shared.utilities.undersampling import undersample_majority_class

RANDOM_SEED = 123
TEST_SIZE = 0.2

# Griglia uniforme (stile esempio ufficiale sklearn: min/max/step), partendo
# da 5 apposta per rendere visibile il bias OOB a basso n_estimators invece
# di aggirarlo implicitamente. Stessa griglia per entrambi i task, per
# restare confrontabili a colpo d'occhio.
MIN_ESTIMATORS = 5
MAX_ESTIMATORS = 200
STEP_ESTIMATORS = 5
DEFAULT_GRID = list(range(MIN_ESTIMATORS, MAX_ESTIMATORS + 1, STEP_ESTIMATORS))

# ---------------------------------------------------------------------------
# CLASSIFICAZIONE -- dataset reale CICIDS
# ---------------------------------------------------------------------------
CLF_TARGET_COL = "Label"
CLF_TARGET_ROWS_PER_DAY = 100_000  # stesso valore di run_baseline.py
CLF_UNDERSAMPLING_RATIO = 1.0
CLF_CONFIG_PATH = os.path.join("outputs_baseline", "config_real.json")
# Sottocampione SOLO per questa diagnostica (non per run_baseline.py, che
# resta sul train set completo) -- stessa tecnica/motivazione di
# analyze_permutation_importance_config.py: la forma della curva OOB è
# governata dal numero di alberi, non dalla dimensione del dataset.
CLF_DIAGNOSTIC_SUBSAMPLE_SIZE = 100_000

# ---------------------------------------------------------------------------
# REGRESSIONE -- dataset sintetico
#
# Iperparametri: default sklearn tranne n_estimators (vedi REGRESSOR_DEFAULT_HP
# in run_baseline.py) -- nessun tuning, il sintetico serve solo da stress-test
# di scalabilità. Persistiti in REG_CONFIG_PATH, questo script si limita a
# leggerli.
#
# RICETTA DEL DATASET: letta anch'essa da REG_CONFIG_PATH (n_samples,
# n_features, noise), NON ridichiarata qui con valori più piccoli. La curva
# OOB va misurata sul dataset di PRODUZIONE esatto: un sottocampione
# diagnostico più piccolo darebbe una stima di n_estimators non trasferibile
# al modello che verrà davvero usato nei test di scalabilità -- stesso motivo
# per cui il tuning stesso è stato tolto dal percorso "dataset più piccolo".
# ---------------------------------------------------------------------------
REG_TARGET_COL = "Target"
# Stesso file già scritto da run_baseline.py per OGNI run sintetico di
# regressione (copia per-task, non un file dedicato al tuning): niente
# artefatto aggiuntivo da tenere sincronizzato con chi consuma già
# outputs_baseline/.
REG_CONFIG_PATH = os.path.join("outputs_baseline", "config_synthetic_regressor.json")


# ---------------------------------------------------------------------------
# Caricamento dati
# ---------------------------------------------------------------------------
def load_tuned_classifier_hp():
    """
    Legge gli iperparametri TUNATI (tranne n_estimators, che qui è la
    variabile indipendente) da config_real.json. Non decide nulla: si
    limita a leggere cosa optuna_oob_hyperparameter_search ha già scelto.
    """
    if not os.path.exists(CLF_CONFIG_PATH):
        raise FileNotFoundError(
            f"'{CLF_CONFIG_PATH}' non trovato: esegui prima run_baseline.py "
            f"(con tuning, non SKIP_TUNING) almeno una volta."
        )
    with open(CLF_CONFIG_PATH, "r") as f:
        config = json.load(f)
    hp = config.get("hyperparameters")
    if not hp:
        raise ValueError(f"'{CLF_CONFIG_PATH}' trovato ma senza sezione 'hyperparameters'.")
    print(f"[INFO] Iperparametri tunati letti da '{CLF_CONFIG_PATH}': {hp}")
    return hp


def prepare_classifier_dataset(tuned_hp):
    """
    Ricostruisce lo STESSO train set che run_baseline.py userebbe con questi
    iperparametri: campionamento ribilanciato per giorno, binarizzazione,
    split, preprocessing, under-sampling, feature selection con lo stesso
    max_features vincente del tuning -- poi un sottocampionamento SOLO
    diagnostico (vedi CLF_DIAGNOSTIC_SUBSAMPLE_SIZE).
    """
    data_folder = os.environ.get("DATASET_LOCAL_PATH", "./dataset_cache")
    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            f"Cartella dataset non trovata: '{data_folder}'. "
            f"Imposta DATASET_LOCAL_PATH come per run_baseline.py."
        )

    print(f"[1/4] Caricamento dati da '{data_folder}' (campionamento ribilanciato per giorno, "
          f"target ~{CLF_TARGET_ROWS_PER_DAY} righe/giorno)...")
    loader = RawCSVDataLoader(
        data_url=data_folder, dataset_seed=RANDOM_SEED,
        target_rows_per_day=CLF_TARGET_ROWS_PER_DAY,
    )
    df_raw = loader.load()

    print("[2/4] Binarizzazione + split stratificato + preprocessing...")
    preprocessor = CICIDSPreprocessor(target_column=CLF_TARGET_COL)
    splitter = StratifiedDataSplitter(target_column=CLF_TARGET_COL, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    df_binarized = preprocessor.binarize_target(df_raw)
    train_df, _ = splitter.split(df_binarized)
    train_df = preprocessor.process(train_df)

    print("[3/4] Under-sampling della classe maggioritaria (solo train set) + feature selection "
          f"(max_features='{tuned_hp.get('max_features', 'sqrt')}', dal tuning)...")
    train_df = undersample_majority_class(
        train_df, target_column=CLF_TARGET_COL,
        majority_class=0, minority_class=1,
        ratio=CLF_UNDERSAMPLING_RATIO, random_state=RANDOM_SEED,
    )
    fs = CICIDSFeatureSelector(
        target_column=CLF_TARGET_COL, rf_random_state=RANDOM_SEED,
        rf_max_features=tuned_hp.get("max_features", "sqrt"),
        reduce_multicollinearity=True, multicollinearity_distance_threshold=0.3,
        dendrogram_plot_path=f"feature_correlation_dendrogram_{tuned_hp.get('max_features', 'sqrt')}_n_est_diagnostica.png",
    )
    train_df = fs.fit_transform(train_df)

    if len(train_df) > CLF_DIAGNOSTIC_SUBSAMPLE_SIZE:
        print(f"[4/4] Sottocampionamento diagnostico: {len(train_df):,} -> "
              f"{CLF_DIAGNOSTIC_SUBSAMPLE_SIZE:,} righe (stratificato, "
              f"seed={RANDOM_SEED}).".replace(",", "."))
        train_df, _ = train_test_split(
            train_df, train_size=CLF_DIAGNOSTIC_SUBSAMPLE_SIZE,
            stratify=train_df[CLF_TARGET_COL], random_state=RANDOM_SEED,
        )
        train_df = train_df.reset_index(drop=True)
    else:
        print(f"[4/4] Train set ({len(train_df):,} righe) sotto la soglia di sottocampionamento: "
              f"uso tutte le righe disponibili.".replace(",", "."))

    X = train_df.drop(columns=[CLF_TARGET_COL]).to_numpy()
    y = train_df[CLF_TARGET_COL].to_numpy()
    return X, y


def load_tuned_regressor_hp():
    """
    Legge da REG_CONFIG_PATH sia gli iperparametri (tranne n_estimators, che
    qui è la variabile indipendente) sia la ricetta del dataset (n_samples,
    n_features, noise) -- prodotti da run_baseline.py. Analoga a
    load_tuned_classifier_hp(), stessa idea: non decide nulla, legge solo
    cosa è già stato deciso altrove.
    """
    if not os.path.exists(REG_CONFIG_PATH):
        raise FileNotFoundError(
            f"'{REG_CONFIG_PATH}' non trovato: esegui prima run_baseline.py "
            f"con dataset_type='synthetic' e tree_type='regressor' almeno una volta."
        )
    with open(REG_CONFIG_PATH, "r") as f:
        config = json.load(f)
    hp = config.get("hyperparameters")
    if not hp:
        raise ValueError(f"'{REG_CONFIG_PATH}' trovato ma senza sezione 'hyperparameters'.")
    dataset_recipe = {
        "n_samples": config.get("n_samples"),
        "n_features": config.get("n_features"),
        "noise": config.get("noise"),
    }
    if any(v is None for v in dataset_recipe.values()):
        raise ValueError(
            f"'{REG_CONFIG_PATH}' non contiene la ricetta completa del dataset "
            f"(n_samples/n_features/noise): {dataset_recipe}."
        )
    print(f"[INFO] Iperparametri letti da '{REG_CONFIG_PATH}': {hp}")
    print(f"[INFO] Ricetta dataset letta da '{REG_CONFIG_PATH}': {dataset_recipe}")
    return hp, dataset_recipe


def prepare_regressor_dataset(dataset_recipe):
    """
    Genera il dataset sintetico di regressione con la ricetta ESATTA di
    produzione (n_samples, n_features, noise letti da REG_CONFIG_PATH, non
    ridichiarati qui) -- Friedman #1 (Friedman 1991; Breiman 1996), stessa
    scelta di SyntheticDataLoader. Nessuna feature selection: sul sintetico
    la separazione segnale/rumore è nota a priori dal generatore (5 feature
    informative fisse, il resto rumore puro).
    """
    n_samples = dataset_recipe["n_samples"]
    n_features = dataset_recipe["n_features"]
    noise = dataset_recipe["noise"]
    print(f"[1/2] Generazione dataset sintetico di regressione (Friedman #1) "
          f"({n_samples:,} righe, {n_features} feature, 5 informative fisse, "
          f"noise={noise})...".replace(",", "."))
    loader = SyntheticDataLoader(
        task="regression",
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_seed=RANDOM_SEED,
        target_column=REG_TARGET_COL,
    )
    df = loader.load()

    print("[2/2] Split train/test (solo il train serve per la curva OOB)...")
    train_df, _ = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    X = train_df.drop(columns=[REG_TARGET_COL]).to_numpy()
    y = train_df[REG_TARGET_COL].to_numpy()
    return X, y


# ---------------------------------------------------------------------------
# Correzione del bias di copertura OOB
# ---------------------------------------------------------------------------
def _classifier_valid_mask(oob_decision):
    """Righe mai OOB: somma della decision function ESATTAMENTE zero."""
    return (oob_decision.sum(axis=1) != 0) & ~np.isnan(oob_decision).any(axis=1)


def _regressor_valid_mask(forest, n_samples):
    """
    Ricostruisce, dagli indici di bootstrap di ciascun albero (API privata
    sklearn.ensemble._forest, la stessa usata internamente da
    RandomForestRegressor per calcolare oob_prediction_), quali righe sono
    state OOB per almeno un albero -- necessario perché una predizione di
    regressione vera può valere 0.0, indistinguibile dal riempimento di
    default usato per le righe mai OOB (vedi docstring del modulo).
    """
    max_samples = forest.max_samples if forest.bootstrap else None
    n_samples_bootstrap = (
        _get_n_samples_bootstrap(n_samples, max_samples) if forest.bootstrap else n_samples
    )
    n_trees = len(forest.estimators_)
    in_bag_count = np.zeros(n_samples, dtype=np.int64)
    for tree in forest.estimators_:
        sampled_indices = _generate_sample_indices(tree.random_state, n_samples, n_samples_bootstrap)
        in_bag_count += (np.bincount(sampled_indices, minlength=n_samples) > 0)
    # Valida se NON è stata campionata da TUTTI gli alberi (cioè è stata
    # out-of-bag per almeno uno di essi).
    return in_bag_count < n_trees


# ---------------------------------------------------------------------------
# Curva OOB con warm_start (nucleo comune ai due task)
# ---------------------------------------------------------------------------
def compute_oob_curve(task, X, y, base_kwargs, grid, marker_n_estimators=None):
    """
    Fa crescere UNA foresta con warm_start=True lungo `grid`, calcolando ad
    ogni checkpoint le metriche OOB naive e corrette. `task` è 'classifier'
    o 'regressor'. Ritorna la lista di righe (una per checkpoint di griglia).
    """
    is_clf = task == "classifier"
    estimator_cls = RandomForestClassifier if is_clf else RandomForestRegressor

    print(f"\n[CURVA OOB] task='{task}', warm_start, griglia {grid[0]}..{grid[-1]} "
          f"(step {grid[1] - grid[0] if len(grid) > 1 else '-'}), altri iperparametri fissi: "
          f"{base_kwargs}")
    print("=" * 100)

    rf = estimator_cls(warm_start=True, oob_score=True, n_jobs=-1,
                        random_state=RANDOM_SEED, **base_kwargs)

    rows = []
    cumulative_time = 0.0
    for i, n in enumerate(grid, start=1):
        print(f"  [{i}/{len(grid)}] fit warm_start fino a n_estimators={n} ...", flush=True)
        rf.set_params(n_estimators=n)
        t0 = time.perf_counter()
        rf.fit(X, y)
        elapsed = time.perf_counter() - t0
        cumulative_time += elapsed
        print(f"  [{i}/{len(grid)}] completato in {elapsed:.2f}s (cumulato: {cumulative_time:.2f}s)", flush=True)

        if is_clf:
            oob_decision = rf.oob_decision_function_
            valid_mask = _classifier_valid_mask(oob_decision)
            oob_pred_naive = np.argmax(oob_decision, axis=1)
            metric_naive_primary = accuracy_score(y, oob_pred_naive)
            metric_naive_secondary = f1_score(y, oob_pred_naive, zero_division=0)
            if valid_mask.any():
                oob_pred_corr = np.argmax(oob_decision[valid_mask], axis=1)
                y_valid = y[valid_mask]
                metric_corr_primary = accuracy_score(y_valid, oob_pred_corr)
                metric_corr_secondary = f1_score(y_valid, oob_pred_corr, zero_division=0)
            else:
                metric_corr_primary = np.nan
                metric_corr_secondary = np.nan
        else:
            oob_pred = rf.oob_prediction_
            valid_mask = _regressor_valid_mask(rf, len(y))
            metric_naive_primary = r2_score(y, oob_pred)
            metric_naive_secondary = mean_squared_error(y, oob_pred)
            if valid_mask.any():
                metric_corr_primary = r2_score(y[valid_mask], oob_pred[valid_mask])
                metric_corr_secondary = mean_squared_error(y[valid_mask], oob_pred[valid_mask])
            else:
                metric_corr_primary = np.nan
                metric_corr_secondary = np.nan

        n_missing = int((~valid_mask).sum())
        coverage_pct = valid_mask.mean() * 100
        if n_missing > 0:
            print(f"     [OOB] n_estimators={n}: {n_missing} righe "
                  f"({100 - coverage_pct:.3f}%) senza copertura OOB, escluse dalla curva corretta.")

        rows.append(dict(
            n=n, coverage_pct=coverage_pct,
            primary_naive=metric_naive_primary, primary_corr=metric_corr_primary,
            secondary_naive=metric_naive_secondary, secondary_corr=metric_corr_secondary,
            cum_time=cumulative_time,
        ))

    # --- Tabella ---
    if is_clf:
        primary_label, secondary_label = "Accuracy", "F1"
    else:
        primary_label, secondary_label = "R2", "MSE"

    print(f"\n  {'n_est':<7} | {'copert.%':<9} | {primary_label + ' naive':<12} | "
          f"{primary_label + ' corr.':<12} | {secondary_label + ' naive':<12} | "
          f"{secondary_label + ' corr.':<12} | {'t cum(s)'}")
    print("  " + "-" * 96)
    for r in rows:
        pc = f"{r['primary_corr']:.5f}" if not np.isnan(r['primary_corr']) else "n/d"
        sc = f"{r['secondary_corr']:.5f}" if not np.isnan(r['secondary_corr']) else "n/d"
        marker = "  <-- valore di riferimento" if r["n"] == marker_n_estimators else ""
        print(f"  {r['n']:<7} | {r['coverage_pct']:<9.3f} | {r['primary_naive']:<12.5f} | "
              f"{pc:<12} | {r['secondary_naive']:<12.5f} | {sc:<12} | {r['cum_time']:8.2f}{marker}")

    fully_covered = [r for r in rows if r["coverage_pct"] >= 100.0 - 1e-9]
    first_full = fully_covered[0]["n"] if fully_covered else None
    print(f"\n  Copertura OOB 100% raggiunta per la prima volta a n_estimators={first_full}.")
    print("  PRIMA di quel punto, le curve naive e corrette DIVERGONO -- verificare nella "
          "tabella quanto è ampio lo scarto ai valori più bassi di n_estimators.")

    return rows, primary_label, secondary_label


def plot_oob_curve(rows, primary_label, secondary_label, task, marker_n_estimators, out_prefix):
    grid_vals = [r["n"] for r in rows]

    # --- Grafico a scala piena ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 11), dpi=150, sharex=True)

    ax1.plot(grid_vals, [r["primary_naive"] for r in rows], marker='o', markersize=3,
              color='#9ca3af', linewidth=1.2, linestyle=':', label=f'{primary_label} OOB naive')
    ax1.plot(grid_vals, [r["primary_corr"] for r in rows], marker='o', markersize=3,
              color='#2563eb', linewidth=2, label=f'{primary_label} OOB corretta')
    ax1.plot(grid_vals, [r["secondary_naive"] for r in rows], marker='s', markersize=3,
              color='#fca5a5', linewidth=1.2, linestyle=':', label=f'{secondary_label} OOB naive')
    ax1.plot(grid_vals, [r["secondary_corr"] for r in rows], marker='s', markersize=3,
              color='#dc2626', linewidth=2, label=f'{secondary_label} OOB corretta')
    if marker_n_estimators:
        ax1.axvline(x=marker_n_estimators, color='#16a34a', linestyle='--', linewidth=1.5,
                    label=f'n_estimators={marker_n_estimators} (valore di riferimento)')
    ax1.set_ylabel("Score OOB")
    ax1.set_title(f"Stabilizzazione OOB al crescere di n_estimators (warm_start) — {task}\n"
                   "leggi a occhio dove le curve piene si appiattiscono")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=7)

    ax2.plot(grid_vals, [r["coverage_pct"] for r in rows], marker='o', markersize=3,
              color='#7c3aed', linewidth=2)
    ax2.axhline(y=100, color='#9ca3af', linestyle=':', linewidth=1)
    ax2.set_ylabel("Copertura OOB (%)")
    ax2.grid(True, alpha=0.3)

    ax3.plot(grid_vals, [r["cum_time"] for r in rows], marker='o', markersize=3,
              color='#dc2626', linewidth=2)
    ax3.set_xlabel("n_estimators")
    ax3.set_ylabel("Tempo cumulato di training (s)")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = f"{out_prefix}_oob_curve_warmstart.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\n  Grafico salvato in: {out_path}")

    # --- Versione zoomata (stessa logica di analyze_classification_n_estimators.py:
    #     limiti dell'asse Y calcolati solo sull'ultimo 75% dei checkpoint, per
    #     non far schiacciare il plateau dalla salita iniziale) ---
    plateau_start_idx = len(rows) // 4
    plateau_rows = rows[plateau_start_idx:]
    primary_vals = [r["primary_corr"] for r in plateau_rows if not np.isnan(r["primary_corr"])]
    secondary_vals = [r["secondary_corr"] for r in plateau_rows if not np.isnan(r["secondary_corr"])]
    if not primary_vals or not secondary_vals:
        print("  [ZOOM] Nessun checkpoint con copertura valida nel plateau: grafico zoomato saltato.")
        return

    y_min_p, y_max_p = min(primary_vals), max(primary_vals)
    y_min_s, y_max_s = min(secondary_vals), max(secondary_vals)
    y_pad_p = (y_max_p - y_min_p) * 0.25 if y_max_p > y_min_p else abs(y_max_p) * 0.01 + 1e-6
    y_pad_s = (y_max_s - y_min_s) * 0.25 if y_max_s > y_min_s else abs(y_max_s) * 0.01 + 1e-6

    fig_zoom, (ax1z, ax2z) = plt.subplots(2, 1, figsize=(8, 8), dpi=150, sharex=True)
    ax1z.plot(grid_vals, [r["primary_corr"] for r in rows], marker='o', markersize=3,
              color='#2563eb', linewidth=1.5, label=f'{primary_label} OOB corretta')
    if marker_n_estimators:
        ax1z.axvline(x=marker_n_estimators, color='#16a34a', linestyle='--', linewidth=1.5,
                     label=f'n_estimators={marker_n_estimators}')
    ax1z.set_ylabel(f"{primary_label} OOB corretta")
    ax1z.set_ylim(y_min_p - y_pad_p, y_max_p + y_pad_p)
    ax1z.set_title(f"Stabilizzazione OOB — VERSIONE ZOOMATA — {task}\n"
                    "asse Y ristretto alla sola parte stabile (la salita iniziale esce dal basso)")
    ax1z.grid(True, alpha=0.3)
    ax1z.legend(loc='best', fontsize=7)

    ax2z.plot(grid_vals, [r["secondary_corr"] for r in rows], marker='s', markersize=3,
              color='#dc2626', linewidth=1.5, label=f'{secondary_label} OOB corretta')
    if marker_n_estimators:
        ax2z.axvline(x=marker_n_estimators, color='#16a34a', linestyle='--', linewidth=1.5)
    ax2z.set_xlabel("n_estimators")
    ax2z.set_ylabel(f"{secondary_label} OOB corretta")
    ax2z.set_ylim(y_min_s - y_pad_s, y_max_s + y_pad_s)
    ax2z.grid(True, alpha=0.3)
    ax2z.legend(loc='best', fontsize=7)

    fig_zoom.tight_layout()
    out_path_zoom = f"{out_prefix}_oob_curve_warmstart_zoom.png"
    fig_zoom.savefig(out_path_zoom)
    plt.close(fig_zoom)
    print(f"  Grafico ZOOMATO salvato in: {out_path_zoom}")


# ---------------------------------------------------------------------------
# Entry point per task
# ---------------------------------------------------------------------------
def run_classifier_analysis():
    print("\n" + "#" * 100)
    print("  TASK: CLASSIFICAZIONE (dataset reale CICIDS)")
    print("#" * 100)
    tuned_hp = load_tuned_classifier_hp()
    X, y = prepare_classifier_dataset(tuned_hp)

    bootstrap = bool(tuned_hp.get("bootstrap", True))
    if not bootstrap:
        raise ValueError(
            "La stima OOB richiede bootstrap=True (Breiman 2001, Definition 1.1): "
            f"config_real.json indica bootstrap={bootstrap}, incompatibile con questa diagnostica."
        )
    base_kwargs = dict(
        max_depth=tuned_hp.get("max_depth"),
        min_samples_split=int(tuned_hp.get("min_samples_split", 2)),
        max_features=tuned_hp.get("max_features", "sqrt"),
        criterion=tuned_hp.get("criterion", "gini"),
        class_weight=tuned_hp.get("class_weight"),
        bootstrap=True,
        max_samples=float(tuned_hp.get("max_samples", 1.0)),
    )
    tuned_n = int(tuned_hp["n_estimators"]) if tuned_hp.get("n_estimators") is not None else None
    grid = sorted(set(DEFAULT_GRID) | ({tuned_n} if tuned_n else set()))

    rows, primary_label, secondary_label = compute_oob_curve(
        "classifier", X, y, base_kwargs, grid, marker_n_estimators=tuned_n,
    )
    plot_oob_curve(rows, primary_label, secondary_label, "Classificazione (CICIDS)",
                    tuned_n, out_prefix="classifier")
    return rows


def run_regressor_analysis():
    print("\n" + "#" * 100)
    print("  TASK: REGRESSIONE (dataset sintetico)")
    print("#" * 100)
    hp, dataset_recipe = load_tuned_regressor_hp()
    X, y = prepare_regressor_dataset(dataset_recipe)

    bootstrap = bool(hp.get("bootstrap", True))
    if not bootstrap:
        raise ValueError(
            "La stima OOB richiede bootstrap=True (Breiman 2001, Definition 1.1): "
            f"'{REG_CONFIG_PATH}' indica bootstrap={bootstrap}, incompatibile con questa diagnostica."
        )
    # max_samples=None è il default sklearn (bootstrap sample_size=n_samples,
    # cioè bootstrap "pieno" classico) -- NON va convertito a float (crash su
    # float(None)); None è un valore legittimo, non un buco da riempire.
    raw_max_samples = hp.get("max_samples")
    base_kwargs = dict(
        max_depth=hp.get("max_depth"),
        min_samples_split=int(hp.get("min_samples_split", 2)),
        max_features=hp.get("max_features", 1 / 3),
        criterion=hp.get("criterion", "squared_error"),
        bootstrap=True,
        max_samples=float(raw_max_samples) if raw_max_samples is not None else None,
    )
    reference_n = int(hp["n_estimators"]) if hp.get("n_estimators") is not None else None
    grid = sorted(set(DEFAULT_GRID) | ({reference_n} if reference_n else set()))

    rows, primary_label, secondary_label = compute_oob_curve(
        "regressor", X, y, base_kwargs, grid, marker_n_estimators=reference_n,
    )
    plot_oob_curve(rows, primary_label, secondary_label, "Regressione (sintetico)",
                    reference_n, out_prefix="regressor")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Curva OOB (warm_start) per la giustificazione empirica di n_estimators."
    )
    parser.add_argument("--task", choices=["classifier", "regressor", "both"], default="both",
                         help="Quale task analizzare (default: both).")
    args = parser.parse_args()

    if args.task in ("classifier", "both"):
        run_classifier_analysis()
    if args.task in ("regressor", "both"):
        run_regressor_analysis()

    print("\n" + "=" * 100)
    print("  PROSSIMO PASSO")
    print("=" * 100)
    print("  1. Apri i grafici '*_oob_curve_warmstart.png' e leggi a occhio dove le curve")
    print("     piene (corrette) si appiattiscono -- unico criterio.")
    print("  2. Verifica che il valore scelto sia >= al n_estimators a cui la copertura OOB")
    print("     raggiunge il 100% (stampato sopra): sotto quella soglia la stima OOB è")
    print("     matematicamente incompleta, a prescindere da come appare la curva.")
    print("  3. Questo script NON scrive alcun manifesto. Se decidi di cambiare n_estimators:")
    print("     - classificazione: rilancia run_baseline.py con SKIP_TUNING (riusa gli altri")
    print("       iperparametri) e modifica manualmente 'n_estimators' in config_real.json,")
    print("       oppure rilancia il tuning includendo il nuovo valore nella griglia Optuna;")
    print("     - regressione: stesso principio, sul manifesto")
    print("       'config_synthetic_regressor.json' (o rilancia il tuning includendo")
    print("       il nuovo valore in REGRESSOR_SEARCH_N_ESTIMATORS, in run_baseline.py).")
    print("     In entrambi i casi cita in relazione:")
    print("     (a) l'esempio ufficiale scikit-learn per il metodo warm_start,")
    print("     (b) la correzione del bias di copertura OOB spiegata in questo script")
    print("         (sum==0 per la classificazione, ricostruzione indici di bootstrap per")
    print("         la regressione),")
    print("     (c) Breiman 2001 Sec. 3.1 per la stima OOB come base del criterio stesso.")


if __name__ == "__main__":
    main()